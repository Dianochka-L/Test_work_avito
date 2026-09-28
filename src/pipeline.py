# Кандидаты + признаки для ранжировщика. Для каждого запроса считаем скоры по всему корпусу (блоками), берём топы разных
# источников, объединяем и для получившихся примерное 500 кандидатов собираем признаки
import gc
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from common import stem_text, normalize, haversine
from textindex import BM25Index, CharTfidf

# сколько кандидатов берём из каждого источника
TOPS = {"comb": 250, "dense": 150, "bm25": 100, "char": 100, "hist": 50, "dense_noloc": 30, "dhist": 50}


def topk_idx(mat, k):
    k = min(k, mat.shape[1] - 1)
    return np.argpartition(-mat, k, axis=1)[:, :k]


def nz(x):
    # делим на максимум по запросу, чтобы разные скоры можно было складывать
    m = x.max(1, keepdims=True)
    return x / np.where(m > 0, m, 1)


def nz_pos(x):
    return nz(np.maximum(x, 0))


def row_rank(mat, idx):
    vals = np.take_along_axis(mat, idx, 1)
    order = np.argsort(-vals, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(idx.shape[1])[None, :].repeat(idx.shape[0], 0), 1)
    return ranks


class LocationModel:
    def __init__(self, train, alpha=1.0):
        c = train.groupby(["search_location_id", "item_location_id"]).size()
        self.n = defaultdict(dict)
        for (s, i), v in c.items():
            self.n[s][i] = v
        self.tot = train.groupby("search_location_id").size().to_dict()
        self.glob = train.item_location_id.value_counts(normalize=True).to_dict()
        self.alpha = alpha
        # центр локации поиска - медиана координат выбранных объявлений
        lat = pd.to_numeric(train.item_latitude, errors="coerce")
        lon = pd.to_numeric(train.item_longitude, errors="coerce")
        t = pd.DataFrame({"s": train.search_location_id, "i": train.item_location_id, "lat": lat, "lon": lon})
        same = t[t.s == t.i].groupby("s")[["lat", "lon"]].median()
        center = t.groupby("s")[["lat", "lon"]].median()
        center.loc[same.index] = same
        self.center = center.to_dict("index")

    def logp_matrix(self, search_locs, item_locs_uni):
        M = np.zeros((len(search_locs), len(item_locs_uni)), dtype=np.float32)
        pos = {l: j for j, l in enumerate(item_locs_uni)}
        g = np.array([self.glob.get(l, 1e-7) for l in item_locs_uni], dtype=np.float64)
        for r, s in enumerate(search_locs):
            row = self.alpha * np.maximum(g, 1e-7)
            for i, v in self.n.get(s, {}).items():
                if i in pos:
                    row[pos[i]] += v
            M[r] = np.log(row / (self.tot.get(s, 0) + self.alpha))
        return M


class MicrocatPrior:
    def fit(self, train):
        t = train.drop_duplicates(["search_query", "search_infm_params_text", "item_microcat_id"])
        X_txt = (t.search_query.map(stem_text) + " "
                 + t.search_infm_params_text.map(normalize).str.replace(" ", "_p ") + "_p")
        self.vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=15, tol=None, random_state=0)
        self.clf.fit(self.vec.fit_transform(X_txt), t.item_microcat_id.values)
        self.classes = {c: j for j, c in enumerate(self.clf.classes_)}
        return self

    def proba(self, queries, params):
        X_txt = [stem_text(q) + " " + (normalize(p).replace(" ", "_p ") + "_p") for q, p in zip(queries, params)]
        return self.clf.predict_proba(self.vec.transform(X_txt)).astype(np.float32)


class Corpus:
    def __init__(self, items, train, dense_emb=None):
        it = items.reset_index(drop=True)
        print("corpus", len(it), flush=True)

        # история объявления тексты запросов из train, по которым его выбирали
        tr = train[train.item_id.isin(set(it.item_id))]
        hist = tr.groupby("item_id").search_query.agg(lambda s: " ".join(dict.fromkeys(s)))
        hist_n = it.item_id.map(tr.item_id.value_counts()).fillna(0).values.astype(np.float32)
        hist_txt = it.item_id.map(hist).fillna("")
        self.hist_pairs = set(zip(tr.search_query, tr.item_id))

        self.bt = BM25Index(k1=1.2, b=0.5).fit(it.item_title_raw.map(stem_text))
        self.bd = BM25Index(k1=1.2, b=0.75).fit(it.item_description_raw.map(stem_text))
        self.bp = BM25Index(k1=1.2, b=0.3).fit(it.item_infm_params_text.map(stem_text))
        self.bh = BM25Index(k1=1.2, b=0.3).fit(hist_txt.map(stem_text))
        self.ct = CharTfidf().fit(it.item_title_raw.map(normalize))
        self.ca = CharTfidf().fit((it.item_title_raw + " " + it.item_description_raw.str[:400]).map(normalize))
        print("indexes built", flush=True)
        self.dense = dense_emb or {}

        self.loc = it.item_location_id.values
        self.loc_uni, self.loc_code = np.unique(self.loc, return_inverse=True)
        self.lat = it.item_latitude.values.astype(np.float64)
        self.lon = it.item_longitude.values.astype(np.float64)
        self.microcat = it.item_microcat_id.values
        self.static = pd.DataFrame({
            "it_rating": it.item_rating.values,
            "it_reviews": np.log1p(it.item_rating_reviews_count.fillna(0).values),
            "it_price": np.log1p(it.item_price.clip(lower=0).values),
            "it_phone_hidden": it.item_is_phone_hidden.astype(float).values,
            "it_msg_forbid": it.item_is_message_forbidden.astype(float).values,
            "it_cat114": (it.item_category_id == 114).astype(float).values,
            "it_desc_len": np.log1p(it.item_description_raw.str.len().values),
            "it_title_len": it.item_title_raw.str.split().str.len().values,
            "it_hist_n": np.log1p(hist_n),
        }).astype(np.float32)
        self.title_stems = [set(s.split()) for s in it.item_title_raw.map(stem_text)]
        self.all_stems = [set(s.split()) for s in (it.item_title_raw + " " + it.item_infm_params_text + " "
                                                   + it.item_description_raw).map(stem_text)]
        self.ids = it.item_id.values


def build_candidates(queries, corpus, locm, mcp, q_dense=None, allowed_mask=None, chunk=256,
                     extra_query_feats=None):
    Q = queries.reset_index(drop=True)
    q_stem = Q.search_query.map(stem_text).tolist()
    q_norm = Q.search_query.map(normalize).tolist()
    p_stem = Q.search_infm_params_text.map(stem_text).tolist()

    s_locs = np.unique(Q.search_location_id.values)
    s_pos = {l: j for j, l in enumerate(s_locs)}
    LP = locm.logp_matrix(s_locs, corpus.loc_uni)
    mc_proba = mcp.proba(Q.search_query, Q.search_infm_params_text)
    mc_col = np.array([mcp.classes.get(m, -1) for m in corpus.microcat])

    out = []
    for st in range(0, len(Q), chunk):
        sl = slice(st, min(st + chunk, len(Q)))
        n = sl.stop - sl.start
        qs, ps = q_stem[sl], p_stem[sl]
        s_bt = corpus.bt.score(corpus.bt.query_matrix(qs)).toarray()
        s_bd = corpus.bd.score(corpus.bd.query_matrix(qs)).toarray()
        s_bp = corpus.bp.score(corpus.bp.query_matrix(qs)).toarray()
        s_bh = corpus.bh.score(corpus.bh.query_matrix(qs)).toarray()
        s_sp = corpus.bp.score(corpus.bp.query_matrix(ps)).toarray()  # фильтры поиска vs параметры
        s_ct = corpus.ct.score(q_norm[sl]).toarray()
        s_ca = corpus.ca.score(q_norm[sl]).toarray()
        dense_s, dhist_s = {}, {}
        for name, qe in (q_dense or {}).items():
            dense_s[name] = qe[sl] @ corpus.dense[name]["items"].T
            if corpus.dense[name].get("hist") is not None:
                dhist_s[name] = qe[sl] @ corpus.dense[name]["hist"].T
        s_de = np.mean([nz_pos(v) for v in dense_s.values()], 0) if dense_s else None

        sloc = Q.search_location_id.values[sl]
        lp = LP[[s_pos[l] for l in sloc]][:, corpus.loc_code]
        same = (corpus.loc[None, :] == sloc[:, None]).astype(np.float32)
        lp_c = np.maximum(lp, -12.0)
        mc = np.where(mc_col[None, :] >= 0, mc_proba[sl][:, np.maximum(mc_col, 0)], 0.0).astype(np.float32)

        # веса подобраны руками, это только для отбора кандидатов, дальше решает бустинг
        bm = s_bt + 0.5 * s_bd + 0.3 * s_bp
        loc_bonus = 0.08 * lp_c + 0.3 * same
        comb = (nz(bm) + nz(s_ca) + (1.5 * s_de if s_de is not None else 0)
                + 0.3 * nz(s_bh) + 0.5 * mc) + loc_bonus
        mask_pen = 0
        if allowed_mask is not None:
            mask_pen = np.where(allowed_mask(np.arange(sl.start, sl.stop)), 0, -1e9).astype(np.float32)

        parts = [topk_idx(comb + mask_pen, TOPS["comb"]),
                 topk_idx(nz(bm) + loc_bonus + mask_pen, TOPS["bm25"]),
                 topk_idx(nz(s_ca) + loc_bonus + mask_pen, TOPS["char"]),
                 topk_idx(nz(s_bh) + loc_bonus + mask_pen, TOPS["hist"])]
        for v in dense_s.values():
            parts.append(topk_idx(v + 0.05 * lp_c + 0.1 * same + mask_pen, TOPS["dense"]))
            parts.append(topk_idx(v + mask_pen, TOPS["dense_noloc"]))
        for v in dhist_s.values():
            parts.append(topk_idx(v + 0.05 * lp_c + 0.1 * same + mask_pen, TOPS["dhist"]))
        cand = np.concatenate(parts, 1)

        rows, width = [], 0
        for r in range(n):
            u = np.unique(cand[r])
            if allowed_mask is not None:
                u = u[mask_pen[r, u] == 0]
            rows.append(u); width = max(width, len(u))
        idx = np.stack([np.pad(u, (0, width - len(u)), constant_values=u[0]) for u in rows])
        valid = np.stack([np.arange(width) < len(u) for u in rows])

        def g(m):
            return np.take_along_axis(m, idx, 1)

        feats = {
            "bm_title": g(s_bt), "bm_desc": g(s_bd), "bm_par": g(s_bp), "bm_hist": g(s_bh),
            "bm_spar": g(s_sp), "ch_title": g(s_ct), "ch_all": g(s_ca),
            "loc_lp": g(lp), "loc_same": g(same), "mc_prob": g(mc), "comb": g(comb),
        }
        rank_keys = ["bm_title", "ch_all", "comb", "bm_hist"]
        for name, v in dense_s.items():
            feats[f"dense_{name}"] = g(v); rank_keys.append(f"dense_{name}")
        for name, v in dhist_s.items():
            feats[f"dhist_{name}"] = g(v); rank_keys.append(f"dhist_{name}")
        if s_de is not None:
            feats["dense_mean"] = g(s_de); rank_keys.append("dense_mean")
        for k in rank_keys:
            m = feats[k].max(1, keepdims=True)
            feats[k + "_rel"] = feats[k] / np.where(m > 0, m, 1)
            feats[k + "_rank"] = row_rank(np.where(valid, feats[k], -1e9),
                                          np.arange(width)[None].repeat(n, 0)).astype(np.float32)
        c = np.array([[locm.center.get(l, {}).get("lat", np.nan), locm.center.get(l, {}).get("lon", np.nan)]
                      for l in sloc])
        feats["dist_km"] = haversine(c[:, :1], c[:, 1:], corpus.lat[idx], corpus.lon[idx]).astype(np.float32)
        feats["mc_max"] = np.repeat(mc_proba[sl].max(1, keepdims=True), width, 1)

        df = pd.DataFrame({k: v.ravel() for k, v in feats.items()})
        df.insert(0, "qi", np.repeat(np.arange(sl.start, sl.stop), width))
        df.insert(1, "ii", idx.ravel())
        out.append(df[valid.ravel()].reset_index(drop=True))
        del s_bt, s_bd, s_bp, s_bh, s_sp, s_ct, s_ca, s_de, dense_s, dhist_s, lp, same, mc, comb
        gc.collect()
        print(f"  chunk {st}: cands/query {valid.sum(1).mean():.0f}", flush=True)

    df = pd.concat(out, ignore_index=True)

    # доля слов запроса, которые есть в заголовке / во всём тексте
    qset = [set(s.split()) for s in q_stem]
    qi, ii = df.qi.values, df.ii.values
    cov_t = np.zeros(len(df), np.float32)
    cov_a = np.zeros(len(df), np.float32)
    for k in range(len(df)):
        qs_ = qset[qi[k]]
        if qs_:
            cov_t[k] = len(qs_ & corpus.title_stems[ii[k]]) / len(qs_)
            cov_a[k] = len(qs_ & corpus.all_stems[ii[k]]) / len(qs_)
    df["cov_title"] = cov_t
    df["cov_all"] = cov_a
    qtext = Q.search_query.values
    df["hist_exact"] = np.fromiter(((qtext[a], corpus.ids[b]) in corpus.hist_pairs for a, b in zip(qi, ii)),
                                   dtype=np.float32, count=len(df))
    df = pd.concat([df, corpus.static.iloc[ii].reset_index(drop=True)], axis=1)
    df["q_nwords"] = np.array([len(s) for s in qset], np.float32)[qi]
    df["q_params_empty"] = (Q.search_infm_params_text.values == "").astype(np.float32)[qi]
    for k, v in (extra_query_feats or {}).items():
        df[k] = np.asarray(v, np.float32)[qi]
    return df
