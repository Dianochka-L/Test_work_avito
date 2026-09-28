# Локальная валидация, похожая на бенчмарк ... Берём группы (запрос, локация, фильтры) из train в локациях бенчмарка, подгоняем распределение длины запросов и пустых фильтров, каждый текст один раз
# Две половины на одной учим ранжировщик, на другой проверяем (и наоборот)
import numpy as np
import pandas as pd

from common import WORK, ITEM_COLS, load_train, load_bench_queries, load_bench_items, clean_items

N_VAL = 6000
SEEN_SHARE = 0.37  # столько текстов бенчмарка встречается в train
SEED = 42


def strat_key(df):
    nw = df.search_query.str.split().str.len().clip(upper=6)
    empty = (df.search_infm_params_text == "").astype(int)
    return nw.astype(str) + "_" + empty.astype(str)


def main():
    rng = np.random.default_rng(SEED)
    tr = load_train()
    bq = load_bench_queries()
    bi = load_bench_items()

    items_all = pd.concat([bi[ITEM_COLS], tr[ITEM_COLS]]).drop_duplicates("item_id")
    items_all = clean_items(items_all).reset_index(drop=True)
    items_all["in_bench"] = items_all.item_id.isin(set(bi.item_id))
    items_all.to_parquet(WORK / "items_all.parquet")

    key = ["search_query", "search_location_id", "search_infm_params_text"]
    tr["gid"] = tr.groupby(key, sort=False).ngroup()
    groups = (tr.groupby("gid")
                .agg(search_query=("search_query", "first"),
                     search_location_id=("search_location_id", "first"),
                     search_infm_params_text=("search_infm_params_text", "first"),
                     search_is_delivery_search=("search_is_delivery_search", "first"),
                     search_category=("search_category", "first"),
                     rel=("item_id", lambda s: list(dict.fromkeys(s))))
                .reset_index())
    cand = groups[groups.search_location_id.isin(set(bq.search_location_id))].copy()
    cand = cand.sample(frac=1, random_state=SEED).drop_duplicates("search_query")
    cand["sk"] = strat_key(cand)

    target = strat_key(bq).value_counts(normalize=True)
    parts = []
    for k, share in target.items():
        pool = cand[cand.sk == k]
        parts.append(pool.sample(min(len(pool), int(round(share * N_VAL))), random_state=SEED))
    val = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    val["query_id"] = [f"val{i:06d}" for i in range(len(val))]
    val["fold"] = np.arange(len(val)) % 2

    rest = tr[~tr.gid.isin(set(val.gid))]
    seen = val.search_query.isin(set(rest.search_query))
    n_drop = max(0, int(seen.sum() - SEEN_SHARE * len(val)))
    drop_texts = set(rng.choice(val.search_query[seen].values, n_drop, replace=False))
    rest = rest[~rest.search_query.isin(drop_texts)]
    val["text_seen"] = val.search_query.isin(set(rest.search_query))
    print("val", len(val), "seen share", round(val.text_seen.mean(), 3), "train rows left", len(rest))

    val.drop(columns=["gid", "sk"]).to_parquet(WORK / "val_queries.parquet")
    rest.drop(columns=["gid"]).to_parquet(WORK / "train_rest.parquet")


if __name__ == "__main__":
    main()
