from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from dense_text import query_text, item_text

DEV = "mps" if torch.backends.mps.is_available() else "cpu"




@torch.no_grad()
def encode(model_dir, texts, max_len, bs=256):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(DEV).eval()
    order = np.argsort([len(t) for t in texts])  # меньше паддинга
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for s in range(0, len(texts), bs):
        ids = order[s:s + bs]
        b = tok([texts[i] for i in ids], padding=True, truncation=True, max_length=max_len,
                return_tensors="pt").to(DEV)
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        out[ids] = F.normalize((h * m).sum(1) / m.sum(1), dim=-1).float().cpu().numpy()
    return out


def load_or_encode(model_dir, items, queries):
    # эмбеддинги объявлений кэшируем рядом с моделью, кодировать их долго
    cache = Path(model_dir) / "items_emb.npy"
    cache_ids = Path(model_dir) / "items_emb_ids.npy"
    have = {}
    if cache.exists():
        E = np.load(cache); ids = np.load(cache_ids, allow_pickle=True)
        have = dict(zip(ids, range(len(ids))))
    need = items[~items.item_id.isin(have.keys())]
    if len(need):
        print("encoding items", len(need), flush=True)
        txt = [item_text(t, p, d) for t, p, d in zip(need.item_title_raw, need.item_infm_params_text,
                                                     need.item_description_raw)]
        E_new = encode(model_dir, txt, 128)
        if have:
            E = np.concatenate([E, E_new]); ids = np.concatenate([ids, need.item_id.values])
        else:
            E, ids = E_new, need.item_id.values
        np.save(cache, E); np.save(cache_ids, ids)
        have = dict(zip(ids, range(len(ids))))
    item_emb = E[[have[i] for i in items.item_id]]
    q_emb = encode(model_dir, [query_text(q, p) for q, p in zip(queries.search_query,
                                                                queries.search_infm_params_text)], 48)
    return item_emb, q_emb


def history_emb(model_dir, items, train):
    # "история" объявления = среднее эмбеддингов запросов из train, по которым его выбирали
    tr = train[train.item_id.isin(set(items.item_id))][["search_query", "item_id"]].drop_duplicates()
    texts = tr.search_query.unique()
    E = encode(model_dir, [query_text(t, "") for t in texts], 48)
    pos = {t: i for i, t in enumerate(texts)}
    row = {i: r for r, i in enumerate(items.item_id.values)}
    H = np.zeros((len(items), E.shape[1]), dtype=np.float32)
    np.add.at(H, tr.item_id.map(row).values, E[tr.search_query.map(pos).values])
    norm = np.linalg.norm(H, axis=1, keepdims=True)
    return H / np.where(norm > 0, norm, 1)


def load_dense_models(model_dirs, items, queries, train):
    dense, qd = {}, {}
    for d in model_dirs:
        name = Path(d).name.replace("dense_", "")
        it_emb, q_emb = load_or_encode(d, items, queries)
        dense[name] = {"items": it_emb, "hist": history_emb(d, items, train)}
        qd[name] = q_emb
    return dense, qd
