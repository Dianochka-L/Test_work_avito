# Кандидаты и признаки для валидации (на них учится и проверяется ранжировщик)
# Корпус... корпус бенчмарка + релевантные объявления валидации
# причём каждому фолду видны только свои релевантные
import argparse

import numpy as np
import pandas as pd

from common import WORK
from pipeline import Corpus, LocationModel, MicrocatPrior, build_candidates
from dense_encode import load_dense_models

p = argparse.ArgumentParser()
p.add_argument("--dense", default="", help="папки с моделями через запятую")
p.add_argument("--out", default=str(WORK / "val_feats.parquet"))
args = p.parse_args()

items = pd.read_parquet(WORK / "items_all.parquet")
val = pd.read_parquet(WORK / "val_queries.parquet")
rest = pd.read_parquet(WORK / "train_rest.parquet")

rel_fold = {}
for f, rel in zip(val.fold, val.rel):
    for i in rel:
        rel_fold[i] = rel_fold.get(i, 0) | (1 << f)
corp_items = (items[items.in_bench | items.item_id.isin(rel_fold.keys())]
              .sample(frac=1, random_state=0).reset_index(drop=True))
item_bits = np.where(corp_items.in_bench.values, 3,
                     corp_items.item_id.map(rel_fold).fillna(0).astype(int).values)
q_bits = (1 << val.fold.values)


def allowed(rows):
    return (item_bits[None, :] & q_bits[rows][:, None]) > 0


dense_items = dense_q = None
if args.dense:
    dense_items, dense_q = load_dense_models(args.dense.split(","), corp_items, val, rest)

corpus = Corpus(corp_items, rest, dense_items)
locm = LocationModel(rest)
mcp = MicrocatPrior().fit(rest)

q_cnt = val.search_query.map(rest.search_query.value_counts()).fillna(0).values
df = build_candidates(val, corpus, locm, mcp, q_dense=dense_q, allowed_mask=allowed,
                      extra_query_feats={"q_train_cnt": np.log1p(q_cnt)})

rel_sets = [set(r) for r in val.rel]
df["label"] = np.fromiter((corpus.ids[i] in rel_sets[q] for q, i in zip(df.qi.values, df.ii.values)),
                          dtype=np.int8, count=len(df))
df["query_id"] = val.query_id.values[df.qi.values]
df["item_id"] = corpus.ids[df.ii.values]
df["fold"] = val.fold.values[df.qi.values]

hit = df.groupby("qi").label.sum().reindex(range(len(val))).fillna(0).values
print("candidate recall", round(np.mean(hit / val.rel.map(len).values), 4), "cands/query", len(df) // len(val))
df.to_parquet(args.out)
