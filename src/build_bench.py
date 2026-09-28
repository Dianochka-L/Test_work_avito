import argparse
import numpy as np

from common import WORK, load_train, load_bench_queries, load_bench_items, clean_items
from pipeline import Corpus, LocationModel, MicrocatPrior, build_candidates
from dense_encode import load_dense_models

p = argparse.ArgumentParser()
p.add_argument("--dense", default="", help="папки с моделями через запятую")
p.add_argument("--out", default=str(WORK / "bench_feats.parquet"))
args = p.parse_args()

train = load_train()
bq = load_bench_queries()
bi = clean_items(load_bench_items()).sample(frac=1, random_state=0).reset_index(drop=True)

dense_items = dense_q = None
if args.dense:
    dense_items, dense_q = load_dense_models(args.dense.split(","), bi, bq, train)

corpus = Corpus(bi, train, dense_items)
locm = LocationModel(train)
mcp = MicrocatPrior().fit(train)
q_cnt = bq.search_query.map(train.search_query.value_counts()).fillna(0).values
df = build_candidates(bq, corpus, locm, mcp, q_dense=dense_q,
                      extra_query_feats={"q_train_cnt": np.log1p(q_cnt)})
df["query_id"] = bq.query_id.values[df.qi.values]
df["item_id"] = corpus.ids[df.ii.values]
df.to_parquet(args.out)
