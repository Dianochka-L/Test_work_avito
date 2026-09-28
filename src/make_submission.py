import argparse
import pandas as pd
from common import ROOT, WORK, load_bench_queries, load_bench_items
from ranker import train_ensemble, predict, top50, feat_cols

p = argparse.ArgumentParser()
p.add_argument("--val", default=str(WORK / "val_feats.parquet"))
p.add_argument("--bench", default=str(WORK / "bench_feats.parquet"))
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--out", default=str(ROOT / "answer.csv"))
args = p.parse_args()

val = pd.read_parquet(args.val)
bench = pd.read_parquet(args.bench)
assert feat_cols(val) == feat_cols(bench)

pred = top50(bench, predict(train_ensemble(val, args.seeds), bench))

bq = load_bench_queries()
corpus_ids = set(load_bench_items().item_id)
rows = []
for qid in bq.query_id:
    ids = list(dict.fromkeys(pred.get(qid, [])))[:50]
    assert all(i in corpus_ids for i in ids)
    rows.append((qid, " ".join(ids)))
ans = pd.DataFrame(rows, columns=["query_id", "answer"])
assert ans.query_id.is_unique and len(ans) == len(bq)
ans.to_csv(args.out, index=False)
print("saved", args.out)
