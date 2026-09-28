# Recall@50 на валидации.. учимся на одном фолде, проверяем на другом и наоборот
# "clean" - запросы, у которых все релевантные и так лежат в корпусе бенчмарка (ничего не подмешивали), хорошая проверка что валидация не врёт
import sys
import pandas as pd
from common import WORK
from ranker import train_ensemble, predict, top50

df = pd.read_parquet(sys.argv[1])
seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
val = pd.read_parquet(WORK / "val_queries.parquet")
items = pd.read_parquet(WORK / "items_all.parquet", columns=["item_id", "in_bench"])
in_bench = set(items.item_id[items.in_bench])

pred = {}
for a, b in [(0, 1), (1, 0)]:
    te = df[df.fold == b]
    pred.update(top50(te, predict(train_ensemble(df[df.fold == a], seeds), te)))

val["r"] = [len(set(pred.get(q, [])) & set(g)) / len(g) for q, g in zip(val.query_id, val.rel)]
val["clean"] = val.rel.map(lambda l: all(i in in_bench for i in l))
cand = df.groupby("query_id").label.sum() / val.set_index("query_id").rel.map(len)
print("recall@50", round(val.r.mean(), 4), "| clean", round(val[val.clean].r.mean(), 4),
      "| candidates ceiling", round(cand.reindex(val.query_id).fillna(0).mean(), 4))
print(val.groupby("text_seen").r.mean().round(4).to_string())
