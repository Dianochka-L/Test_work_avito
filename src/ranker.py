import lightgbm as lgb
import numpy as np
import pandas as pd

NON_FEATS = {"qi", "ii", "label", "query_id", "item_id", "fold"}
ROUNDS = 1000
PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[50], lambdarank_truncation_level=60,
              learning_rate=0.03, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=10)


def feat_cols(df):
    return [c for c in df.columns if c not in NON_FEATS]


def train(df, rounds=ROUNDS, params=None):
    df = df.sort_values("qi", kind="stable")
    # запросы без единого релевантного кандидата ничему не учат
    df = df[df.groupby("qi").label.transform("max") > 0]
    grp = df.groupby("qi", sort=False).size().values
    ds = lgb.Dataset(df[feat_cols(df)], df.label, group=grp)
    return lgb.train(params or PARAMS, ds, num_boost_round=rounds)


def train_ensemble(df, seeds=5):
    return [train(df, params=dict(PARAMS, seed=s)) for s in range(seeds)]


def predict(models, df):
    return sum(m.predict(df[feat_cols(df)], num_threads=10) for m in models)


def top50(df, scores, k=50):
    d = pd.DataFrame({"query_id": df.query_id.values, "item_id": df.item_id.values, "s": scores})
    d = d.sort_values(["query_id", "s"], ascending=[True, False])
    return d.groupby("query_id").head(k).groupby("query_id").item_id.agg(list).to_dict()


def recall(pred, val):
    return float(np.mean([len(set(pred.get(q, [])) & set(g)) / len(g) for q, g in zip(val.query_id, val.rel)]))
