#!/usr/bin/env bash
# На M4 Pro это часа 2, большая часть уходит на дообучение моделей
set -euo pipefail
cd "$(dirname "$0")/src"
PY=../.venv/bin/python

$PY make_split.py

if [ ! -f ../work/dense_e5s/model.safetensors ]; then
  $PY finetune_dense.py --base intfloat/multilingual-e5-small --out ../work/dense_e5s
fi
if [ ! -f ../work/dense_tiny/model.safetensors ]; then
  $PY finetune_dense.py --base sergeyzh/rubert-tiny-turbo --out ../work/dense_tiny \
      --max_per_query 50 --epochs 2 --lr 1e-4
fi

$PY build_val.py   --dense ../work/dense_e5s,../work/dense_tiny --out ../work/val_feats.parquet
$PY build_bench.py --dense ../work/dense_e5s,../work/dense_tiny --out ../work/bench_feats.parquet
$PY make_submission.py --val ../work/val_feats.parquet --bench ../work/bench_feats.parquet
