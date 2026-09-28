# Дообучаем bi-encoder на парах "запрос - выбранное объявление" из train_rest
# Лосс InfoNCE, негативы - остальные объявления в батче. В батче не должно быть
# одинаковых запросов/объявлений, иначе получаются ложные негативы
# Паддинг фиксированный, потому что на MPS переменные формы всё жутко тормозят :)
import argparse
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from common import WORK, clean_items
from dense_text import query_text, item_text

p = argparse.ArgumentParser()
p.add_argument("--base", default="intfloat/multilingual-e5-small")
p.add_argument("--out", default=str(WORK / "dense_e5s"))
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--temp", type=float, default=0.05)
p.add_argument("--qlen", type=int, default=48)
p.add_argument("--ilen", type=int, default=128)
p.add_argument("--max_per_query", type=int, default=30)
args = p.parse_args()

dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
rng = np.random.default_rng(0)

rest = clean_items(pd.read_parquet(WORK / "train_rest.parquet"))
rest = rest.drop_duplicates(["search_query", "search_infm_params_text", "item_id"])
rest = rest.sample(frac=1, random_state=0)
# режем популярные запросы, в бенчмарке в основном хвост
rest = rest[rest.groupby("search_query").cumcount() < args.max_per_query].reset_index(drop=True)
print("pairs", len(rest), flush=True)

q_txt = [query_text(q, p) for q, p in zip(rest.search_query, rest.search_infm_params_text)]
i_txt = [item_text(t, pr, d) for t, pr, d in zip(rest.item_title_raw, rest.item_infm_params_text,
                                                  rest.item_description_raw)]
q_key = rest.search_query.values
i_key = rest.item_id.values


def make_batches():
    order = rng.permutation(len(rest))
    batches, cur, used_q, used_i = [], [], set(), set()
    leftovers = []
    for idx in order:
        if q_key[idx] in used_q or i_key[idx] in used_i:
            leftovers.append(idx)
            continue
        cur.append(idx); used_q.add(q_key[idx]); used_i.add(i_key[idx])
        if len(cur) == args.bs:
            batches.append(cur); cur, used_q, used_i = [], set(), set()
    # что не влезло - вторым проходом
    for idx in leftovers:
        if q_key[idx] in used_q or i_key[idx] in used_i:
            continue
        cur.append(idx); used_q.add(q_key[idx]); used_i.add(i_key[idx])
        if len(cur) == args.bs:
            batches.append(cur); cur, used_q, used_i = [], set(), set()
    return batches


tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModel.from_pretrained(args.base).to(dev)
model.train()


def encode(texts, max_len):
    b = tok(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt").to(dev)
    out = model(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).float()
    return F.normalize((out * m).sum(1) / m.sum(1), dim=-1)


opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
total = args.epochs * (len(rest) // args.bs)
warm = int(0.05 * total)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))

step, t0, run_loss = 0, time.time(), 0.0
for ep in range(args.epochs):
    for bidx in make_batches():
        qe = encode([q_txt[i] for i in bidx], args.qlen)
        ie = encode([i_txt[i] for i in bidx], args.ilen)
        logits = qe @ ie.T / args.temp
        labels = torch.arange(len(bidx), device=dev)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        run_loss += loss.item()
        if step % 100 == 0:
            el = time.time() - t0
            print(f"step {step}/{total} loss {run_loss / 100:.4f} eta {el / step * (total - step) / 60:.1f}m", flush=True)
            run_loss = 0.0

model.save_pretrained(args.out)
tok.save_pretrained(args.out)
print("saved", args.out)
