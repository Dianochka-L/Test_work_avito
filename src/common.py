import re
from pathlib import Path
import numpy as np
import pandas as pd
import Stemmer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)

ITEM_COLS = ["item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
             "item_category_id", "item_microcat_id", "item_price", "item_rating",
             "item_rating_reviews_count", "item_location_id", "item_latitude",
             "item_longitude", "item_is_phone_hidden", "item_is_message_forbidden"]

_ru = Stemmer.Stemmer("russian")
_en = Stemmer.Stemmer("english")
_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")
_CYR_RE = re.compile(r"[а-я]")
_stem_cache = {}


def normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("ё", "е")
    return " ".join(_TOKEN_RE.findall(text))


def tokens(text):
    return normalize(text).split()


def stem_word(w):
    s = _stem_cache.get(w)
    if s is None:
        s = _ru.stemWord(w) if _CYR_RE.search(w) else _en.stemWord(w)
        _stem_cache[w] = s
    return s


def stem_text(text):
    return " ".join(stem_word(w) for w in tokens(text))


def load_train():
    return pd.read_parquet(DATA / "train.parquet")


def load_bench_queries():
    return pd.read_parquet(DATA / "benchmark_queries.parquet")


def load_bench_items():
    return pd.read_parquet(DATA / "benchmark_items.parquet")


def clean_items(df):
    # цена и координаты в parquet лежат строками
    df = df.copy()
    for c in ["item_price", "item_latitude", "item_longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in ["item_title_raw", "item_description_raw", "item_infm_params_text"]:
        df[c] = df[c].fillna("")
    return df


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
