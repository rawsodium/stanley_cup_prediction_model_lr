# Util functions to cache data from NHL API
import os
import pandas as pd

# makes a directory for cached data if it doesn't exist, and provides functions to read/write from it
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.csv")

def load_cache(key: str):
    path = _path(key)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def save_cache(key: str, df: pd.DataFrame):
    path = _path(key)
    df.to_csv(path, index=False)

def get_or_create(key: str, builder_fn):
    """
    Generic cache wrapper.

    builder_fn must return a pandas DataFrame.
    """
    cached = load_cache(key)
    if cached is not None:
        return cached

    df = builder_fn()

    if df is not None and not df.empty:
        save_cache(key, df)

    return df