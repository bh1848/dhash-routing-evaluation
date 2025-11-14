from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import NP_RNG

# CLF-like NASA log parser
_CLF_RE = re.compile(
    r'^(?P<host>\S+) \S+ \S+ \[(?P<time>.*?)\] '
    r'"(?P<method>\S+)\s+(?P<url>\S+)\s+(?P<proto>[^"]+)" '
    r"(?P<status>\d{3}) (?P<size>\S+)"
)


def load_logs_dataset(path: str) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Parse CLF-like NASA logs and use URL as key. Keep natural distribution.
    Returns (keys, metadata_by_key).
    """
    keys: List[str] = []
    meta: Dict[str, Any] = {}
    with open(path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            m = _CLF_RE.match(line.strip())
            if not m:
                continue
            url = m.group("url")
            status = int(m.group("status"))
            if not url or status == 0:
                continue
            keys.append(url)
            meta.setdefault(url, []).append(
                {
                    "host": m.group("host"),
                    "timestamp": m.group("time"),
                    "method": m.group("method"),
                    "protocol": m.group("proto"),
                    "status": status,
                    "size": m.group("size"),
                }
            )
    return keys, meta


def load_csv_dataset(
    path: str,
    key_column: str = "auctionid",
    natural_hot_threshold: Optional[int] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Load CSV and use key_column as key. If natural_hot_threshold is set, drop keys with count > threshold.
    """
    df = pd.read_csv(path)
    if key_column not in df.columns:
        raise ValueError(f"'{key_column}' column not found in CSV.")
    vc = df[key_column].value_counts()
    if natural_hot_threshold is None:
        keys = vc.index.sort_values().tolist()
    else:
        keys = vc[vc <= natural_hot_threshold].index.sort_values().tolist()
    meta = {str(k): df[df[key_column] == k].to_dict(orient="records") for k in keys}
    return [str(k) for k in keys], meta


def generate_zipf_workload(keys: List[Any], size: int, alpha: float = 1.1) -> List[Any]:
    """
    Truncated Zipf over ranks (1..n): p_i ∝ i^(-alpha). Avoid raw np.random.zipf modulo bias.
    """
    if not keys:
        raise ValueError("Key list is empty.")
    n = len(keys)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = ranks ** (-alpha)
    p = weights / weights.sum()
    idx = NP_RNG.choice(n, size=size, replace=True, p=p)
    return [keys[i] for i in idx]
