from __future__ import annotations

import logging
import os
import platform
from typing import Any, Dict, List

import numpy as np
from numpy.random import default_rng

# -----------------------------------------------------------------------------
# Logging configuration (shared by all modules)
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Global experiment constants
# -----------------------------------------------------------------------------
NODES: List[str] = [f"redis-{i}" for i in range(1, 5 + 1)]
REPLICAS: int = 100
TTL_SECONDS: int = 600
PIPELINE_SIZE_DEFAULT: int = 500
VALUE_BYTES: int = 0

NUM_REPEATS: int = 10  # n=10 for all stages

ZIPF_ALPHAS: List[float] = [1.1, 1.3, 1.5]
PIPELINE_SWEEP: List[int] = [50, 100, 200, 500, 1000]  # Stage A1 sweep

# Ablation sweep (Stage B): R=2 fixed, W fixed, vary T
ABLAT_THRESHOLDS: List[int] = [100, 200, 300, 500, 800]

SEED: int = 1337
MICROBENCH_OPS: int = 2_000_000
MICROBENCH_NUM_KEYS: int = 10_000

# -----------------------------------------------------------------------------
# RNG management (shared Zipf RNG)
# -----------------------------------------------------------------------------
NP_RNG = default_rng(SEED)


def reset_np_rng(seed: int) -> None:
    """Reset the shared NumPy RNG used for Zipf workloads."""
    global NP_RNG
    NP_RNG = default_rng(seed)


# -----------------------------------------------------------------------------
# Environment metadata & reproducibility logging
# -----------------------------------------------------------------------------
def runtime_env_metadata(repeats: int = NUM_REPEATS) -> Dict[str, Any]:
    import redis as _redis_pkg  # imported here to avoid hard dependency in other modules

    try:
        import hiredis  # noqa: F401

        hiredis_enabled = True
    except Exception:
        hiredis_enabled = False

    return {
        "seed": SEED,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "redis_py": _redis_pkg.__version__,
        "hiredis": hiredis_enabled,
        "nodes": ",".join(NODES),
        "replicas": REPLICAS,
        "ttl": TTL_SECONDS,
        "pipeline": PIPELINE_SIZE_DEFAULT,
        "value_bytes": VALUE_BYTES,
        "py_hashseed": os.environ.get("PYTHONHASHSEED", ""),
        "repeats": repeats,
    }


def log_reproducibility_info() -> None:
    phs = os.environ.get("PYTHONHASHSEED")
    if phs is None:
        logger.warning(
            "PYTHONHASHSEED not set. For stricter reproducibility: export PYTHONHASHSEED=123"
        )
    else:
        logger.info("PYTHONHASHSEED=%s", phs)

    try:
        import hiredis  # noqa: F401

        logger.info("hiredis: enabled")
    except Exception:
        logger.info("hiredis: not in use")


# Run once on import
log_reproducibility_info()
