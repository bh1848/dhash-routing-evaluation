from __future__ import annotations

import gc
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .algorithms import ConsistentHashing, DHash, RendezvousHashing, WeightedConsistentHashing
from .bench import benchmark_cluster, flush_databases, load_stddev, warmup_cluster
from .config import (
    ABLAT_THRESHOLDS,
    MICROBENCH_NUM_KEYS,
    MICROBENCH_OPS,
    NODES,
    NUM_REPEATS,
    PIPELINE_SIZE_DEFAULT,
    PIPELINE_SWEEP,
    REPLICAS,
    SEED,
    ZIPF_ALPHAS,
    reset_np_rng,
    runtime_env_metadata,
)
from .workloads import generate_zipf_workload, load_csv_dataset, load_logs_dataset

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Algorithm selection (CLI-level)
# -----------------------------------------------------------------------------
ALL_MODES: Tuple[str, ...] = ("Consistent Hashing", "Weighted CH", "Rendezvous", "D-HASH")

_ALIAS_MAP: Dict[str, str] = {
    "ch": "Consistent Hashing",
    "wch": "Weighted CH",
    "hrw": "Rendezvous",
    "dhash": "D-HASH",
}


def _parse_algos_list(algos_list: str) -> List[str]:
    items = [s.strip().lower() for s in algos_list.split(",") if s.strip()]
    resolved: List[str] = []
    for it in items:
        if it not in _ALIAS_MAP:
            raise ValueError(
                f"Unknown algorithm alias '{it}'. Use one of: {', '.join(_ALIAS_MAP)}"
            )
        resolved.append(_ALIAS_MAP[it])
    if not resolved:
        raise ValueError("Empty --algos_list after parsing.")
    return resolved


def resolve_algorithms(stage: str, algos: str, algos_list: str) -> List[str]:
    """
    stage ∈ {"pipeline", "zipf", "microbench", "ablation", "redistrib"}
    """
    if stage == "microbench":
        return ["CH", "D-HASH"]  # microbench uses CH vs D-HASH label schema
    if stage == "ablation":
        return ["D-HASH"]
    if stage == "redistrib":
        return ["CH", "WCH", "Rendezvous"]  # used only for report

    if algos == "all":
        return list(ALL_MODES)
    if algos == "minimal":
        return ["Consistent Hashing", "D-HASH"]
    if algos == "custom":
        return _parse_algos_list(algos_list)

    # auto (default):
    if stage == "pipeline":
        return ["Consistent Hashing", "D-HASH"]
    if stage == "zipf":
        return list(ALL_MODES)
    return list(ALL_MODES)


# -----------------------------------------------------------------------------
# Common helpers
# -----------------------------------------------------------------------------
def gc_collect() -> None:
    try:
        gc.collect()
    except Exception:
        pass


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    from statistics import mean, stdev

    return float(mean(xs)), float(stdev(xs))


def run_single_mode(
    keys: List[Any],
    mode_name: str,
    pipeline_size: int = PIPELINE_SIZE_DEFAULT,
    dhash_params: Optional[Dict[str, int]] = None,
) -> Tuple[float, float, float, float, float]:
    if mode_name == "Consistent Hashing":
        sh = ConsistentHashing(NODES, replicas=REPLICAS)
    elif mode_name == "Weighted CH":
        weights = {n: 1.0 + 0.1 * i for i, n in enumerate(NODES)}
        sh = WeightedConsistentHashing(NODES, weights, base_replicas=REPLICAS)
    elif mode_name == "Rendezvous":
        sh = RendezvousHashing(NODES)
    elif mode_name == "D-HASH":
        params = dhash_params or {"T": 50, "W": 1024}
        sh = DHash(
            NODES,
            hot_key_threshold=int(params["T"]),
            window_size=int(params["W"]),
        )
    else:
        raise ValueError(f"unknown mode: {mode_name}")

    flush_databases(NODES, flush_async=False)
    warmup_cluster(sh, keys)
    metrics = benchmark_cluster(keys, sh, pipeline_size=pipeline_size)

    thr = metrics["throughput_ops_s"]
    avg_ms = metrics["avg_ms"]
    p95_ms = metrics["p95_ms"]
    p99_ms = metrics["p99_ms"]
    sd = load_stddev(metrics["node_load"])

    logger.info(
        "[Result] mode=%s B=%d thr=%.1f ops/s avg=%.3fms p95=%.3fms p99=%.3fms load_sd=%.3f",
        mode_name,
        pipeline_size,
        thr,
        avg_ms,
        p95_ms,
        p99_ms,
        sd,
    )

    return thr, avg_ms, p95_ms, p99_ms, sd


# -----------------------------------------------------------------------------
# Stage A1: Pipeline sweep
# -----------------------------------------------------------------------------
def run_pipeline_sweep(
    name: str,
    keys: List[Any],
    out_csv: str,
    alpha: float = 1.5,
    sweep: Optional[List[int]] = None,
    repeats: int = NUM_REPEATS,
    algos: str = "auto",
    algos_list: str = "",
) -> None:
    sweep = sweep or PIPELINE_SWEEP
    modes = resolve_algorithms("pipeline", algos, algos_list)
    results: List[Dict[str, Any]] = []

    for B in sweep:
        logger.info("[%s] Pipeline sweep: B=%d, alpha=%.3f", name, B, alpha)
        per_mode_vals = {m: {"thr": [], "avg": [], "p95": [], "p99": [], "sd": []} for m in modes}

        for rep in range(repeats):
            gc_collect()
            reset_np_rng(SEED + rep)
            kz = generate_zipf_workload(keys, size=len(keys), alpha=alpha)

            for mode in modes:
                if mode == "D-HASH":
                    W = B  # align window to pipeline to avoid batch fragmentation
                    params = {"T": max(30, B), "W": W}
                    thr, avg, p95, p99, sd = run_single_mode(
                        kz,
                        mode_name=mode,
                        pipeline_size=B,
                        dhash_params=params,
                    )
                else:
                    thr, avg, p95, p99, sd = run_single_mode(
                        kz,
                        mode_name=mode,
                        pipeline_size=B,
                    )
                per_mode_vals[mode]["thr"].append(thr)
                per_mode_vals[mode]["avg"].append(avg)
                per_mode_vals[mode]["p95"].append(p95)
                per_mode_vals[mode]["p99"].append(p99)
                per_mode_vals[mode]["sd"].append(sd)

        for mode in modes:
            m_thr, s_thr = _mean_std(per_mode_vals[mode]["thr"])
            m_avg, s_avg = _mean_std(per_mode_vals[mode]["avg"])
            m_p95, s_p95 = _mean_std(per_mode_vals[mode]["p95"])
            m_p99, s_p99 = _mean_std(per_mode_vals[mode]["p99"])
            m_sd, s_sd = _mean_std(per_mode_vals[mode]["sd"])

            results.append(
                {
                    "Dataset": name,
                    "Stage": "Pipeline",
                    "Zipf α": alpha,
                    "Pipeline B": B,
                    "Mode": mode,
                    "DHash W": (B if mode == "D-HASH" else ""),
                    "Throughput (ops/sec) (avg)": m_thr,
                    "Throughput (ops/sec) (std)": s_thr,
                    "Avg (ms) (avg)": m_avg,
                    "Avg (ms) (std)": s_avg,
                    "P95 (ms) (avg)": m_p95,
                    "P95 (ms) (std)": s_p95,
                    "P99 (ms) (avg)": m_p99,
                    "P99 (ms) (std)": s_p99,
                    "Load Stddev (avg)": m_sd,
                    "Load Stddev (std)": s_sd,
                    "Repeats": repeats,
                }
            )

    pd.DataFrame(results).to_csv(out_csv, index=False)


# -----------------------------------------------------------------------------
# Stage A2: Microbench (get_node only)
# -----------------------------------------------------------------------------
def _microbench_once_get_node(
    algo: str,
    num_ops: int,
    num_keys: int,
    dhash_params: Optional[Dict[str, int]] = None,
    hot: bool = False,
    rng_seed: int = SEED,
) -> float:
    if algo == "CH":
        sh = ConsistentHashing(NODES, replicas=REPLICAS)
    elif algo == "D-HASH":
        params = dhash_params or {"T": 50, "W": 1024}
        sh = DHash(
            NODES,
            hot_key_threshold=int(params["T"]),
            window_size=int(params["W"]),
        )
    else:
        raise ValueError("algo must be 'CH' or 'D-HASH' for microbench")

    keys = [f"mbkey-{i}" for i in range(num_keys)]
    if algo == "D-HASH" and hot:
        T = getattr(sh, "hot_key_threshold", int((dhash_params or {}).get("T", 50)))
        for k in keys:
            for _ in range(T + 1):
                _ = sh.get_node(k)

    rng = random.Random(rng_seed)
    kidx = 0
    start = time.perf_counter_ns()
    for _ in range(num_ops):
        k = keys[kidx]
        _ = sh.get_node(k)
        kidx += 1
        if kidx == num_keys:
            kidx = 0
            rng.shuffle(keys)
    ns_per_op = (time.perf_counter_ns() - start) / num_ops
    return float(ns_per_op)


def run_microbench(
    name: str,
    out_csv: str,
    num_ops: int = MICROBENCH_OPS,
    num_keys: int = MICROBENCH_NUM_KEYS,
    dhash_params: Optional[Dict[str, int]] = None,
    repeats: int = NUM_REPEATS,
) -> None:
    results: List[Tuple[str, str, float]] = []

    for rep in range(repeats):
        # 동일 반복 내 공정 비교: 같은 키 순서를 모든 알고리즘에 적용 (seed=SEED+rep)
        ns = _microbench_once_get_node("CH", num_ops, num_keys, rng_seed=SEED + rep)
        results.append(("CH", "cold", ns))

        ns = _microbench_once_get_node(
            "D-HASH",
            num_ops,
            num_keys,
            dhash_params,
            hot=False,
            rng_seed=SEED + rep,
        )
        results.append(("D-HASH", "cold", ns))

        ns = _microbench_once_get_node(
            "D-HASH",
            num_ops,
            num_keys,
            dhash_params,
            hot=True,
            rng_seed=SEED + rep,
        )
        results.append(("D-HASH", "hot", ns))

        logger.info("[Progress] microbench rep=%d/%d", rep + 1, repeats)

    rows: List[Dict[str, Any]] = []
    for algo in ["CH", "D-HASH"]:
        phases = ["cold"] if algo == "CH" else ["cold", "hot"]
        for phase in phases:
            vals = [r[2] for r in results if r[0] == algo and r[1] == phase]
            m, s = _mean_std(vals)
            rows.append(
                {
                    "Dataset": name,
                    "Stage": "Microbench",
                    "Algorithm": algo,
                    "Phase": phase,
                    "ns/op (avg)": float(m),
                    "ns/op (std)": float(s),
                    "Promotions (sum)": 0,
                    "Lock Acquires (sum)": 0,
                    "Repeats": repeats,
                    "Ops per Repeat": num_ops,
                    "Keys": num_keys,
                    "DHash.T": dhash_params.get("T") if dhash_params else "",
                    "DHash.R": 2 if algo == "D-HASH" else "",
                    "DHash.W": dhash_params.get("W") if dhash_params else "",
                }
            )

    pd.DataFrame(rows).to_csv(out_csv, index=False)


# -----------------------------------------------------------------------------
# Stage B: Ablation (T with fixed R=2, W)
# -----------------------------------------------------------------------------
def run_ablation(
    name: str,
    keys: List[Any],
    out_csv: str,
    alpha: float,
    thresholds: List[int],
    fixed_window: int,
    repeats: int = NUM_REPEATS,
) -> None:
    results: List[Dict[str, Any]] = []
    logger.info("[%s] Ablation (Zipf α=%.3f, W=%d, R=2)", name, alpha, fixed_window)

    for T in thresholds:
        vals = {"thr": [], "avg": [], "p95": [], "p99": [], "sd": []}

        for rep in range(repeats):
            gc_collect()
            reset_np_rng(SEED + rep)
            kz = generate_zipf_workload(keys, size=len(keys), alpha=alpha)

            dh = DHash(NODES, hot_key_threshold=T, window_size=fixed_window)
            flush_databases(NODES, flush_async=False)
            warmup_cluster(dh, kz)
            metrics = benchmark_cluster(kz, dh, pipeline_size=fixed_window)

            vals["thr"].append(metrics["throughput_ops_s"])
            vals["avg"].append(metrics["avg_ms"])
            vals["p95"].append(metrics["p95_ms"])
            vals["p99"].append(metrics["p99_ms"])
            vals["sd"].append(load_stddev(metrics["node_load"]))

        m_thr, s_thr = _mean_std(vals["thr"])
        m_avg, s_avg = _mean_std(vals["avg"])
        m_p95, s_p95 = _mean_std(vals["p95"])
        m_p99, s_p99 = _mean_std(vals["p99"])
        m_sd, s_sd = _mean_std(vals["sd"])

        results.append(
            {
                "Dataset": name,
                "Stage": "Ablation",
                "Mode": "D-HASH",
                "Replicas (R)": 2,
                "Threshold (T)": T,
                "Window (W)": fixed_window,
                "Zipf α": alpha,
                "Throughput (ops/sec) (avg)": m_thr,
                "Throughput (ops/sec) (std)": s_thr,
                "Avg (ms) (avg)": m_avg,
                "Avg (ms) (std)": s_avg,
                "P95 (ms) (avg)": m_p95,
                "P95 (ms) (std)": s_p95,
                "P99 (ms) (avg)": m_p99,
                "P99 (ms) (std)": s_p99,
                "Load Stddev (avg)": m_sd,
                "Load Stddev (std)": s_sd,
                "Repeats": repeats,
            }
        )

    pd.DataFrame(results).to_csv(out_csv, index=False)


# -----------------------------------------------------------------------------
# Stage C: Zipf main results
# -----------------------------------------------------------------------------
def run_zipf(
    name: str,
    keys: List[Any],
    out_csv: str,
    alphas: List[float],
    dhash_params: Optional[Dict[str, int]] = None,
    pipeline_size: int = PIPELINE_SIZE_DEFAULT,
    repeats: int = NUM_REPEATS,
    algos: str = "auto",
    algos_list: str = "",
) -> None:
    results: List[Dict[str, Any]] = []
    modes = resolve_algorithms("zipf", algos, algos_list)

    for a in alphas:
        logger.info("[%s] Zipf alpha=%.3f", name, a)
        per_mode = {m: {"thr": [], "avg": [], "p95": [], "p99": [], "sd": []} for m in modes}

        for rep in range(repeats):
            gc_collect()
            reset_np_rng(SEED + rep)
            kz = generate_zipf_workload(keys, size=len(keys), alpha=a)

            for mode in modes:
                if mode == "D-HASH":
                    params = dict(dhash_params or {})
                    W = int(params.get("W", pipeline_size))
                    params.setdefault("T", max(30, W))  # safety default
                    thr, avg, p95, p99, sd = run_single_mode(
                        kz,
                        mode_name=mode,
                        pipeline_size=pipeline_size,
                        dhash_params=params,
                    )
                else:
                    thr, avg, p95, p99, sd = run_single_mode(
                        kz,
                        mode_name=mode,
                        pipeline_size=pipeline_size,
                    )
                per_mode[mode]["thr"].append(thr)
                per_mode[mode]["avg"].append(avg)
                per_mode[mode]["p95"].append(p95)
                per_mode[mode]["p99"].append(p99)
                per_mode[mode]["sd"].append(sd)

        for mode in modes:
            m_thr, s_thr = _mean_std(per_mode[mode]["thr"])
            m_avg, s_avg = _mean_std(per_mode[mode]["avg"])
            m_p95, s_p95 = _mean_std(per_mode[mode]["p95"])
            m_p99, s_p99 = _mean_std(per_mode[mode]["p99"])
            m_sd, s_sd = _mean_std(per_mode[mode]["sd"])

            results.append(
                {
                    "Dataset": name,
                    "Stage": "Zipf",
                    "Mode": mode,
                    "Zipf α": a,
                    "Throughput (ops/sec) (avg)": m_thr,
                    "Throughput (ops/sec) (std)": s_thr,
                    "Avg (ms) (avg)": m_avg,
                    "Avg (ms) (std)": s_avg,
                    "P95 (ms) (avg)": m_p95,
                    "P95 (ms) (std)": s_p95,
                    "P99 (ms) (avg)": m_p99,
                    "P99 (ms) (std)": s_p99,
                    "Load Stddev (avg)": m_sd,
                    "Load Stddev (std)": s_sd,
                    "Repeats": repeats,
                    "DHash.T": (dhash_params or {}).get("T") if (mode == "D-HASH") else "",
                    "DHash.R": 2 if mode == "D-HASH" else "",
                    "DHash.W": (dhash_params or {}).get("W", pipeline_size)
                    if (mode == "D-HASH")
                    else "",
                    "Pipeline B": pipeline_size if mode == "D-HASH" else "",
                }
            )

    pd.DataFrame(results).to_csv(out_csv, index=False)


# -----------------------------------------------------------------------------
# Redistribution (optional)
# -----------------------------------------------------------------------------
def compute_redistribution_rate(
    nodes_before: List[str],
    nodes_after: List[str],
    keys: List[Any],
    ctor,
) -> float:
    sh_b = ctor(nodes_before)
    sh_a = ctor(nodes_after)
    moved = sum(1 for k in keys if sh_b.get_node(k) != sh_a.get_node(k))
    return moved / max(1, len(keys))


def run_redistribution_report(
    name: str,
    keys: List[Any],
    out_csv: str,
    sample_k: int = 100_000,
    sizes: Tuple[int, int] = (5, 6),
) -> None:
    K = keys[: min(sample_k, len(keys))]
    n1, n2 = sizes
    nodes_a = [f"redis-{i}" for i in range(1, n1 + 1)]
    nodes_b = [f"redis-{i}" for i in range(1, n2 + 1)]

    def _ch_ctor(ns: List[str]):
        return ConsistentHashing(ns, replicas=REPLICAS)

    def _wch_ctor(ns: List[str]):
        weights = {n: 1.0 + 0.1 * i for i, n in enumerate(ns)}
        return WeightedConsistentHashing(ns, weights=weights, base_replicas=REPLICAS)

    def _rv_ctor(ns: List[str]):
        return RendezvousHashing(ns)

    rows = []
    for algo, ctor in (("CH", _ch_ctor), ("WCH", _wch_ctor), ("Rendezvous", _rv_ctor)):
        rows.append(
            {
                "Algorithm": algo,
                "Event": f"{n1}->{n2}",
                "Move (%)": compute_redistribution_rate(nodes_a, nodes_b, K, ctor) * 100,
            }
        )
        rows.append(
            {
                "Algorithm": algo,
                "Event": f"{n2}->{n1}",
                "Move (%)": compute_redistribution_rate(nodes_b, nodes_a, K, ctor) * 100,
            }
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
def run_experiments(
    mode: str,
    alpha_for_ablation: float,
    dataset_filter: str = "ALL",
    fixed_window: Optional[int] = None,
    dhash_T: Optional[int] = None,
    pipeline_for_zipf: Optional[int] = None,
    repeats: int = NUM_REPEATS,
    algos: str = "auto",
    algos_list: str = "",
) -> None:
    os.makedirs("results", exist_ok=True)

    datasets = [
        ("NASA", os.path.join("data", "nasa_http_logs.log"), "logs"),
        ("eBay", os.path.join("data", "ebay_auction_logs.csv"), "csv"),
    ]
    if dataset_filter != "ALL":
        datasets = [d for d in datasets if d[0] == dataset_filter]

    for name, path, kind in datasets:
        if not os.path.exists(path):
            logger.warning("[%s] Dataset not found: %s → skip", name, path)
            continue

        if kind == "logs":
            keys, _ = load_logs_dataset(path)
        else:
            keys, _ = load_csv_dataset(path, natural_hot_threshold=None)

        logger.info("[%s] Loaded %d keys (kind=%s)", name, len(keys), kind)

        # A1: pipeline sweep
        if mode in ("pipeline", "all"):
            out_csv = os.path.join("results", f"{name.lower()}_pipeline_sweep.csv")
            run_pipeline_sweep(
                name=name,
                keys=keys,
                out_csv=out_csv,
                alpha=1.5,
                sweep=PIPELINE_SWEEP,
                repeats=repeats,
                algos=algos,
                algos_list=algos_list,
            )
            env_path = os.path.join("results", f"{name.lower()}_pipeline_env_meta.csv")
            pd.DataFrame([runtime_env_metadata(repeats=repeats)]).to_csv(env_path, index=False)

        # A2: microbench
        if mode in ("microbench", "all"):
            W = fixed_window or PIPELINE_SIZE_DEFAULT
            out_csv = os.path.join("results", f"{name.lower()}_microbench_ns.csv")
            run_microbench(
                name=name,
                out_csv=out_csv,
                num_ops=MICROBENCH_OPS,
                num_keys=MICROBENCH_NUM_KEYS,
                dhash_params={"T": max(30, W), "W": W},
                repeats=repeats,
            )
            env_path = os.path.join("results", f"{name.lower()}_microbench_env_meta.csv")
            pd.DataFrame([runtime_env_metadata(repeats=repeats)]).to_csv(env_path, index=False)

        # B: ablation(T), fixed R=2, W
        if mode in ("ablation", "all"):
            W = fixed_window or pipeline_for_zipf or PIPELINE_SIZE_DEFAULT
            out_csv = os.path.join("results", f"{name.lower()}_ablation_results.csv")
            run_ablation(
                name=name,
                keys=keys,
                out_csv=out_csv,
                alpha=alpha_for_ablation,
                thresholds=ABLAT_THRESHOLDS,
                fixed_window=W,
                repeats=repeats,
            )
            env_path = os.path.join("results", f"{name.lower()}_ablation_env_meta.csv")
            pd.DataFrame([runtime_env_metadata(repeats=repeats)]).to_csv(env_path, index=False)

        # C: zipf main
        if mode in ("zipf", "all"):
            W = fixed_window or PIPELINE_SIZE_DEFAULT
            B = pipeline_for_zipf or W
            T = dhash_T if dhash_T is not None else max(30, W)
            out_csv = os.path.join("results", f"{name.lower()}_zipf_results.csv")
            run_zipf(
                name=name,
                keys=keys,
                out_csv=out_csv,
                alphas=ZIPF_ALPHAS,
                dhash_params={"T": T, "W": W},
                pipeline_size=B,
                repeats=repeats,
                algos=algos,
                algos_list=algos_list,
            )
            env_path = os.path.join("results", f"{name.lower()}_zipf_env_meta.csv")
            pd.DataFrame([runtime_env_metadata(repeats=repeats)]).to_csv(env_path, index=False)

        # Optional: redistribution
        if mode == "redistrib":
            out_csv = os.path.join("results", f"{name.lower()}_redistribution.csv")
            run_redistribution_report(
                name,
                keys,
                out_csv=out_csv,
                sample_k=100_000,
                sizes=(5, 6),
            )
            env_path = os.path.join("results", f"{name.lower()}_redistribution_env_meta.csv")
            pd.DataFrame([runtime_env_metadata(repeats=repeats)]).to_csv(env_path, index=False)
