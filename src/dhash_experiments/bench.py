from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from statistics import stdev
from typing import Any, Dict, List, Tuple

from redis import ConnectionPool, StrictRedis

from .config import NODES, PIPELINE_SIZE_DEFAULT, SEED, TTL_SECONDS, VALUE_BYTES

logger = logging.getLogger(__name__)

_connection_pools: Dict[str, ConnectionPool] = {}


# -----------------------------------------------------------------------------
# Redis connections
# -----------------------------------------------------------------------------
def _redis_client(host: str) -> StrictRedis:
    if host not in _connection_pools:
        _connection_pools[host] = ConnectionPool(host=host, port=6379, db=0)
    return StrictRedis(connection_pool=_connection_pools[host])


def redis_client_for_node(node: str) -> StrictRedis:
    return _redis_client(node)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def _value_payload(value_bytes: int) -> bytes:
    base = b'{"v":0}'
    if value_bytes <= len(base):
        return base[: max(value_bytes, 0)]
    return base + b"x" * (value_bytes - len(base))


def _weighted_percentile(samples: List[Tuple[float, int]], q: float) -> float:
    """
    Weighted percentile over (value, weight) samples with linear interpolation.
    """
    if not samples:
        return 0.0
    samples_sorted = sorted(samples, key=lambda x: x[0])
    total_w = sum(w for _, w in samples_sorted)
    if total_w <= 0:
        return 0.0
    target = q * total_w
    cum = 0.0
    prev_v = samples_sorted[0][0]
    for v, w in samples_sorted:
        next_cum = cum + w
        if next_cum >= target:
            if w == 0:
                return v
            frac = (target - cum) / w
            return prev_v + (v - prev_v) * frac
        prev_v = v
        cum = next_cum
    return samples_sorted[-1][0]


def load_stddev(node_load: Dict[str, int]) -> float:
    vals = [node_load.get(n, 0) for n in NODES]
    return stdev(vals) if len(vals) > 1 else 0.0


# -----------------------------------------------------------------------------
# Warmup & flush
# -----------------------------------------------------------------------------
def warmup_cluster(sharding: Any, keys: List[Any], ratio: float = 0.01, cap: int = 1000) -> None:
    import random

    n = max(1, min(int(len(keys) * ratio), cap))
    rng = random.Random(SEED)
    sample = rng.sample(keys, n) if len(keys) >= n else list(keys)

    write_buckets: Dict[str, List[Any]] = defaultdict(list)
    read_buckets: Dict[str, List[Any]] = defaultdict(list)

    for k in sample:
        write_buckets[sharding.get_node(k, op="write")].append(k)
        read_buckets[sharding.get_node(k, op="read")].append(k)

    payload = b'{"warm":1}'

    for node, node_keys in write_buckets.items():
        cli = redis_client_for_node(node)
        pipe = cli.pipeline()
        for k in node_keys:
            pipe.set(str(k), payload, ex=60)
        pipe.execute()

    for node, node_keys in read_buckets.items():
        cli = redis_client_for_node(node)
        pipe = cli.pipeline()
        for k in node_keys:
            pipe.get(str(k))
        pipe.execute()

    logger.info(
        "[Warmup] sample=%d keys across %d nodes",
        n,
        len(set(write_buckets) | set(read_buckets)),
    )


def flush_databases(redis_nodes: List[str], flush_async: bool = False) -> None:
    """
    Flush Redis DBs on all nodes. If async, poll DBSIZE to zero.
    """

    def _init_one(container: str) -> None:
        try:
            cli = _redis_client(container)
            if flush_async:
                try:
                    cli.flushdb(asynchronous=True)
                except TypeError:
                    cli.execute_command("FLUSHDB", "ASYNC")
                # poll until empty
                for _ in range(10_000):  # ~50s max
                    try:
                        if int(cli.dbsize()) == 0:
                            break
                    except Exception:
                        pass
                    time.sleep(0.005)
            else:
                try:
                    cli.flushdb()
                except TypeError:
                    cli.execute_command("FLUSHDB")
        except Exception as e:
            logger.warning("Redis(%s) flush failed: %s", container, e)

    with ThreadPoolExecutor(max_workers=len(redis_nodes)) as ex:
        list(ex.map(_init_one, redis_nodes))


# -----------------------------------------------------------------------------
# Core benchmark
# -----------------------------------------------------------------------------
def benchmark_cluster(
    keys: List[Any],
    sharding: Any,
    ex_seconds: int = TTL_SECONDS,
    pipeline_size: int = PIPELINE_SIZE_DEFAULT,
    value_bytes: int = VALUE_BYTES,
) -> Dict[str, Any]:
    """
    KSII-style cluster bench:
      - Throughput = total ops / (max write wall + max read wall)
      - Latency = weighted stats of per-batch per-op averages (write/read/combined)
      - Load Stddev = stdev of node (write+read) counts
    """
    # Bucketing by actual routing
    write_buckets: Dict[str, List[Any]] = defaultdict(list)
    read_buckets: Dict[str, List[Any]] = defaultdict(list)
    for k in keys:
        write_buckets[sharding.get_node(k, op="write")].append(k)
        read_buckets[sharding.get_node(k, op="read")].append(k)

    node_load: Dict[str, int] = {
        n: len(write_buckets.get(n, [])) + len(read_buckets.get(n, []))
        for n in NODES
    }
    if sum(node_load.values()) == 0:
        return {
            "throughput_ops_s": 0.0,
            "avg_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "write_avg_ms": 0.0,
            "write_p95_ms": 0.0,
            "write_p99_ms": 0.0,
            "read_avg_ms": 0.0,
            "read_p95_ms": 0.0,
            "read_p99_ms": 0.0,
            "node_load": node_load,
        }

    payload = _value_payload(value_bytes)

    def _io_write(item: Tuple[str, List[Any]]) -> Tuple[float, List[Tuple[float, int]]]:
        node, node_keys = item
        cli = redis_client_for_node(node)
        total = 0.0
        samples: List[Tuple[float, int]] = []
        for i in range(0, len(node_keys), pipeline_size):
            chunk = node_keys[i : i + pipeline_size]
            pipe = cli.pipeline()
            for k in chunk:
                pipe.set(str(k), payload, ex=ex_seconds)
            t0 = time.perf_counter_ns()
            pipe.execute()
            dt = (time.perf_counter_ns() - t0) / 1e9
            total += dt
            ops = max(len(chunk), 1)
            samples.append((dt / ops, ops))
        return total, samples

    def _io_read(item: Tuple[str, List[Any]]) -> Tuple[float, List[Tuple[float, int]]]:
        node, node_keys = item
        cli = redis_client_for_node(node)
        total = 0.0
        samples: List[Tuple[float, int]] = []
        for i in range(0, len(node_keys), pipeline_size):
            chunk = node_keys[i : i + pipeline_size]
            pipe = cli.pipeline()
            for k in chunk:
                pipe.get(str(k))
            t0 = time.perf_counter_ns()
            _ = pipe.execute()
            dt = (time.perf_counter_ns() - t0) / 1e9
            total += dt
            ops = max(len(chunk), 1)
            samples.append((dt / ops, ops))
        return total, samples

    logger.info(
        "[Bench] nodes(write=%d, read=%d), pipeline=%d, payload=%d bytes",
        len(write_buckets),
        len(read_buckets),
        pipeline_size,
        len(payload),
    )

    write_node_totals: List[float] = []
    read_node_totals: List[float] = []
    write_all_samples: List[Tuple[float, int]] = []
    read_all_samples: List[Tuple[float, int]] = []

    with ThreadPoolExecutor(max_workers=max(1, len(write_buckets))) as ex:
        for total, samples in ex.map(_io_write, write_buckets.items()):
            write_node_totals.append(total)
            write_all_samples.extend(samples)

    with ThreadPoolExecutor(max_workers=max(1, len(read_buckets))) as ex:
        for total, samples in ex.map(_io_read, read_buckets.items()):
            read_node_totals.append(total)
            read_all_samples.extend(samples)

    write_ops = sum(len(v) for v in write_buckets.values())
    read_ops = sum(len(v) for v in read_buckets.values())
    total_ops = write_ops + read_ops

    write_wall = max(write_node_totals) if write_node_totals else 0.0
    read_wall = max(read_node_totals) if read_node_totals else 0.0
    cluster_wall = write_wall + read_wall
    throughput = (total_ops / cluster_wall) if cluster_wall > 0 else 0.0

    def _wavg(samples: List[Tuple[float, int]]) -> float:
        wsum = sum(w for _, w in samples)
        if wsum == 0:
            return 0.0
        return sum(v * w for v, w in samples) / wsum

    write_avg_ms = _wavg(write_all_samples) * 1000.0
    read_avg_ms = _wavg(read_all_samples) * 1000.0
    combined = write_all_samples + read_all_samples
    avg_ms = _wavg(combined) * 1000.0

    write_p95_ms = _weighted_percentile(write_all_samples, 0.95) * 1000.0
    write_p99_ms = _weighted_percentile(write_all_samples, 0.99) * 1000.0
    read_p95_ms = _weighted_percentile(read_all_samples, 0.95) * 1000.0
    read_p99_ms = _weighted_percentile(read_all_samples, 0.99) * 1000.0
    p95_ms = _weighted_percentile(combined, 0.95) * 1000.0
    p99_ms = _weighted_percentile(combined, 0.99) * 1000.0

    return {
        "throughput_ops_s": float(throughput),
        "avg_ms": float(avg_ms),
        "p95_ms": float(p95_ms),
        "p99_ms": float(p99_ms),
        "write_avg_ms": float(write_avg_ms),
        "write_p95_ms": float(write_p95_ms),
        "write_p99_ms": float(write_p99_ms),
        "read_avg_ms": float(read_avg_ms),
        "read_p95_ms": float(read_p95_ms),
        "read_p99_ms": float(read_p99_ms),
        "node_load": {n: int(node_load.get(n, 0)) for n in NODES},
    }
