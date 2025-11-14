from __future__ import annotations

from bisect import bisect
from typing import Any, Dict, List, Optional

from .config import REPLICAS

# -----------------------------------------------------------------------------
# Hash (xxHash64)
# -----------------------------------------------------------------------------
try:
    import xxhash as _xx
except Exception as e:
    raise RuntimeError("xxhash package is required. Install with: pip install xxhash") from e


def fast_hash64(key: Any) -> int:
    return _xx.xxh64(str(key).encode("utf-8")).intdigest()


# -----------------------------------------------------------------------------
# Consistent Hashing (CH)
# -----------------------------------------------------------------------------
class ConsistentHashing:
    def __init__(self, nodes: List[str], replicas: int = REPLICAS) -> None:
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        for node in nodes:
            self.add_node(node)

    @staticmethod
    def _hash(key: Any) -> int:
        return fast_hash64(key)

    def add_node(self, node: str) -> None:
        for i in range(self.replicas):
            k = self._hash(f"{node}:{i}")
            self.ring[k] = node
            self.sorted_keys.append(k)
        self.sorted_keys.sort()

    # op is ignored (compat with D-HASH)
    def get_node(self, key: Any, op: str = "read") -> str:
        hk = self._hash(key)
        idx = bisect(self.sorted_keys, hk) % len(self.sorted_keys)
        return self.ring[self.sorted_keys[idx]]


# -----------------------------------------------------------------------------
# Weighted Consistent Hashing (WCH)
# -----------------------------------------------------------------------------
class WeightedConsistentHashing:
    """Largest remainder allocation keeping total virtual points equal to CH."""

    def __init__(
        self,
        nodes: List[str],
        weights: Optional[Dict[str, float]] = None,
        base_replicas: int = REPLICAS,
    ) -> None:
        self.base_replicas = base_replicas
        self.weights = weights or {n: 1.0 for n in nodes}
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self._build_ring()

    @staticmethod
    def _hash(key: Any) -> int:
        return fast_hash64(key)

    def _build_ring(self) -> None:
        total_points = len(self.weights) * self.base_replicas
        wsum = sum(self.weights.values()) or 1.0
        quotas = {n: self.weights[n] / wsum * total_points for n in self.weights}
        floors = {n: int(quotas[n]) for n in quotas}
        remain = total_points - sum(floors.values())
        order = sorted(
            self.weights.keys(),
            key=lambda n: (quotas[n] - floors[n], n),
            reverse=True,
        )
        alloc = floors.copy()
        for n in order[:remain]:
            alloc[n] += 1
        for node, reps in alloc.items():
            for i in range(reps):
                k = self._hash(f"{node}:{i}")
                self.ring[k] = node
                self.sorted_keys.append(k)
        self.sorted_keys.sort()

    def get_node(self, key: Any, op: str = "read") -> str:
        hk = self._hash(key)
        idx = bisect(self.sorted_keys, hk) % len(self.sorted_keys)
        return self.ring[self.sorted_keys[idx]]


# -----------------------------------------------------------------------------
# Rendezvous / HRW
# -----------------------------------------------------------------------------
class RendezvousHashing:
    """Highest Random Weight (Rendezvous/HRW) hashing."""

    def __init__(self, nodes: List[str]) -> None:
        self.nodes = list(nodes)

    @staticmethod
    def _score(key: Any, node: str) -> int:
        return fast_hash64(f"{key}|{node}")

    def get_node(self, key: Any, op: str = "read") -> str:
        best_node: Optional[str] = None
        best_score = -1
        for n in self.nodes:
            s = self._score(key, n)
            if s > best_score:
                best_score = s
                best_node = n
        assert best_node is not None
        return best_node


# -----------------------------------------------------------------------------
# D-HASH (R=2, Sticky-Window Alternation)
# -----------------------------------------------------------------------------
class DHash:
    """
    D-HASH (R=2, Sticky-Window Alternation)
      - write(op='write'): always CH primary
      - read(op='read')  : after promotion (cnt>=T), alternate primary<->alt every W requests
      - Guard            : first W requests after promotion always use primary (tail spike guard)
      - Alternate        : chosen deterministically via per-key stride (not consecutive successor)
    Window is request-count based (not time). For dynamic membership, invalidate alt externally.
    """

    __slots__ = ("nodes", "T", "W", "reads", "alt", "ch", "hot_key_threshold")

    def __init__(
        self,
        nodes: List[str],
        hot_key_threshold: int = 50,
        window_size: int = 500,
        replicas: int = REPLICAS,
        ring: Optional[ConsistentHashing] = None,
    ) -> None:
        if not nodes:
            raise ValueError("DHash requires at least one node.")
        self.nodes: List[str] = list(nodes)
        self.T: int = int(hot_key_threshold)
        self.W: int = max(1, int(window_size))
        self.reads: Dict[Any, int] = {}
        self.alt: Dict[Any, str] = {}
        self.ch = ring if ring is not None else ConsistentHashing(nodes, replicas=replicas)
        self.hot_key_threshold: int = self.T  # external compatibility

    # --- helpers ---
    @staticmethod
    def _h(key: Any) -> int:
        return fast_hash64(key)

    def _primary_safe(self, key: Any) -> str:
        rk = getattr(self.ch, "sorted_keys", None)
        ring = getattr(self.ch, "ring", None)
        if rk and ring:
            hk = self._h(key)
            idx = bisect(rk, hk) % len(rk)
            return ring[rk[idx]]
        # deterministic fallback
        return self.nodes[self._h(f"{key}|p") % len(self.nodes)]

    def _ensure_alternate(self, key: Any) -> None:
        if key in self.alt:
            return

        rk = getattr(self.ch, "sorted_keys", None)
        ring = getattr(self.ch, "ring", None)

        if not rk or not ring or len(self.nodes) <= 1:
            self.alt[key] = self._primary_safe(key)
            return

        hk = self._h(key)
        i = bisect(rk, hk) % len(rk)
        primary = ring[rk[i]]

        # stride in [1, num_nodes-1]
        stride_span = max(1, len(self.nodes) - 1)
        stride = 1 + (self._h(f"{key}|alt") % stride_span)

        # 1) stride search
        j = i
        for _ in range(len(rk)):
            j = (j + stride) % len(rk)
            cand = ring[rk[j]]
            if cand != primary:
                self.alt[key] = cand
                return

        # 2) linear scan backup
        j = i
        for _ in range(len(rk)):
            j = (j + 1) % len(rk)
            cand = ring[rk[j]]
            if cand != primary:
                self.alt[key] = cand
                return

        # 3) fallback
        self.alt[key] = primary

    # --- public API ---
    def get_node(self, key: Any, op: str = "read") -> str:
        if op == "write":
            return self._primary_safe(key)

        cnt = self.reads.get(key, 0) + 1
        self.reads[key] = cnt

        if cnt < self.T and key not in self.alt:
            return self._primary_safe(key)

        self._ensure_alternate(key)

        delta = max(0, cnt - self.T)
        if delta < self.W:
            return self._primary_safe(key)

        epoch = (delta - self.W) // self.W  # 0-based after guard
        return self.alt[key] if (epoch % 2 == 0) else self._primary_safe(key)
