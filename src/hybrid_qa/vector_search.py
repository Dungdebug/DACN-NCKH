"""
Vector Search Engine with HNSW (Hierarchical Navigable Small World) & Cosine Similarity.

Provides efficient nearest-neighbor search over dense passage embedding vectors.

Core Algorithms:
- Cosine Similarity & Distance: D_cos(u, v) = 1.0 - (u . v) / (||u|| * ||v||)
- HNSW Multi-Layer Proximity Graph:
  - Multi-layer graph traversal with logarithmic search complexity O(log N).
  - Greedy layer stepping from top level down to layer 0.
  - Bidirectional neighbor connection with heuristic pruning.
- Dual-mode Execution:
  - Exact Cosine Search for small sets (< 100 passages) with zero approximation loss.
  - HNSW Graph Traversal for large-scale passage collections.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Compute Cosine Similarity between vectors u and v."""
    u_f = u.flatten()
    v_f = v.flatten()
    dot = float(np.dot(u_f, v_f))
    norm_u = float(np.linalg.norm(u_f))
    norm_v = float(np.linalg.norm(v_f))
    if norm_u < 1e-9 or norm_v < 1e-9:
        return 0.0
    return dot / (norm_u * norm_v)


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Compute Cosine Distance: 1.0 - CosineSimilarity."""
    sim = cosine_similarity(u, v)
    return max(0.0, 1.0 - sim)


@dataclass
class HNSWNode:
    """A node inside the HNSW proximity graph."""
    node_id: str
    vector: np.ndarray
    max_level: int
    # neighbors[l] is list of neighbor node_ids at layer l
    neighbors: List[List[str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HNSWIndex:
    """
    Pure Python/NumPy Implementation of Hierarchical Navigable Small World (HNSW).
    Guarantees logarithmic search latency without requiring native compiled C++ extensions.
    """

    def __init__(
        self,
        dimension: int = 384,
        m: int = 16,
        m0: int = 32,
        ef_construction: int = 64,
        ef_search: int = 32,
        seed: int = 42,
    ):
        self.dimension = dimension
        self.m = m
        self.m0 = m0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m_l = 1.0 / math.log(m)

        self.nodes: Dict[str, HNSWNode] = {}
        self.entry_point_id: Optional[str] = None
        self.max_level: int = -1

        self._rng = random.Random(seed)

    def _random_level(self) -> int:
        """Assign random level with exponentially decaying probability."""
        r = self._rng.random()
        if r == 0:
            r = 1e-9
        return int(-math.log(r) * self.m_l)

    def _dist(self, u_vec: np.ndarray, v_node_id: str) -> float:
        v_vec = self.nodes[v_node_id].vector
        return cosine_distance(u_vec, v_vec)

    def _search_layer(
        self,
        query_vec: np.ndarray,
        entry_points: List[str],
        ef: int,
        layer: int,
    ) -> List[Tuple[float, str]]:
        """
        Greedy search on a specific layer of the HNSW graph.
        Returns list of (distance, node_id) sorted by distance.
        """
        visited: Set[str] = set(entry_points)
        # candidates: min-heap of (distance, node_id)
        candidates: List[Tuple[float, str]] = []
        # best_w: max-heap (negated dist) keeping top ef elements
        best_w: List[Tuple[float, str]] = []

        for ep in entry_points:
            d = self._dist(query_vec, ep)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(best_w, (-d, ep))

        while candidates:
            d_c, c_id = heapq.heappop(candidates)
            furthest_best_dist = -best_w[0][0]

            if d_c > furthest_best_dist:
                break

            node = self.nodes.get(c_id)
            if not node or layer >= len(node.neighbors):
                continue

            for neighbor_id in node.neighbors[layer]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    d_neighbor = self._dist(query_vec, neighbor_id)
                    furthest_best_dist = -best_w[0][0]

                    if d_neighbor < furthest_best_dist or len(best_w) < ef:
                        heapq.heappush(candidates, (d_neighbor, neighbor_id))
                        heapq.heappush(best_w, (-d_neighbor, neighbor_id))
                        if len(best_w) > ef:
                            heapq.heappop(best_w)

        results = [(-neg_d, node_id) for neg_d, node_id in best_w]
        results.sort(key=lambda x: x[0])
        return results

    def insert(
        self,
        node_id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a vector into the HNSW graph."""
        vec = vector.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm

        target_level = self._random_level()
        node = HNSWNode(
            node_id=node_id,
            vector=vec,
            max_level=target_level,
            neighbors=[[] for _ in range(target_level + 1)],
            metadata=metadata or {},
        )
        self.nodes[node_id] = node

        if self.entry_point_id is None:
            self.entry_point_id = node_id
            self.max_level = target_level
            return

        curr_ep = self.entry_point_id
        top_l = self.max_level

        # 1. From top layer down to target_level + 1: greedy 1-NN traversal
        for l in range(top_l, target_level, -1):
            res = self._search_layer(vec, [curr_ep], ef=1, layer=l)
            if res:
                curr_ep = res[0][1]

        # 2. From min(top_l, target_level) down to 0: find candidates and establish connections
        entry_points = [curr_ep]
        for l in range(min(top_l, target_level), -1, -1):
            candidates = self._search_layer(vec, entry_points, ef=self.ef_construction, layer=l)
            m_curr = self.m0 if l == 0 else self.m
            # Select top-M nearest neighbors
            chosen_neighbors = [nid for _, nid in candidates[:m_curr]]
            node.neighbors[l] = chosen_neighbors

            # Add bidirectional links
            for n_id in chosen_neighbors:
                neighbor_node = self.nodes[n_id]
                if l < len(neighbor_node.neighbors):
                    neighbor_node.neighbors[l].append(node_id)
                    # Prune excess neighbors if over capacity
                    if len(neighbor_node.neighbors[l]) > (m_curr * 2):
                        n_list = neighbor_node.neighbors[l]
                        n_dists = [(self._dist(neighbor_node.vector, nid), nid) for nid in n_list]
                        n_dists.sort(key=lambda x: x[0])
                        neighbor_node.neighbors[l] = [nid for _, nid in n_dists[:m_curr]]

            entry_points = [nid for _, nid in candidates]

        if target_level > self.max_level:
            self.max_level = target_level
            self.entry_point_id = node_id

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        ef_search: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for top-k nearest neighbors using Cosine Similarity.
        Returns: list of (node_id, cosine_similarity_score).
        """
        if not self.nodes:
            return []

        q_vec = query_vec.flatten().astype(np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 1e-9:
            q_vec = q_vec / norm

        # If graph is small, exact search is faster and 100% accurate
        if len(self.nodes) <= 64 or self.entry_point_id is None:
            exact_scores = []
            for nid, node in self.nodes.items():
                sim = float(np.dot(q_vec, node.vector))
                exact_scores.append((nid, sim))
            exact_scores.sort(key=lambda x: x[1], reverse=True)
            return exact_scores[:top_k]

        ef = ef_search or max(self.ef_search, top_k)
        curr_ep = self.entry_point_id

        # 1. Top level down to 1
        for l in range(self.max_level, 0, -1):
            res = self._search_layer(q_vec, [curr_ep], ef=1, layer=l)
            if res:
                curr_ep = res[0][1]

        # 2. Layer 0 search
        layer0_results = self._search_layer(q_vec, [curr_ep], ef=ef, layer=0)
        # Convert distances back to cosine similarities: sim = 1.0 - dist
        top_results = []
        for dist, nid in layer0_results[:top_k]:
            sim = max(-1.0, min(1.0, 1.0 - dist))
            top_results.append((nid, sim))

        return top_results

    def clear(self) -> None:
        """Clear the entire index."""
        self.nodes.clear()
        self.entry_point_id = None
        self.max_level = -1

    def __len__(self) -> int:
        return len(self.nodes)
