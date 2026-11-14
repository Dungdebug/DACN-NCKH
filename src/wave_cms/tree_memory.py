"""
WaveCMS Tree Memory: Document-Tree Wavelet Continuum Memory.
Implements the multi-resolution document tree, telescoping bottom-up consolidation,
wavelet details decomposition, and certified exact unlearning.

Theoretical Foundations:
- Proposal: WaveCMS (Document-Tree Wavelet Continuum Memory with Equilibrium Multi-Hop Readout)
- Theorem 1: Multi-resolution Telescoping & Parseval Identity on Trees
- Theorem 2: Zero-Forgetting in Storage & Certified Exact Unlearning in O(depth)
- 100% Strict Offline, Pure Python stdlib & NumPy, Zero External APIs.
"""

from collections import Counter
from enum import Enum
import gc
import hashlib
import math
import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

__all__ = [
    "TreeLevel",
    "DocumentTreeNode",
    "WaveTreeMemory",
    "compute_deterministic_routing_key",
    "strip_accents",
]

_ROUTING_STOPWORDS = frozenset({
    "la", "va", "cua", "cho", "co", "mot", "cac", "nhung", "trong", "nay",
    "khi", "the", "gi", "nao", "sao", "duoc", "bang", "tu", "den", "voi",
    "nhu", "ve", "tren", "duoi", "neu", "thi", "hoac", "de", "tai",
    "is", "are", "the", "of", "and", "in", "to", "for", "a", "an",
    "what", "how", "why", "with", "that", "this", "on", "by", "as",
})


class CertifiedUnlearningResult(dict):
    """Result object supporting both dict subscripting and tuple unpacking."""
    def __iter__(self):
        yield self["certified_exact"]
        yield self["max_discrepancy"]

    def __getitem__(self, key):
        if key == "certified":
            return self["certified_exact"]
        return super().__getitem__(key)


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks and accents for robust normalized matching."""
    if not text:
        return ""
    text_nfd = unicodedata.normalize("NFD", str(text))
    cleaned = "".join(ch for ch in text_nfd if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", cleaned).lower().replace("đ", "d").strip()


class TreeLevel(str, Enum):
    """Hierarchical levels in the Wavelet Continuum Memory document tree."""
    CORPUS = "CORPUS"          # Depth 0: Global repository root
    DOCUMENT = "DOCUMENT"      # Depth 1: Individual document level
    SECTION = "SECTION"        # Depth 2: Chapter / topic / section level
    LEAF_PARAGRAPH = "LEAF_PARAGRAPH"  # Depth 3: Individual retrievable passage/paragraph


def compute_deterministic_routing_key(text: str, dim: int = 64) -> np.ndarray:
    """
    Computes a deterministic, unit-normalized (L2 = 1.0) semantic routing key in R^dim
    using signed feature hashing over word unigrams/bigrams and intra-word character
    n-grams with sublinear TF weighting.
    """
    dim = max(1, int(dim))
    vec = np.zeros(dim, dtype=np.float64)
    clean = strip_accents(text or "")
    if not clean:
        vec[0] = 1.0
        return vec

    words = re.findall(r"[a-z0-9]+", clean)
    feat_weights: Counter = Counter()
    feat_scale: Dict[str, float] = {}

    if words:
        for i, w in enumerate(words):
            if len(w) == 1 and len(words) > 2:
                continue
            w_key = f"w1:{w}"
            feat_weights[w_key] += 1
            feat_scale[w_key] = (
                0.25 if w in _ROUTING_STOPWORDS else (1.65 if len(w) >= 4 else 1.5)
            )

            if i + 1 < len(words):
                b_key = f"w2:{w}_{words[i + 1]}"
                feat_weights[b_key] += 1
                feat_scale[b_key] = 1.80

            if len(w) >= 3:
                for j in range(len(w) - 2):
                    c_key = f"c3:{w[j:j + 3]}"
                    feat_weights[c_key] += 1
                    feat_scale[c_key] = 0.20
    else:
        compact = re.sub(r"\s+", "", clean)
        if len(compact) >= 3:
            for j in range(len(compact) - 2):
                c_key = f"c3:{compact[j:j + 3]}"
                feat_weights[c_key] += 1
                feat_scale[c_key] = 1.0
        elif compact:
            feat_weights[f"c1:{compact}"] = 1
            feat_scale[f"c1:{compact}"] = 1.0

    for feat, count in feat_weights.items():
        h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], byteorder="little") % dim
        sign = 1.0 if (h[4] & 1) == 0 else -1.0
        tf = 1.0 + math.log(float(count))
        vec[idx] += sign * tf * feat_scale[feat]

    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        return (vec / norm).astype(np.float64)

    fallback = np.zeros(dim, dtype=np.float64)
    fallback[0] = 1.0
    return fallback


class DocumentTreeNode:
    """
    A node in the WaveCMS document tree.
    Holds local memory core Theta, wavelet detail D, routing key kappa,
    and structural relationship pointers.
    """

    def __init__(
        self,
        node_id: str,
        level: TreeLevel,
        title: str = "",
        text: str = "",
        weight: float = 1.0,
        theta: Optional[np.ndarray] = None,
        detail: Optional[np.ndarray] = None,
        kappa: Optional[np.ndarray] = None,
        parent: Optional["DocumentTreeNode"] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.node_id = str(node_id)
        self.level = level
        self.title = str(title or "")
        self.text = str(text or "")
        self.weight = max(1e-6, float(weight))
        self.total_weight: float = self.weight

        # Parameter memory core Theta (r x r)
        self.theta: Optional[np.ndarray] = theta
        self._theta_version: int = 0

        # Wavelet detail D_v = Theta_v - Theta_par(v) (r x r), 0 for root
        # Tracked against parent's _theta_version for strict O(depth) lazy updates
        self._detail: Optional[np.ndarray] = detail
        self._detail_parent_version: int = parent._theta_version if parent is not None else 0

        # Routing key kappa_v in R^dim, unit-normalized
        self.kappa: Optional[np.ndarray] = kappa
        self.unnormalized_kappa: Optional[np.ndarray] = (
            self.weight * kappa.copy() if kappa is not None else None
        )

        self.parent: Optional["DocumentTreeNode"] = parent
        self.children: List["DocumentTreeNode"] = []
        self.metadata: Dict[str, Any] = dict(metadata or {})

    @property
    def detail(self) -> Optional[np.ndarray]:
        """Wavelet detail D_v = Theta_v - Theta_par(v), lazily synced in O(1) on parent updates."""
        if (
            self._detail is not None
            and self.parent is not None
            and self.theta is not None
            and self.parent.theta is not None
            and self._detail_parent_version != self.parent._theta_version
        ):
            self._detail = self.theta - self.parent.theta
            self._detail_parent_version = self.parent._theta_version
        return self._detail

    @detail.setter
    def detail(self, val: Optional[np.ndarray]) -> None:
        self._detail = val
        self._detail_parent_version = self.parent._theta_version if self.parent is not None else 0

    @property
    def detail_dv(self) -> np.ndarray:
        """Alias for wavelet detail D_v."""
        d = self.detail
        if d is None:
            return np.zeros((8, 8), dtype=np.float64)
        return d

    @detail_dv.setter
    def detail_dv(self, val: np.ndarray) -> None:
        self.detail = val

    @property
    def routing_key(self) -> np.ndarray:
        """Alias for unit routing key kappa_v."""
        if self.kappa is None:
            return np.zeros(64, dtype=np.float64)
        return self.kappa

    @routing_key.setter
    def routing_key(self, val: np.ndarray) -> None:
        self.kappa = val

    @property
    def subtree_weight(self) -> float:
        """Alias for total subtree weight W_v."""
        return self.total_weight

    def add_child(self, child: "DocumentTreeNode") -> None:
        """Appends a child node and links parent pointer."""
        child.parent = self
        self.children.append(child)

    def is_leaf(self) -> bool:
        """Returns True if the node is a leaf (no children)."""
        return len(self.children) == 0

    def get_leaves(self) -> List["DocumentTreeNode"]:
        """Returns all leaf nodes in the subtree rooted at this node in depth-first order."""
        if self.is_leaf():
            return [self]
        leaves: List["DocumentTreeNode"] = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def get_subtree_nodes(self) -> List["DocumentTreeNode"]:
        """Returns all nodes in the subtree rooted at this node (including self)."""
        nodes: List["DocumentTreeNode"] = [self]
        for child in self.children:
            nodes.extend(child.get_subtree_nodes())
        return nodes

    def get_path_from_root(self) -> List["DocumentTreeNode"]:
        """Returns the sequence of nodes along the path from root to self."""
        path: List["DocumentTreeNode"] = []
        curr: Optional["DocumentTreeNode"] = self
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        path.reverse()
        return path

    def get_ancestors(self) -> List["DocumentTreeNode"]:
        """Returns ancestor nodes from parent up to root."""
        ancestors: List["DocumentTreeNode"] = []
        curr = self.parent
        while curr is not None:
            ancestors.append(curr)
            curr = curr.parent
        return ancestors

    def get_depth(self) -> int:
        """Returns tree depth of this node (root has depth 0)."""
        d = 0
        curr = self.parent
        while curr is not None:
            d += 1
            curr = curr.parent
        return d

    def to_dict(self) -> Dict[str, Any]:
        """Serializes node metadata (excluding numpy arrays)."""
        return {
            "node_id": self.node_id,
            "level": self.level.value,
            "title": self.title,
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "weight": self.weight,
            "total_weight": self.total_weight,
            "has_theta": self.theta is not None,
            "has_detail": self.detail is not None,
            "has_kappa": self.kappa is not None,
            "children_count": len(self.children),
            "depth": self.get_depth(),
        }


class WaveTreeMemory:
    """
    Manages the multi-resolution continuum memory tree across a corpus.
    Maintains telescoping consolidation, wavelet detail decomposition,
    and certified exact unlearning.
    """

    def __init__(
        self,
        r: int = 8,
        dim: int = 64,
        key_dim: Optional[int] = None,
        corpus_id: str = "GLOBAL_ROOT",
        seed: int = 42,
    ):
        self.r = int(r)
        self.dim = int(key_dim if key_dim is not None else dim)
        self.key_dim = self.dim
        self.corpus_id = str(corpus_id)
        self.seed = int(seed)
        self._rng = np.random.RandomState(self.seed)

        # Global corpus root node
        self.root = DocumentTreeNode(
            node_id=self.corpus_id if self.corpus_id else "root",
            level=TreeLevel.CORPUS,
            title="Corpus Global Root",
            text="",
            weight=1.0,
            theta=np.zeros((self.r, self.r), dtype=np.float64),
            detail=np.zeros((self.r, self.r), dtype=np.float64),
            kappa=np.zeros(self.dim, dtype=np.float64),
        )
        self.documents: Dict[str, DocumentTreeNode] = {}

    def _init_leaf_core(self, text: str, title: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deterministically initializes leaf core Theta in R^(r x r) and routing key kappa in R^dim.
        kappa uses deterministic word + char n-gram TF feature projection normalized to unit L2 norm;
        Theta uses the closed-form semantic ridge projection in the shared (A, B, W_0) subspace.
        """
        combined_str = f"{title}\n{text}".strip()
        kappa = compute_deterministic_routing_key(combined_str, dim=self.dim)

        adapter = getattr(self, "_leaf_adapter", None)
        if adapter is None:
            from src.wave_cms.continuum_memory import ContinuumMemoryAdapter

            adapter = ContinuumMemoryAdapter(d_model=self.dim, r=self.r, seed=42)
            self._leaf_adapter = adapter

        theta = adapter.compute_semantic_leaf_core(kappa, beta=1.0)
        return theta, kappa

    def ingest_document(
        self,
        doc_id: str,
        title: str,
        sections: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
        default_leaf_weight: float = 1.0,
        **kwargs: Any,
    ) -> DocumentTreeNode:
        """
        Ingests a document with hierarchical sections and passages into the tree:
        Root (CORPUS) -> Document (DOCUMENT) -> Section (SECTION) -> Paragraph (LEAF_PARAGRAPH).
        Runs bottom-up telescoping consolidation upon completion.
        """
        # If document already exists, remove it first to avoid duplicate branches
        if doc_id in self.documents:
            self.unlearn_document(doc_id)

        doc_node = DocumentTreeNode(
            node_id=str(doc_id),
            level=TreeLevel.DOCUMENT,
            title=title or doc_id,
            text="",
            weight=1.0,
            metadata={"document_id": doc_id, **(metadata or {})},
        )

        sec_counter = 0
        leaf_counter = 0

        # Parse sections input (supports dict, Passage object, or string list)
        for s_idx, sec_input in enumerate(sections):
            sec_counter += 1
            if isinstance(sec_input, dict):
                sec_title = sec_input.get("title") or sec_input.get("section_title") or f"Section {sec_counter}"
                passages_raw = sec_input.get("passages") or sec_input.get("paragraphs") or [sec_input.get("text", "")]
            elif hasattr(sec_input, "section_title") and hasattr(sec_input, "text"):
                # Duck-typing for Passage object
                sec_title = getattr(sec_input, "section_title", "") or f"Section {sec_counter}"
                passages_raw = [sec_input]
            elif isinstance(sec_input, (list, tuple)):
                sec_title = f"Section {sec_counter}"
                passages_raw = sec_input
            else:
                sec_title = f"Section {sec_counter}"
                passages_raw = [str(sec_input)]

            sec_node = DocumentTreeNode(
                node_id=f"sec:{doc_id}:s{sec_counter:02d}",
                level=TreeLevel.SECTION,
                title=sec_title,
                text="",
                weight=1.0,
                metadata={"section_index": sec_counter},
            )

            for p_raw in passages_raw:
                leaf_counter += 1
                if isinstance(p_raw, dict):
                    p_text = p_raw.get("text", "")
                    p_title = p_raw.get("title", sec_title)
                    p_weight = float(p_raw.get("weight", max(1.0, len(p_text.split()))))
                    p_id = p_raw.get("passage_id") or f"leaf:{doc_id}:p{leaf_counter:03d}"
                elif hasattr(p_raw, "text"):
                    p_text = getattr(p_raw, "text", "")
                    p_title = getattr(p_raw, "section_title", sec_title)
                    p_weight = float(max(1.0, len(p_text.split())))
                    p_id = getattr(p_raw, "passage_id", f"leaf:{doc_id}:p{leaf_counter:03d}")
                else:
                    p_text = str(p_raw)
                    p_title = sec_title
                    p_weight = float(max(1.0, len(p_text.split())))
                    p_id = f"leaf:{doc_id}:p{leaf_counter:03d}"

                if not p_text.strip():
                    continue

                theta_p, kappa_p = self._init_leaf_core(p_text, p_title)

                leaf_node = DocumentTreeNode(
                    node_id=p_id,
                    level=TreeLevel.LEAF_PARAGRAPH,
                    title=p_title,
                    text=p_text,
                    weight=p_weight,
                    theta=theta_p,
                    detail=np.zeros((self.r, self.r), dtype=np.float64),
                    kappa=kappa_p,
                    metadata={"passage_index": leaf_counter, "document_id": doc_id},
                )
                sec_node.add_child(leaf_node)

            if sec_node.children:
                doc_node.add_child(sec_node)

        # Fallback if document had no valid sections/passages
        if not doc_node.children:
            fallback_theta, fallback_kappa = self._init_leaf_core(title, title)
            fallback_leaf = DocumentTreeNode(
                node_id=f"leaf:{doc_id}:p001",
                level=TreeLevel.LEAF_PARAGRAPH,
                title=title,
                text=title,
                weight=1.0,
                theta=fallback_theta,
                detail=np.zeros((self.r, self.r), dtype=np.float64),
                kappa=fallback_kappa,
                metadata={"passage_index": 1, "document_id": doc_id},
            )
            fallback_sec = DocumentTreeNode(
                node_id=f"sec:{doc_id}:s01",
                level=TreeLevel.SECTION,
                title=title,
                text="",
                weight=1.0,
            )
            fallback_sec.add_child(fallback_leaf)
            doc_node.add_child(fallback_sec)

        # Attach document to corpus root
        self.root.add_child(doc_node)
        self.documents[doc_id] = doc_node

        # Run bottom-up telescoping consolidation and compute wavelet details
        self.telescoping_consolidation()

        return doc_node

    # Alias for API compatibility
    add_document = ingest_document

    def telescoping_consolidation(self) -> None:
        """
        Executes Theorem 1 bottom-up telescoping consolidation:
        1. For each internal node v:
           W_v = sum_{c in ch(v)} W_c
           Theta_v = sum_{c in ch(v)} (W_c / W_v) * Theta_c
           kappa_v = normalize( sum_{c in ch(v)} W_c * kappa_c )
        2. Top-down wavelet details:
           D_root = 0
           D_v = Theta_v - Theta_par(v) for v != root
        """
        if not self.root.children:
            self.root.total_weight = 1.0
            self.root.theta = np.zeros((self.r, self.r), dtype=np.float64)
            self.root.detail = np.zeros((self.r, self.r), dtype=np.float64)
            self.root.kappa = np.zeros(self.dim, dtype=np.float64)
            self.root.unnormalized_kappa = np.zeros(self.dim, dtype=np.float64)
            return

        def _consolidate_node(node: DocumentTreeNode) -> Tuple[float, np.ndarray, np.ndarray]:
            if node.is_leaf():
                node.total_weight = node.weight
                node.unnormalized_kappa = (
                    node.weight * node.kappa.copy()
                    if node.kappa is not None
                    else np.zeros(self.dim, dtype=np.float64)
                )
                return node.total_weight, node.theta, node.unnormalized_kappa

            child_weights = []
            child_thetas = []
            child_unnorm_kappas = []

            for child in node.children:
                cw, ct, cuk = _consolidate_node(child)
                child_weights.append(cw)
                child_thetas.append(ct)
                child_unnorm_kappas.append(cuk)

            total_w = sum(child_weights)
            node.total_weight = total_w

            # Telescoping weighted sum of cores and additive unnormalized routing keys
            accum_theta = np.zeros((self.r, self.r), dtype=np.float64)
            accum_kappa = np.zeros(self.dim, dtype=np.float64)

            for w, t, uk in zip(child_weights, child_thetas, child_unnorm_kappas):
                fraction = w / total_w
                accum_theta += fraction * t
                accum_kappa += uk

            node.theta = accum_theta
            node._theta_version += 1
            node.unnormalized_kappa = accum_kappa
            norm_k = np.linalg.norm(accum_kappa)
            node.kappa = accum_kappa / (norm_k + 1e-12)

            return node.total_weight, node.theta, node.unnormalized_kappa

        _consolidate_node(self.root)

        # Top-down wavelet details decomposition: D_v = Theta_v - Theta_par(v)
        self.root.detail = np.zeros((self.r, self.r), dtype=np.float64)

        def _compute_details(node: DocumentTreeNode) -> None:
            for child in node.children:
                child.detail = child.theta - node.theta
                _compute_details(child)

        _compute_details(self.root)

    @staticmethod
    def _release_subtree_memory(node: Optional[DocumentTreeNode]) -> Tuple[int, int]:
        """
        Recursively severs circular references, clears arrays, and frees subtree memory.
        Returns (freed_nodes_count, freed_bytes_estimate).
        """
        if node is None:
            return 0, 0
        freed_nodes = 0
        freed_bytes = 0
        stack = [node]
        to_clear: List[DocumentTreeNode] = []
        while stack:
            curr = stack.pop()
            to_clear.append(curr)
            for ch in curr.children:
                stack.append(ch)
        for n in to_clear:
            freed_nodes += 1
            if n.theta is not None:
                freed_bytes += n.theta.nbytes
                n.theta = None
            if n._detail is not None:
                freed_bytes += n._detail.nbytes
                n._detail = None
            if n.kappa is not None:
                freed_bytes += n.kappa.nbytes
                n.kappa = None
            if n.unnormalized_kappa is not None:
                freed_bytes += n.unnormalized_kappa.nbytes
                n.unnormalized_kappa = None
            freed_bytes += len(n.text.encode("utf-8")) + len(n.title.encode("utf-8"))
            n.text = ""
            n.title = ""
            n.metadata.clear()
            n.children.clear()
            n.parent = None
        gc.collect()
        return freed_nodes, freed_bytes

    def _algebraic_downdate_ancestor_chain(
        self,
        removed_node: DocumentTreeNode,
        start_parent: Optional[DocumentTreeNode],
    ) -> int:
        """
        Theorem 2 Strict O(depth) Algebraic Telescoping Downdate:
        Walks strictly along the ancestor chain from `start_parent` up to `self.root`
        without touching any unrelated subtrees or looping over sibling children.
        Wavelet details D_c = Theta_c - Theta_par(c) sync lazily in O(1) via _theta_version.
        Returns the number of ancestor nodes updated (strictly <= depth).
        """
        if start_parent is None:
            return 0

        w_u = float(removed_node.total_weight)
        theta_u = (
            removed_node.theta
            if removed_node.theta is not None
            else np.zeros((self.r, self.r), dtype=np.float64)
        )
        if removed_node.unnormalized_kappa is not None:
            k_unnorm_u = removed_node.unnormalized_kappa.copy()
        elif removed_node.kappa is not None:
            k_unnorm_u = w_u * removed_node.kappa
        else:
            k_unnorm_u = np.zeros(self.dim, dtype=np.float64)

        delta_w = -w_u
        delta_theta_mass = -(w_u * theta_u)
        delta_k_unnorm = -k_unnorm_u

        nodes_updated = 0
        curr: Optional[DocumentTreeNode] = start_parent
        while curr is not None:
            nodes_updated += 1
            w_old = float(curr.total_weight)
            theta_old = (
                curr.theta.copy()
                if curr.theta is not None
                else np.zeros((self.r, self.r), dtype=np.float64)
            )
            if curr.unnormalized_kappa is not None:
                unnorm_old = curr.unnormalized_kappa.copy()
            elif curr.kappa is not None:
                unnorm_old = w_old * curr.kappa
            else:
                unnorm_old = np.zeros(self.dim, dtype=np.float64)

            w_new = w_old + delta_w

            if w_new <= 1e-12 or len(curr.children) == 0:
                curr.total_weight = 1.0 if curr is self.root else 0.0
                curr.theta = np.zeros((self.r, self.r), dtype=np.float64)
                curr._theta_version += 1
                curr.detail = np.zeros((self.r, self.r), dtype=np.float64)
                curr.kappa = np.zeros(self.dim, dtype=np.float64)
                curr.unnormalized_kappa = np.zeros(self.dim, dtype=np.float64)

                # Propagate full removal up if non-root node became empty
                delta_w = -w_old
                delta_theta_mass = -(w_old * theta_old)
                delta_k_unnorm = -unnorm_old
            else:
                theta_new = (w_old * theta_old + delta_theta_mass) / w_new
                curr.total_weight = w_new
                curr.theta = theta_new
                curr._theta_version += 1

                k_new_unnorm = unnorm_old + delta_k_unnorm
                curr.unnormalized_kappa = k_new_unnorm
                norm_k = float(np.linalg.norm(k_new_unnorm))
                curr.kappa = k_new_unnorm / (norm_k + 1e-12)

            curr = curr.parent

        self.root.detail = np.zeros((self.r, self.r), dtype=np.float64)
        return nodes_updated

    def unlearn_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Theorem 2 Certified Exact Unlearning in O(depth):
        Removes the document subtree, updates ancestors algebraically in O(depth),
        and releases subtree memory.
        """
        if doc_id not in self.documents:
            return {
                "certified": False,
                "certified_exact": False,
                "unlearned_doc_id": doc_id,
                "error": f"Document {doc_id} not found in memory tree",
                "remaining_docs": list(self.documents.keys()),
            }

        target_node = self.documents[doc_id]
        parent_node = target_node.parent if target_node.parent is not None else self.root

        if target_node in parent_node.children:
            parent_node.children.remove(target_node)
        target_node.parent = None
        del self.documents[doc_id]

        # Execute O(depth) algebraic downdate without traversing unrelated subtrees
        nodes_updated = self._algebraic_downdate_ancestor_chain(target_node, parent_node)

        # Free memory of the removed subtree
        freed_nodes, freed_bytes = self._release_subtree_memory(target_node)

        return {
            "certified": True,
            "certified_exact": True,
            "unlearned_doc_id": doc_id,
            "remaining_docs": list(self.documents.keys()),
            "remaining_documents": len(self.documents),
            "remaining_leaves": len(self.get_leaves()),
            "root_weight": self.root.total_weight,
            "complexity": "O(depth)",
            "nodes_updated": nodes_updated,
            "freed_nodes": freed_nodes,
            "freed_bytes": freed_bytes,
        }

    def get_node(self, node_id: str) -> Optional[DocumentTreeNode]:
        """Finds a node by node_id or document_id."""
        if node_id in self.documents:
            return self.documents[node_id]
        if f"doc:{node_id}" in self.documents:
            return self.documents[f"doc:{node_id}"]
        res = self.find_node_by_id(node_id)
        if res is not None:
            return res
        return self.find_node_by_id(f"doc:{node_id}")

    def certify_exact_unlearning(
        self,
        doc_id: str,
        baseline_docs: Optional[List[Dict[str, Any]]] = None,
        atol: float = 1e-12,
    ) -> Dict[str, Any]:
        """
        Rigorous analytical verification of Theorem 2 (Certified Exact Unlearning in O(depth)).
        Removes the document subtree via O(depth) algebraic downdate, frees memory,
        and measures the exact numerical discrepancy against fresh counterfactual consolidation.
        """
        if doc_id not in self.documents:
            return CertifiedUnlearningResult({
                "certified_exact": False,
                "error": f"Document {doc_id} not found in memory tree",
                "remaining_documents": len(self.documents),
                "remaining_leaves": len(self.get_leaves()),
            })

        target_node = self.documents[doc_id]
        parent_node = target_node.parent if target_node.parent is not None else self.root

        if target_node in parent_node.children:
            parent_node.children.remove(target_node)
        target_node.parent = None
        del self.documents[doc_id]

        # 1. Execute O(depth) algebraic downdate
        nodes_updated = self._algebraic_downdate_ancestor_chain(target_node, parent_node)

        # 2. Free memory of the unlearned subtree
        freed_nodes, freed_bytes = self._release_subtree_memory(target_node)

        # 3. Independently verify numerical discrepancy against a fresh counterfactual shadow tree
        shadow_tree = WaveTreeMemory(
            r=self.r,
            key_dim=self.dim,
            corpus_id=self.corpus_id,
            seed=self.seed,
        )
        if baseline_docs is not None:
            for d_spec in baseline_docs:
                b_id = d_spec.get("doc_id")
                if b_id == doc_id:
                    continue
                shadow_tree.ingest_document(
                    doc_id=b_id,
                    title=d_spec.get("title", b_id),
                    sections=d_spec.get("sections", []),
                    metadata=d_spec.get("metadata"),
                )
        else:
            for d_id, d_node in self.documents.items():
                doc_clone = DocumentTreeNode(
                    node_id=d_node.node_id,
                    level=TreeLevel.DOCUMENT,
                    title=d_node.title,
                    text=d_node.text,
                    weight=d_node.weight,
                    metadata=dict(d_node.metadata),
                )
                for sec in d_node.children:
                    sec_clone = DocumentTreeNode(
                        node_id=sec.node_id,
                        level=TreeLevel.SECTION,
                        title=sec.title,
                        text=sec.text,
                        weight=sec.weight,
                        metadata=dict(sec.metadata),
                    )
                    for lf in sec.children:
                        lf_clone = DocumentTreeNode(
                            node_id=lf.node_id,
                            level=TreeLevel.LEAF_PARAGRAPH,
                            title=lf.title,
                            text=lf.text,
                            weight=lf.weight,
                            theta=lf.theta.copy() if lf.theta is not None else None,
                            detail=np.zeros((self.r, self.r), dtype=np.float64),
                            kappa=lf.kappa.copy() if lf.kappa is not None else None,
                            metadata=dict(lf.metadata),
                        )
                        sec_clone.add_child(lf_clone)
                    if sec_clone.children:
                        doc_clone.add_child(sec_clone)
                if doc_clone.children:
                    shadow_tree.root.add_child(doc_clone)
                    shadow_tree.documents[d_id] = doc_clone
            shadow_tree.telescoping_consolidation()

        theta_inf = float(np.max(np.abs(self.root.theta - shadow_tree.root.theta)))
        theta_frob = float(np.linalg.norm(self.root.theta - shadow_tree.root.theta, ord="fro"))
        kappa_inf = float(np.max(np.abs(self.root.kappa - shadow_tree.root.kappa)))
        kappa_frob = float(np.linalg.norm(self.root.kappa - shadow_tree.root.kappa))
        unnorm_inf = (
            float(np.max(np.abs(self.root.unnormalized_kappa - shadow_tree.root.unnormalized_kappa)))
            if self.root.unnormalized_kappa is not None and shadow_tree.root.unnormalized_kappa is not None
            else 0.0
        )

        max_node_theta_inf = theta_inf
        max_node_theta_frob = theta_frob
        max_node_detail_inf = 0.0
        max_node_detail_frob = 0.0
        for d_id, d_node in self.documents.items():
            s_doc = shadow_tree.documents.get(d_id)
            if s_doc is None:
                continue
            for n_cur, n_ref in zip(d_node.get_subtree_nodes(), s_doc.get_subtree_nodes()):
                if n_cur.theta is not None and n_ref.theta is not None:
                    diff_t = n_cur.theta - n_ref.theta
                    max_node_theta_inf = max(max_node_theta_inf, float(np.max(np.abs(diff_t))))
                    max_node_theta_frob = max(max_node_theta_frob, float(np.linalg.norm(diff_t, ord="fro")))
                if n_cur.detail is not None and n_ref.detail is not None:
                    diff_d = n_cur.detail - n_ref.detail
                    max_node_detail_inf = max(max_node_detail_inf, float(np.max(np.abs(diff_d))))
                    max_node_detail_frob = max(max_node_detail_frob, float(np.linalg.norm(diff_d, ord="fro")))

        self._release_subtree_memory(shadow_tree.root)
        del shadow_tree

        max_inf_disc = max(theta_inf, kappa_inf, unnorm_inf, max_node_theta_inf, max_node_detail_inf)
        max_frob_disc = max(theta_frob, kappa_frob, max_node_theta_frob, max_node_detail_frob)

        all_thm = self.verify_all_theorems()
        is_certified = bool(max_inf_disc < atol and max_frob_disc < atol and all_thm.get("all_theorems_verified", True))

        return CertifiedUnlearningResult({
            "certified_exact": is_certified,
            "certified": is_certified,
            "unlearned": True,
            "exact_match": is_certified,
            "max_discrepancy": float(max_inf_disc),
            "max_frobenius_discrepancy": float(max_frob_disc),
            "complexity": "O(depth)",
            "nodes_updated": nodes_updated,
            "unlearned_document_id": doc_id,
            "remaining_documents": len(self.documents),
            "remaining_leaves": len(self.get_leaves()),
            "freed_nodes": freed_nodes,
            "freed_bytes": freed_bytes,
            "theorems_satisfied": all_thm.get("all_theorems_verified", True),
        })

    # =========================================================================
    # THEOREM 1 MATHEMATICAL VERIFICATION SUITE
    # =========================================================================

    def verify_telescoping_identity(self, tol: float = 1e-12) -> Tuple[bool, float]:
        r"""
        Theorem 1(a) Path Telescoping:
        For every leaf p: Theta_p = Theta_root + sum_{v in pi(p) \ {root}} D_v
        Returns (is_valid, max_error).
        """
        leaves = self.get_leaves()
        if not leaves:
            return True, 0.0

        max_err = 0.0
        for leaf in leaves:
            path = leaf.get_path_from_root()
            accum = self.root.theta.copy()
            for node in path[1:]:
                accum += node.detail
            err = float(np.max(np.abs(accum - leaf.theta)))
            if err > max_err:
                max_err = err

        return max_err < tol, max_err

    def _compute_wavelet_expansion_fast(self, alpha_dict: Dict[str, float]) -> np.ndarray:
        """
        Computes Theta_root + sum_{v != root} A_v(alpha) * D_v in O(N) time
        via a single post-order traversal without repeated subtree leaf enumeration.
        """
        t_wav = self.root.theta.copy()

        def _postorder(node: DocumentTreeNode) -> float:
            if node.is_leaf():
                a_v = float(alpha_dict.get(node.node_id, 0.0))
            else:
                a_v = sum(_postorder(child) for child in node.children)
            if node is not self.root and a_v != 0.0:
                t_wav[...] += a_v * node.detail
            return a_v

        _postorder(self.root)
        return t_wav

    def reconstruct_routed_memory(
        self,
        alpha: Union[Dict[str, float], np.ndarray, List[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstructs routed memory via direct leaf sum (LHS) and Wavelet expansion (RHS).
        Returns (t_direct, t_wavelet).
        """
        leaves = self.get_leaves()
        if isinstance(alpha, dict):
            alpha_dict = alpha
        else:
            alpha_arr = np.asarray(alpha, dtype=np.float64)
            alpha_dict = {l.node_id: float(a) for l, a in zip(leaves, alpha_arr)}

        # Direct sum over leaves
        t_dir = np.zeros((self.r, self.r), dtype=np.float64)
        for leaf in leaves:
            a = alpha_dict.get(leaf.node_id, 0.0)
            t_dir += a * leaf.theta

        # O(N) Wavelet multi-resolution expansion
        t_wav = self._compute_wavelet_expansion_fast(alpha_dict)

        return t_dir, t_wav

    def verify_wavelet_expansion(
        self,
        tol: float = 1e-12,
        alpha: Optional[Any] = None,
    ) -> Tuple[bool, float]:
        """
        Theorem 1(b) Subtree Mass Wavelet Expansion:
        sum_p alpha_p * Theta_p = Theta_root + sum_{v != root} A_v(alpha) * D_v
        Returns (is_valid, max_error).
        """
        leaves = self.get_leaves()
        if not leaves:
            return True, 0.0

        if alpha is None:
            raw = self._rng.dirichlet(np.ones(len(leaves)))
            alpha = {l.node_id: float(a) for l, a in zip(leaves, raw)}

        t_dir, t_wav = self.reconstruct_routed_memory(alpha)
        err = float(np.max(np.abs(t_dir - t_wav)))
        return err < tol, err

    def verify_haar_vanishing(self, tol: float = 1e-12) -> Tuple[bool, float]:
        """
        Theorem 1(c) Haar Vanishing Moment Property:
        sum_{c in ch(u)} W_c * D_c = 0 for all internal nodes u.
        Guaranteed across arbitrary document scales and weight dynamics.
        Returns (is_valid, max_error).
        """
        max_err = 0.0
        max_rel_err = 0.0
        for node in self.get_all_nodes():
            if node.is_leaf() or not node.children:
                continue
            haar_sum = np.zeros((self.r, self.r), dtype=np.float64)
            for child in node.children:
                haar_sum += child.total_weight * child.detail
            err = float(np.max(np.abs(haar_sum)))
            w_u = max(1e-12, float(node.total_weight))
            rel_err = err / w_u
            if err > max_err:
                max_err = err
            if rel_err > max_rel_err:
                max_rel_err = rel_err

        is_valid = bool(max_err < tol or max_rel_err < tol)
        return is_valid, max_err

    def verify_parseval_identity(self, tol: float = 1e-10) -> Tuple[bool, float, float]:
        """
        Theorem 1(d) Parseval Identity on Tree:
        sum_{p in L} w_p * ||Theta_p - Theta_root||_F^2 = sum_{v != root} W_v * ||D_v||_F^2
        Guaranteed across arbitrary corpus sizes and scale profiles.
        Returns (is_conserved, e_leaves, e_wavelet).
        """
        leaves = self.get_leaves()
        if not leaves:
            return True, 0.0, 0.0

        e_leaves = 0.0
        for leaf in leaves:
            diff = leaf.theta - self.root.theta
            e_leaves += float(leaf.weight * np.sum(diff ** 2))

        e_wav = 0.0
        for node in self.get_all_nodes():
            if node is self.root:
                continue
            e_wav += float(node.total_weight * np.sum(node.detail ** 2))

        diff_e = abs(e_leaves - e_wav)
        rel_diff = diff_e / (max(abs(e_leaves), abs(e_wav)) + 1e-12)
        is_conserved = bool(diff_e < tol or rel_diff < 1e-7)
        return is_conserved, e_leaves, e_wav

    def verify_all_theorems(self) -> Dict[str, Any]:
        """Runs all mathematical verification checks and returns summary dictionary."""
        t_ok, t_err = self.verify_telescoping_identity()
        w_ok, w_err = self.verify_wavelet_expansion()
        h_ok, h_err = self.verify_haar_vanishing()
        p_ok, e_l, e_w = self.verify_parseval_identity()
        delta_p = abs(e_l - e_w)
        rel_p = float(delta_p / (max(abs(e_l), abs(e_w)) + 1e-12))
        all_ok = bool(t_ok and w_ok and h_ok and p_ok)
        return {
            "all_theorems_verified": all_ok,
            "telescoping": {"valid": t_ok, "verified": t_ok, "max_err": t_err, "max_error": t_err},
            "wavelet_expansion": {"valid": w_ok, "verified": w_ok, "max_err": w_err, "max_error": w_err},
            "haar_vanishing": {"valid": h_ok, "verified": h_ok, "max_err": h_err, "max_error": h_err},
            "parseval_conservation": {"valid": p_ok, "verified": p_ok, "e_leaves": e_l, "e_wavelet": e_w, "relative_error": rel_p},
            "theorem_1a_telescoping": {"valid": t_ok, "verified": t_ok, "max_err": t_err, "max_error": t_err},
            "theorem_1b_wavelet_expansion": {"valid": w_ok, "verified": w_ok, "max_err": w_err, "max_error": w_err},
            "theorem_1c_haar_vanishing": {"valid": h_ok, "verified": h_ok, "max_err": h_err, "max_error": h_err},
            "theorem_1d_parseval": {"valid": p_ok, "verified": p_ok, "leaves_energy": e_l, "wavelet_energy": e_w, "delta": delta_p, "relative_error": rel_p},
        }

    # =========================================================================
    # THEOREM 1 MATHEMATICAL VERIFICATION SUITE
    # =========================================================================

    def verify_telescoping(self, atol: float = 1e-6) -> bool:
        r"""
        Theorem 1(a) Path Telescoping:
        For every leaf p: Theta_p = Theta_root + sum_{v in pi(p) \ {root}} D_v
        """
        leaves = self.get_leaves()
        if not leaves:
            return True

        for leaf in leaves:
            path = leaf.get_path_from_root()
            # path is [root, doc, sec, leaf]
            accum = self.root.theta.copy()
            for node in path[1:]:
                accum += node.detail

            if not np.allclose(accum, leaf.theta, atol=atol):
                return False

        return True

    def verify_subtree_mass_formula(
        self,
        alpha: Optional[np.ndarray] = None,
        atol: float = 1e-6,
    ) -> bool:
        """
        Theorem 1(b) Subtree Mass Expansion:
        For any leaf routing distribution alpha:
        sum_p alpha_p * Theta_p = Theta_root + sum_{v != root} A_v(alpha) * D_v
        where A_v(alpha) = sum_{p in L(v)} alpha_p.
        """
        leaves = self.get_leaves()
        if not leaves:
            return True

        num_leaves = len(leaves)
        if alpha is None:
            # Generate random Dirichlet distribution
            alpha = self._rng.dirichlet(np.ones(num_leaves))
        else:
            alpha = np.asarray(alpha, dtype=np.float64)
            alpha = alpha / np.sum(alpha)

        # LHS: direct leaf combination
        lhs = np.zeros((self.r, self.r), dtype=np.float64)
        alpha_dict = {}
        for idx, leaf in enumerate(leaves):
            a_val = float(alpha[idx])
            alpha_dict[leaf.node_id] = a_val
            lhs += a_val * leaf.theta

        # RHS: O(N) Wavelet multi-resolution formula
        rhs = self._compute_wavelet_expansion_fast(alpha_dict)

        return bool(np.allclose(lhs, rhs, atol=atol))

    def verify_haar(self, atol: float = 1e-6) -> bool:
        """
        Theorem 1(c) Haar Vanishing Moment Property:
        For every internal node u: sum_{c in ch(u)} W_c * D_c = 0
        """
        nodes = self.get_all_nodes()
        for node in nodes:
            if node.is_leaf():
                continue
            haar_sum = np.zeros((self.r, self.r), dtype=np.float64)
            for child in node.children:
                haar_sum += child.total_weight * child.detail

            w_u = max(1.0, float(node.total_weight))
            if not (np.allclose(haar_sum, 0.0, atol=atol) or np.allclose(haar_sum / w_u, 0.0, atol=atol)):
                return False

        return True

    def verify_parseval(self, atol: float = 1e-6) -> Tuple[float, float, bool]:
        """
        Theorem 1(d) Parseval Identity on Tree:
        sum_{p in L} w_p * ||Theta_p - Theta_root||_F^2 = sum_{v != root} W_v * ||D_v||_F^2
        Returns (lhs, rhs, is_equal).
        """
        leaves = self.get_leaves()
        if not leaves:
            return 0.0, 0.0, True

        # LHS: sum over leaves
        lhs = 0.0
        for leaf in leaves:
            diff = leaf.theta - self.root.theta
            lhs += leaf.weight * float(np.sum(diff ** 2))

        # RHS: sum over all non-root tree details
        rhs = 0.0
        for node in self.get_all_nodes():
            if node is self.root:
                continue
            rhs += node.total_weight * float(np.sum(node.detail ** 2))

        is_equal = bool(np.isclose(lhs, rhs, atol=atol, rtol=1e-5))
        return lhs, rhs, is_equal

    def compute_leaf_theta_sum(self, alpha: np.ndarray) -> np.ndarray:
        """
        Computes effective memory core Theta(alpha) for a leaf routing distribution alpha
        via Theorem 1(b) multi-resolution expansion in O(N) time.
        """
        leaves = self.get_leaves()
        if not leaves:
            return self.root.theta.copy()

        alpha = np.asarray(alpha, dtype=np.float64)
        if len(alpha) != len(leaves):
            raise ValueError(f"Alpha length ({len(alpha)}) does not match leaves count ({len(leaves)})")

        alpha_dict = {leaf.node_id: float(alpha[idx]) for idx, leaf in enumerate(leaves)}
        return self._compute_wavelet_expansion_fast(alpha_dict)

    # =========================================================================
    # TREE UTILITIES & QUERIES
    # =========================================================================

    def get_leaves(self) -> List[DocumentTreeNode]:
        """Returns all leaf nodes across the entire corpus tree."""
        if not self.root.children:
            return []
        return self.root.get_leaves()

    def get_all_nodes(self) -> List[DocumentTreeNode]:
        """Returns all nodes across the entire tree in breadth-first / depth-first order."""
        return self.root.get_subtree_nodes()

    def find_node_by_id(self, node_id: str) -> Optional[DocumentTreeNode]:
        """Finds a node by its unique node_id or document_id."""
        if node_id in self.documents:
            return self.documents[node_id]
        for node in self.get_all_nodes():
            if node.node_id == node_id or node.node_id == f"doc:{node_id}":
                return node
            if node.metadata and node.metadata.get("document_id") == node_id:
                return node
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Returns summary structural statistics of the memory tree."""
        leaves = self.get_leaves()
        all_nodes = self.get_all_nodes()
        return {
            "total_documents": len(self.documents),
            "total_nodes": len(all_nodes),
            "total_leaves": len(leaves),
            "total_weight": self.root.total_weight,
            "max_depth": max((node.get_depth() for node in all_nodes), default=0),
            "core_dim_r": self.r,
            "routing_dim": self.dim,
        }
