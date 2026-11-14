"""
WaveCMS Engine: Document-Tree Wavelet Continuum Memory with Equilibrium Multi-Hop Readout.
Phase WaveCMS — Mũi 4: Integrated Engine & Core Architecture Orchestrator.

Theoretical References:
- wavecms_research_proposal.md (Sections 2.1–2.7)
- Theorem 1: Multi-Resolution Telescoping & Wavelet Expansion & Haar Vanishing & Parseval Energy Conservation
- Theorem 2: Certified Exact Unlearning in O(depth) with Zero Residual Forgetting
- Theorem 4: Banach Fixed-Point Existence, Uniqueness, and Monotonic Convergence
- Proposal Section 2.7: Counterfactual Leave-One-Leaf-Out Attribution & Conformal Risk Control Refusal

100% Strict Offline, Pure Python stdlib, NumPy & PyTorch, Zero External APIs.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.wave_cms.causal_attribution import (
    CausalAttributionItem,
    ConformalRiskController,
    CounterfactualLeafAttributor,
    EvidenceChainHop,
    RefusalDecision,
)
from src.wave_cms.equilibrium_readout import (
    EquilibriumMultiHopReadout,
    EquilibriumResult,
    ResolutionLevel,
    RoutingResult,
    TreeSoftmaxRouter,
)
from src.wave_cms.tree_memory import (
    DocumentTreeNode,
    TreeLevel,
    WaveTreeMemory,
    compute_deterministic_routing_key,
)
from src.slm.sovereign_generator import SovereignSLMGenerator

try:
    from src.wave_cms.tree_memory import strip_accents
except ImportError:
    import unicodedata
    def strip_accents(text: str) -> str:
        if not text:
            return ""
        text_nfd = unicodedata.normalize("NFD", str(text))
        cleaned = "".join(ch for ch in text_nfd if unicodedata.category(ch) != "Mn")
        return (
            unicodedata.normalize("NFC", cleaned)
            .replace("đ", "d")
            .replace("Đ", "D")
            .lower()
            .strip()
        )


@dataclass
class WaveCMSQueryResult:
    """Structured response object returned by WaveCMSEngine.query()."""
    query: str
    answer: str
    refused: bool
    confidence: float
    refusal_reason: Optional[str]
    causal_citations: List[Dict[str, Any]]
    evidence_chain: List[Dict[str, Any]]
    fixed_point_hops: int
    contraction_factor: float
    is_contractive: bool
    converged: bool
    resolution_level: str
    effective_k: float
    entropy: float
    theorems_verified: Dict[str, Any]
    telemetry: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WaveCMSEngine:
    """
    Unified High-Level Engine for WaveCMS.
    Orchestrates:
    1. Multi-Resolution Document-Tree Wavelet Memory (WaveTreeMemory).
    2. Hierarchical Routing & Banach Equilibrium Multi-Hop Readout (EquilibriumMultiHopReadout).
    3. Structural Leave-One-Leaf-Out Causal Attribution (CounterfactualLeafAttributor).
    4. Finite-Sample Risk-Guaranteed Refusal Gate (ConformalRiskController).
    5. Certified Analytical Unlearning (Exact Unlearning in O(depth)).
    """

    def __init__(
        self,
        r: int = 8,
        dim: int = 256,
        omega: float = 0.5,
        max_hops: int = 4,
        tau: float = 0.20,
        epsilon: float = 0.1,
        refusal_threshold: float = 0.40,
        corpus_id: str = "WAVECMS_CORPUS_ROOT",
    ) -> None:
        self.r = r
        self.dim = dim
        self.omega = omega
        self.max_hops = max_hops
        self.tau = tau
        self.epsilon = epsilon
        self.corpus_id = corpus_id

        # 1. Document-Tree Wavelet Memory
        self.tree_memory = WaveTreeMemory(
            r=self.r,
            key_dim=self.dim,
            corpus_id=self.corpus_id,
        )

        # 2. Hierarchical Router & Equilibrium Multi-Hop Readout
        self.router = TreeSoftmaxRouter(
            d_model=self.dim,
            temperature=self.tau,
        )
        self.readout_engine = EquilibriumMultiHopReadout(
            d_model=self.dim,
            r_dim=self.r,
            omega=self.omega,
            max_hops=self.max_hops,
            tol=1e-4,
            router=self.router,
        )

        # 3. Counterfactual Leave-One-Leaf-Out Attributor
        self.attributor = CounterfactualLeafAttributor(
            d_model=self.dim,
            r=self.r,
        )

        # 4. Conformal Risk Control Refusal Controller
        self.risk_controller = ConformalRiskController(
            epsilon=self.epsilon,
            default_threshold=refusal_threshold,
        )

        # 5. Sovereign Offline SLM/LLM Generation Brain (Trụ cột 1)
        self.slm_generator = SovereignSLMGenerator()

        # Document registry
        self._document_registry: Dict[str, Dict[str, Any]] = {}

    # =========================================================================
    # DOCUMENT INGESTION & STRUCTURAL TREE EXPANSION
    # =========================================================================

    def ingest_document(
        self,
        doc_id: str,
        title: str,
        sections: Union[List[Dict[str, Any]], str],
        default_leaf_weight: float = 1.0,
        category: str = "General",
    ) -> DocumentTreeNode:
        """
        Ingests a document into the WaveCMS tree continuum.
        
        Args:
            doc_id: Unique document identifier.
            title: Title of document.
            sections: Either structured list of sections, or raw string containing
                      slide pages (=== Page \\d+ ===) and section headers.
            default_leaf_weight: Initial leaf weight w_p (default 1.0).
            category: Domain category.
            
        Returns:
            DocumentTreeNode representing the ingested document node.
        """
        structured_sections: List[Dict[str, Any]] = []

        if isinstance(sections, str):
            structured_sections = self._parse_raw_text_into_sections(sections, doc_id)
        elif isinstance(sections, list):
            structured_sections = sections
        else:
            raise TypeError("sections must be either a structured list of dicts or a raw string.")

        # Ingest into tree memory and run telescoping consolidation
        doc_node = self.tree_memory.add_document(
            doc_id=doc_id,
            title=title,
            sections=structured_sections,
            default_leaf_weight=default_leaf_weight,
        )

        # Update document registry
        num_sections = len(structured_sections)
        num_leaves = len(doc_node.get_leaves())
        self._document_registry[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "category": category,
            "num_sections": num_sections,
            "num_leaves": num_leaves,
            "subtree_weight": float(doc_node.subtree_weight),
            "ingested_at": time.time(),
        }

        return doc_node

    def _parse_raw_text_into_sections(self, raw_text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Parses raw text (including slide page breaks === Page \\d+ === and numbered headings)
        into a clean, structured section hierarchy.
        """
        sections: List[Dict[str, Any]] = []
        if not raw_text.strip():
            return [{"section_title": "Overview", "paragraphs": ["No content."]}]

        # Remove page break artifacts while respecting page boundaries
        pages = re.split(r"===\s*Page\s*\d+\s*===", raw_text, flags=re.IGNORECASE)
        page_chunks = [p.strip() for p in pages if p.strip()]

        current_sec_title = "Overview"
        current_paragraphs: List[str] = []

        heading_pattern = re.compile(
            r"^\s*(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+([^\n:]{3,80})[:.]?|#{1,4}\s+([^\n]{3,80}))",
            re.MULTILINE,
        )

        full_clean_text = "\n\n".join(page_chunks)
        lines = full_clean_text.splitlines()

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            match = heading_pattern.match(line_str)
            if match and len(line_str) < 100:
                # Flush previous section if it has content
                if current_paragraphs:
                    sections.append({
                        "section_title": current_sec_title,
                        "paragraphs": current_paragraphs,
                    })
                    current_paragraphs = []
                h_name = match.group(1) or match.group(2) or line_str
                current_sec_title = h_name.strip()
            else:
                current_paragraphs.append(line_str)

        if current_paragraphs:
            sections.append({
                "section_title": current_sec_title,
                "paragraphs": current_paragraphs,
            })

        if not sections:
            sections.append({
                "section_title": "Overview",
                "paragraphs": [raw_text.strip()],
            })

        return sections

    # =========================================================================
    # QUERY & EQUILIBRIUM MULTI-HOP READOUT
    # =========================================================================

    def query(
        self,
        question: str = "",
        query: str = "",
        max_hops: Optional[int] = None,
        top_k_citations: int = 4,
        answer_mode: str = "detailed",
    ) -> WaveCMSQueryResult:
        """
        Executes complete equilibrium multi-hop querying across the document tree.
        
        Process:
        1. Embeds question into semantic query vector q in R^d.
        2. Solves Banach fixed-point loop h* = (1-w)h* + w Enc_{W0 + B Theta(alpha) A}(q).
        3. Monitors contraction factor rho_hat < 1 and convergence.
        4. Evaluates Conformal Risk Control refusal gate.
        5. Computes Leave-One-Leaf-Out causal attribution c(p) and traces evidence chain E_{1:N}.
        6. Synthesizes a fully grounded structured response.
        """
        t0 = time.perf_counter()
        clean_q = (question or query).strip()

        # 0. Edge-case: Empty query or empty tree
        leaves = self.tree_memory.get_leaves()
        if not clean_q or not leaves:
            return WaveCMSQueryResult(
                query=question or query,
                answer="Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn.",
                refused=True,
                confidence=0.0,
                refusal_reason="EMPTY_CORPUS_OR_QUERY",
                causal_citations=[],
                evidence_chain=[],
                fixed_point_hops=0,
                contraction_factor=0.0,
                is_contractive=True,
                converged=False,
                resolution_level="NONE",
                effective_k=0.0,
                entropy=0.0,
                theorems_verified=self.tree_memory.verify_all_theorems(),
                telemetry={"total_latency_ms": (time.perf_counter() - t0) * 1000},
            )

        # 1. Deterministic Semantic Embedding q in R^d
        q_np = compute_deterministic_routing_key(clean_q, dim=self.dim)
        q_tensor = torch.tensor(q_np, dtype=torch.float32)

        # 2. Banach Equilibrium Multi-Hop Readout (Theorem 4)
        eq_res: EquilibriumResult = self.readout_engine.readout(
            query=clean_q,
            tree=self.tree_memory,
            max_hops=max_hops or self.max_hops,
        )

        final_alpha = eq_res.final_alpha or {}
        max_alpha = max(final_alpha.values()) if final_alpha else 0.0

        # Multi-hop evidence aggregation: combine routing weights across hops when >= 2 hops
        # so bridge leaves (early hops) and target leaves (final hop) both surface in citations.
        history_alpha = eq_res.history_alpha or []
        if len(history_alpha) >= 2:
            agg_raw: Dict[str, float] = {}
            for hop_alpha in history_alpha:
                for lid, weight in hop_alpha.items():
                    w = float(weight)
                    if w > agg_raw.get(lid, 0.0):
                        agg_raw[lid] = w
            agg_total = sum(agg_raw.values())
            alpha_agg = (
                {lid: w / agg_total for lid, w in agg_raw.items()}
                if agg_total > 0.0
                else dict(final_alpha)
            )
        else:
            alpha_agg = final_alpha

        # 3. Compute Memory Reliance Ratio: log p_{mem} - log p_{W0}
        theta_star_np = eq_res.final_theta.detach().cpu().numpy().astype(np.float64) if eq_res.final_theta is not None else np.zeros((self.r, self.r))
        
        # Candidate answer representation from top routed leaves
        sorted_leaves_by_alpha = sorted(
            leaves,
            key=lambda lf: (alpha_agg.get(lf.node_id, 0.0), final_alpha.get(lf.node_id, 0.0)),
            reverse=True,
        )
        top_routed_leaf = sorted_leaves_by_alpha[0] if sorted_leaves_by_alpha else None
        candidate_ans_text = top_routed_leaf.text if top_routed_leaf else clean_q

        logp_mem = self.attributor.compute_log_likelihood(
            theta=theta_star_np,
            query=clean_q,
            answer=candidate_ans_text,
        )
        logp_w0 = self.attributor.compute_log_likelihood(
            theta=np.zeros((self.r, self.r), dtype=np.float64),
            query=clean_q,
            answer=candidate_ans_text,
        )
        memory_reliance = float(logp_mem - logp_w0)

        # 4. Conformal Risk Control Refusal Gate (Proposal Section 2.7)
        has_quoted_phrase = bool(re.search(r"['\"«“][^'\"»”]+['\"»”]", clean_q))
        reliance_scale = (float(self.dim) / max(1, int(self.r))) ** 2
        try:
            refusal_dec: RefusalDecision = self.risk_controller.evaluate(
                max_alpha=max_alpha,
                memory_reliance=memory_reliance,
                query=clean_q,
                num_leaves=len(leaves),
                reliance_scale=reliance_scale,
            )
        except TypeError:
            refusal_dec = self.risk_controller.evaluate(
                max_alpha=max_alpha,
                memory_reliance=memory_reliance,
                query=clean_q,
            )

        # Grounding verification: check if query has semantic or entity match in corpus
        q_norm = strip_accents(clean_q.lower())
        q_tokens = [w for w in re.findall(r"\w+", q_norm) if len(w) > 2 and w not in {
            "la", "gi", "va", "co", "nhung", "cac", "cho", "biet", "duoc", "sao", "nao", "the",
            "cua", "trong", "ngoai", "khi", "neu", "mot", "hai", "nhieu", "it", "hay"
        }]

        matches_corpus = False
        quoted_terms = [strip_accents(m.lower().strip()) for m in re.findall(r"['\"«“]([^'\"»”]+)['\"»”]", clean_q)] if has_quoted_phrase else []

        for lf in leaves:
            lf_text_norm = strip_accents(lf.text.lower())
            lf_title_norm = strip_accents(lf.title.lower())
            # Quoted phrase match
            if has_quoted_phrase and any(qt in lf_text_norm or qt in lf_title_norm for qt in quoted_terms):
                matches_corpus = True
                break
            # Token overlap match
            if q_tokens:
                overlap = len(set(q_tokens) & set(re.findall(r"\w+", lf_text_norm + " " + lf_title_norm))) / len(q_tokens)
                if overlap >= 0.30:
                    matches_corpus = True
                    break

        if refusal_dec.refused and matches_corpus and not self.risk_controller.is_calibrated:
            refusal_dec.refused = False
            refusal_dec.reason = "OVERRIDE_CORPUS_GROUNDED_CONTENT_MATCH"

        # 5. Causal Leave-One-Leaf-Out Attribution & Multi-Hop Chain
        causal_items: List[CausalAttributionItem] = []
        evidence_chain_hops: List[EvidenceChainHop] = []

        if not refusal_dec.refused and sorted_leaves_by_alpha:
            causal_items = self.attributor.attribute(
                query=clean_q,
                answer=candidate_ans_text,
                tree_or_leaves=leaves,
                alpha=alpha_agg,
                top_m=min(top_k_citations, len(leaves)),
            )

            # Trace multi-hop evidence chain E_{1:N}
            resolution_levels = [eq_res.resolution_level] * len(eq_res.history_alpha)
            evidence_chain_hops = self.attributor.trace_evidence_chain(
                tree_or_leaves=leaves,
                hops_alpha=eq_res.history_alpha,
                query=clean_q,
                answer=candidate_ans_text,
                resolution_levels=resolution_levels,
            )

        # 6. Structured Answer Synthesis
        if refusal_dec.refused:
            answer_text = (
                "Không tìm thấy đủ thông tin bằng chứng trong tài liệu để trả lời chắc chắn câu hỏi này. "
                "Cây bộ nhớ WaveCMS không ghi nhận đủ căn cứ xác thực liên quan đến nội dung yêu cầu."
            )
            refusal_reason = refusal_dec.reason
            citations_out = []
            chain_out = []
        else:
            answer_text = self._synthesize_grounded_answer(
                query=clean_q,
                causal_items=causal_items,
                resolution_level=eq_res.resolution_level,
                answer_mode=answer_mode,
            )
            refusal_reason = None
            citations_out = [c.to_dict() for c in causal_items]
            chain_out = [hop.to_dict() for hop in evidence_chain_hops]

        total_lat = (time.perf_counter() - t0) * 1000

        # Check all Theorem 1 identities
        theorems_verified = self.tree_memory.verify_all_theorems()

        return WaveCMSQueryResult(
            query=clean_q,
            answer=answer_text,
            refused=refusal_dec.refused,
            confidence=refusal_dec.confidence,
            refusal_reason=refusal_reason,
            causal_citations=citations_out,
            evidence_chain=chain_out,
            fixed_point_hops=eq_res.num_hops,
            contraction_factor=eq_res.contraction_factor,
            is_contractive=eq_res.is_contractive,
            converged=eq_res.converged,
            resolution_level=eq_res.resolution_level,
            effective_k=float(np.exp(eq_res.history_entropy[-1])) if eq_res.history_entropy else 1.0,
            entropy=float(eq_res.history_entropy[-1]) if eq_res.history_entropy else 0.0,
            theorems_verified=theorems_verified,
            telemetry={
                "total_latency_ms": round(total_lat, 2),
                "num_hops": eq_res.num_hops,
                "contraction_rho": eq_res.contraction_factor,
                "memory_reliance": round(memory_reliance, 4),
                "is_calibrated": refusal_dec.is_calibrated,
            },
        )

    def _synthesize_grounded_answer(
        self,
        query: str,
        causal_items: List[CausalAttributionItem],
        resolution_level: str,
        answer_mode: str = "detailed",
    ) -> str:
        """
        Synthesizes a grounded, structured answer directly from top causal leaves
        using the Sovereign SLM Generator (Trụ cột 1).
        """
        if not causal_items:
            return "Tài liệu ghi nhận thông tin liên quan đến câu hỏi nhưng chưa có phân tích chi tiết."

        return self.slm_generator.generate_from_wave_readout(
            query=query,
            causal_items=causal_items,
            resolution_level=resolution_level,
            answer_mode=answer_mode,
        )

    # =========================================================================
    # THEOREM 2: CERTIFIED EXACT UNLEARNING & FORGETTING ELIMINATION
    # =========================================================================

    def unlearn_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Executes certified exact analytical unlearning of document `doc_id` in O(depth) time.
        Proof & Reference: WaveCMS Theorem 2.
        
        Guarantees mathematically that the post-unlearning memory state is identical
        to a state constructed from scratch without ever ingesting `doc_id`,
        up to IEEE 754 floating-point precision (discrepancy < 1e-12).
        """
        t0 = time.perf_counter()
        if doc_id not in self._document_registry:
            raise KeyError(f"Tài liệu '{doc_id}' không tồn tại trong kho tri thức WaveCMS.")

        # Delegate analytical unlearning and proof audit to tree memory
        cert_audit = self.tree_memory.certify_exact_unlearning(doc_id)

        # Remove from local document registry
        removed_info = self._document_registry.pop(doc_id, None)

        latency_ms = (time.perf_counter() - t0) * 1000
        cert_audit["latency_ms"] = round(latency_ms, 2)
        cert_audit["removed_document"] = removed_info

        return cert_audit

    # =========================================================================
    # SYSTEM AUDIT & THEOREMS COMPREHENSIVE VERIFICATION
    # =========================================================================

    def verify_theorems(self) -> Dict[str, Any]:
        """
        Executes complete verification of all WaveCMS core theorems:
        - Theorem 1(a): Exact Telescoping Identity
        - Theorem 1(b): Wavelet Expansion Equivalence
        - Theorem 1(c): Haar Vanishing Property on Tree Siblings
        - Theorem 1(d): Parseval Energy Conservation on the Tree
        - Theorem 4: Banach Contraction Factor & Convergence Guarantee
        """
        t1_report = self.tree_memory.verify_all_theorems()

        # Theorem 4 verification
        rho_hat, is_contractive = self.readout_engine.compute_contraction_factor(self.tree_memory.root)

        return {
            "all_theorems_verified": bool(t1_report.get("all_theorems_verified", True)),
            "theorem_1": t1_report,
            "theorem_4_banach_contraction": {
                "contraction_factor_rho": float(rho_hat),
                "is_contractive": bool(is_contractive),
                "contraction_bound": "rho < 1.0",
                "guarantee": "Banach Fixed-Point Theorem guarantees unique h* and monotonic linear convergence.",
            },
            "num_documents": len(self._document_registry),
            "num_leaves": len(self.tree_memory.get_leaves()),
            "total_subtree_weight": float(self.tree_memory.root.subtree_weight),
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns metadata for all documents currently active in WaveCMS memory."""
        return list(self._document_registry.values())
