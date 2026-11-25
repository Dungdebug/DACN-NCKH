"""
Unified Reasoning Loop for WaveCMS-R^2 (Ticket T8 / w7-reason-loop).

100% offline. Reuses existing read-only modules under src/reasoning/ and
src/hybrid_qa/grounding_critic.py:
  - QueryDecomposer
  - ConceptHypergraph
  - MultiHopEvidenceDAG
  - PremiseValidator
  - BoundaryInspector
  - GroundingCritic
  - symbolic_solver (solve_quadratic, balance_equation, caesar, vigenere,
                     affine, verify_numeric_provenance)
"""

from __future__ import annotations

import inspect
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from src.hybrid_qa.grounding_critic import GroundingCritic
from src.reasoning.boundary_inspector import BoundaryInspector
from src.reasoning.dual_process import PremiseValidator
from src.reasoning.hypergraph import ConceptHypergraph
from src.reasoning.multi_hop_dag import MultiHopEvidenceDAG
from src.reasoning.query_decomposer import QueryDecomposer
from src.reasoning.symbolic_solver import (
    affine,
    balance_equation,
    caesar,
    solve_quadratic,
    verify_numeric_provenance,
    vigenere,
)

VALID_STEP_KINDS = frozenset({"premise", "mechanism", "formula", "critique", "conclusion"})
VALID_FACETS = (
    "definition",
    "mechanism",
    "relation_comparison",
    "application",
    "limitation",
)
VALID_RESOLUTIONS = ("LEAF_PARAGRAPH", "SECTION", "DOCUMENT", "CORPUS")
VALID_FACET_STATUSES = frozenset({"ok", "dropped"})
DROPPED_FACET_NOTE = "không đủ căn cứ trong tài liệu"


# ---------------------------------------------------------------------------
# API Contract §5 Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReasoningStep:
    kind: str  # "premise" | "mechanism" | "formula" | "critique" | "conclusion"
    text: str
    citations: List[Any] = field(default_factory=list)
    verified: bool = False

    def __post_init__(self) -> None:
        if self.kind not in VALID_STEP_KINDS:
            raise ValueError(
                f"Invalid ReasoningStep.kind '{self.kind}'. Must be one of {sorted(VALID_STEP_KINDS)}."
            )


@dataclass
class FacetEvidence:
    facet: str  # "definition" | "mechanism" | "relation_comparison" | "application" | "limitation"
    resolution: str  # "LEAF_PARAGRAPH" | "SECTION" | "DOCUMENT" | "CORPUS"
    text: str
    citations: List[Any] = field(default_factory=list)
    status: str = "ok"  # "ok" | "dropped"

    def __post_init__(self) -> None:
        if self.facet not in VALID_FACETS:
            raise ValueError(
                f"Invalid FacetEvidence.facet '{self.facet}'. Must be one of {VALID_FACETS}."
            )
        if self.resolution not in VALID_RESOLUTIONS:
            raise ValueError(
                f"Invalid FacetEvidence.resolution '{self.resolution}'. Must be one of {VALID_RESOLUTIONS}."
            )
        if self.status not in VALID_FACET_STATUSES:
            raise ValueError(
                f"Invalid FacetEvidence.status '{self.status}'. Must be one of {sorted(VALID_FACET_STATUSES)}."
            )


@dataclass
class ReasoningTrace:
    mode: str
    hops: int
    rho_hat: float
    rounds: int
    conformal: Dict[str, Any] = field(default_factory=dict)
    steps: List[ReasoningStep] = field(default_factory=list)
    facets: List[FacetEvidence] = field(default_factory=list)
    removed_claims: List[str] = field(default_factory=list)
    answer: str = ""
    solver_traces: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Submodular Greedy Selection (Proposition 6: >= (1 - 1/e) OPT)
# ---------------------------------------------------------------------------


def _extract_cover_set(candidate: Any) -> Set[Any]:
    """Extract the coverage set from a candidate dict or object."""
    if isinstance(candidate, dict):
        for key in ("cover", "covers", "elements", "concepts", "facets"):
            val = candidate.get(key)
            if val is not None:
                return set(val)
        # Fallback: derive coverage from facet, resolution, doc_id, and content tokens
        covered: Set[Any] = set()
        if candidate.get("facet"):
            covered.add(("facet", candidate["facet"]))
        if candidate.get("resolution"):
            covered.add(("resolution", candidate["resolution"]))
        if candidate.get("document_id"):
            covered.add(("doc", candidate["document_id"]))
        text = str(candidate.get("text", ""))
        for tok in re.findall(r"\w{4,}", text.lower())[:12]:
            covered.add(("tok", tok))
        return covered

    for attr in ("cover", "covers", "elements", "concepts"):
        if hasattr(candidate, attr):
            val = getattr(candidate, attr)
            if val is not None:
                return set(val)
    if isinstance(candidate, (set, frozenset, list, tuple)):
        return set(candidate)
    return {candidate}


def greedy_submodular_select(
    candidates: List[Any],
    budget: int,
    item_weights: Optional[Dict[Any, float]] = None,
) -> List[Any]:
    """
    Greedy monotone submodular maximization for coverage F(S) = sum_{u in U(S)} w(u).
    Guarantees F(S_greedy) >= (1 - 1/e) * F(OPT) for cardinality constraint |S| <= budget.
    """
    if budget <= 0 or not candidates:
        return []

    cand_covers: List[Set[Any]] = [_extract_cover_set(c) for c in candidates]
    selected: List[Any] = []
    selected_indices: Set[int] = set()
    covered_so_far: Set[Any] = set()

    def _weight(elem: Any) -> float:
        if item_weights is None:
            return 1.0
        return float(item_weights.get(elem, 1.0))

    steps = min(budget, len(candidates))
    for _ in range(steps):
        best_idx = -1
        best_gain = 0.0
        for idx, cov in enumerate(cand_covers):
            if idx in selected_indices:
                continue
            new_elems = cov - covered_so_far
            gain = sum(_weight(e) for e in new_elems)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx < 0 or best_gain <= 0.0:
            break
        selected_indices.add(best_idx)
        selected.append(candidates[best_idx])
        covered_so_far |= cand_covers[best_idx]

    return selected


# ---------------------------------------------------------------------------
# UnifiedReasoner
# ---------------------------------------------------------------------------

_FACET_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "definition": ("là gì", "định nghĩa", "khái niệm", "là một", "bao gồm", "gồm"),
    "mechanism": ("cơ chế", "hoạt động", "thuật toán", "phương pháp", "kiến trúc", "quy trình", "bằng cách"),
    "relation_comparison": ("so sánh", "khác", "vượt trội", "tốt hơn", "liên hệ", "tương quan", "thay thế"),
    "application": ("ứng dụng", "áp dụng", "sử dụng", "triển khai", "thực tế", "kết quả", "đạt"),
    "limitation": ("hạn chế", "nhược điểm", "rủi ro", "thách thức", "điều kiện", "chỉ khi", "tuy nhiên", "giới hạn"),
}

_QUAD_RE = re.compile(
    r"(?:([+-]?\s*\d+(?:\.\d+)?)\s*\*?\s*)?x\s*\^\s*2"
    r"(?:\s*([+-]\s*(?:\d+(?:\.\d+)?\s*\*?\s*)?x))?"
    r"(?:\s*([+-]\s*\d+(?:\.\d+)?))?\s*=\s*0",
    re.IGNORECASE,
)

_CHEM_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9()\[\]]*(?:\s*\+\s*[A-Z][A-Za-z0-9()\[\]]*)*\s*(?:-->|->|→)\s*"
    r"[A-Z][A-Za-z0-9()\[\]]*(?:\s*\+\s*[A-Z][A-Za-z0-9()\[\]]*)*)"
)


def _parse_coeff(raw: Optional[str], default_if_empty: float = 1.0) -> float:
    if raw is None:
        return 0.0
    s = raw.replace(" ", "").replace("*", "").replace("x", "").replace("X", "")
    if s in ("", "+"):
        return default_if_empty
    if s == "-":
        return -default_if_empty
    return float(s)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class UnifiedReasoner:
    """
    Unified System 1 (fast) / System 2 (deep, broad, deep_broad) reasoning engine.
    """

    def __init__(
        self,
        retrieve_fn: Callable[..., List[Dict[str, Any]]],
        gate_fn: Optional[Callable[..., Any]] = None,
        solver: Optional[Any] = None,
        max_deep_rounds: int = 2,
        broad_budget: int = 5,
    ) -> None:
        self.retrieve_fn = retrieve_fn
        self.gate_fn = gate_fn
        self.solver = solver
        self.max_deep_rounds = max(1, int(max_deep_rounds))
        self.broad_budget = max(1, int(broad_budget))

        self._decomposer = QueryDecomposer()
        self._premise_validator = PremiseValidator()
        self._dag_builder = MultiHopEvidenceDAG()
        self._boundary_inspector = BoundaryInspector()
        self._critic = GroundingCritic()

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _normalize_docs(
        self, docs: Optional[Iterable[Dict[str, Any]]], default_res: str = "LEAF_PARAGRAPH"
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not docs:
            return normalized
        for idx, raw in enumerate(docs):
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or raw.get("content") or "").strip()
            if not text:
                continue
            p_id = str(raw.get("passage_id") or raw.get("id") or f"doc_{idx}")
            cit = str(raw.get("citation_label") or f"[{idx + 1}]")
            res = str(raw.get("resolution") or default_res)
            if res not in VALID_RESOLUTIONS:
                res = default_res
            doc_id = str(raw.get("document_id") or raw.get("doc_id") or p_id)
            sec = str(raw.get("section_title") or raw.get("section") or f"Section {idx + 1}")
            normalized.append(
                {
                    **raw,
                    "passage_id": p_id,
                    "document_id": doc_id,
                    "section_title": sec,
                    "citation_label": cit,
                    "resolution": res,
                    "text": text,
                }
            )
        return normalized

    def _call_retrieve(
        self,
        query: str,
        top_k: int = 5,
        resolution: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            sig = inspect.signature(self.retrieve_fn)
            kwargs: Dict[str, Any] = {}
            if "top_k" in sig.parameters:
                kwargs["top_k"] = top_k
            if resolution is not None and "resolution" in sig.parameters:
                kwargs["resolution"] = resolution
            res = self.retrieve_fn(query, **kwargs)
        except (TypeError, ValueError):
            res = self.retrieve_fn(query)
        return self._normalize_docs(res, default_res=resolution or "LEAF_PARAGRAPH")

    @staticmethod
    def _merge_evidence(
        existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Merge evidence lists preserving order; return (merged, changed)."""
        seen_keys = {
            (d["passage_id"], d["text"]) for d in existing
        }
        merged = list(existing)
        changed = False
        for d in incoming:
            key = (d["passage_id"], d["text"])
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(d)
                changed = True
        return merged, changed

    def _run_symbolic_solver(
        self, query: str, evidence: List[Dict[str, Any]]
    ) -> List[str]:
        traces: List[str] = []
        combined = " \n ".join([query] + [d["text"] for d in evidence])

        # 1. Quadratic equations ax^2 + bx + c = 0
        for m in _QUAD_RE.finditer(combined):
            try:
                a_val = _parse_coeff(m.group(1) if m.group(1) is not None else "1", 1.0)
                b_val = _parse_coeff(m.group(2), 1.0) if m.group(2) is not None else 0.0
                c_val = _parse_coeff(m.group(3), 0.0) if m.group(3) is not None else 0.0
                if a_val != 0:
                    res = solve_quadratic(a_val, b_val, c_val)
                    traces.append(
                        f"solve_quadratic({a_val:g}, {b_val:g}, {c_val:g}): "
                        f"delta={res['delta']}, nature={res['nature']}, "
                        f"x1={res['x1']}, x2={res['x2']}"
                    )
            except Exception:
                continue

        # 2. Chemical equations
        for m in _CHEM_RE.finditer(combined):
            eq_str = m.group(1).strip()
            try:
                res = balance_equation(eq_str)
                traces.append(f"balance_equation({eq_str}): {res['balanced']}")
            except Exception:
                continue

        # 3. Classical ciphers (Caesar / Vigenere / Affine)
        q_low = combined.lower()
        caesar_m = re.search(
            r"caesar\s*\(\s*['\"]([A-Za-z\s]+)['\"]\s*,\s*(-?\d+)(?:\s*,\s*decrypt\s*=\s*(True|False))?\s*\)",
            combined,
            re.IGNORECASE,
        )
        if caesar_m:
            txt, sh = caesar_m.group(1), int(caesar_m.group(2))
            dec = (caesar_m.group(3) or "").lower() == "true" or "decrypt" in q_low or "giải mã" in q_low
            traces.append(f"caesar({txt!r}, {sh}, decrypt={dec}): {caesar(txt, sh, decrypt=dec)}")

        vig_m = re.search(
            r"vigen[eè]re\s*\(\s*['\"]([A-Za-z\s]+)['\"]\s*,\s*['\"]([A-Za-z]+)['\"](?:\s*,\s*decrypt\s*=\s*(True|False))?\s*\)",
            combined,
            re.IGNORECASE,
        )
        if vig_m:
            txt, key = vig_m.group(1), vig_m.group(2)
            dec = (vig_m.group(3) or "").lower() == "true" or "decrypt" in q_low or "giải mã" in q_low
            traces.append(f"vigenere({txt!r}, {key!r}, decrypt={dec}): {vigenere(txt, key, decrypt=dec)}")

        aff_m = re.search(
            r"affine\s*\(\s*['\"]([A-Za-z\s]+)['\"]\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*decrypt\s*=\s*(True|False))?\s*\)",
            combined,
            re.IGNORECASE,
        )
        if aff_m:
            txt, a_p, b_p = aff_m.group(1), int(aff_m.group(2)), int(aff_m.group(3))
            dec = (aff_m.group(4) or "").lower() == "true" or "decrypt" in q_low or "giải mã" in q_low
            try:
                traces.append(
                    f"affine({txt!r}, {a_p}, {b_p}, decrypt={dec}): {affine(txt, a_p, b_p, decrypt=dec)}"
                )
            except Exception:
                pass

        # 4. Optional injected solver hook
        if callable(self.solver):
            try:
                extra = self.solver(query, evidence)
                if isinstance(extra, str) and extra.strip():
                    traces.append(extra.strip())
                elif isinstance(extra, list):
                    traces.extend(str(x).strip() for x in extra if str(x).strip())
            except Exception:
                pass

        return list(dict.fromkeys(traces))

    def _filter_ungrounded_numbers(
        self,
        text: str,
        evidence: List[Dict[str, Any]],
        solver_traces: List[str],
        query: str = "",
    ) -> Tuple[str, List[str]]:
        """
        Enforce Numeric Provenance Invariant (§3.5):
        Remove any sentence containing numbers not in Num(evidence) U Num(solver_traces).
        For natural QA (no solver_traces), also allow numbers grounded in `query` or
        equivalent unsigned/thousand-separator forms in evidence/query.
        """
        if not text.strip():
            return "", []
        ev_texts = [d["text"] for d in evidence]
        if query and not solver_traces:
            ev_texts.append(query)

        fallback_nums: Optional[Set[str]] = None
        kept: List[str] = []
        removed: List[str] = []
        for sent in _split_sentences(text):
            ok, ungrounded = verify_numeric_provenance(sent, ev_texts, solver_trace=solver_traces)
            if not ok and not solver_traces:
                if fallback_nums is None:
                    joined = " ".join(ev_texts)
                    stripped = re.sub(r"(?<=\d)[.,](?=\d{3}(?!\d))", "", joined)
                    fallback_nums = {
                        tok.replace(",", ".")
                        for src in (joined, stripped)
                        for tok in re.findall(r"(?<![A-Za-z0-9_])\d+(?:[.,]\d+)*(?![A-Za-z0-9_])", src)
                    }
                if all(u in fallback_nums for u in ungrounded):
                    ok = True
            if ok:
                kept.append(sent)
            else:
                removed.append(sent)
        return " ".join(kept).strip(), removed

    def _check_gate(self, facet: str, candidates: List[Dict[str, Any]]) -> bool:
        if self.gate_fn is None:
            return len(candidates) > 0
        try:
            res = self.gate_fn(facet, candidates)
        except TypeError:
            try:
                res = self.gate_fn(candidates)
            except TypeError:
                res = self.gate_fn(facet)
        if isinstance(res, tuple):
            return bool(res[0])
        if isinstance(res, dict):
            return bool(res.get("passed", res.get("ok", False)))
        return bool(res)

    @staticmethod
    def _classify_facet(text: str) -> str:
        t_low = text.lower()
        best_facet = "definition"
        best_hits = 0
        for facet, kws in _FACET_KEYWORDS.items():
            hits = sum(1 for kw in kws if kw in t_low)
            if hits > best_hits:
                best_hits = hits
                best_facet = facet
        return best_facet

    # ------------------------------------------------------------------
    # Mode Implementations
    # ------------------------------------------------------------------

    def _run_fast(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        rho_hat: float,
        conformal: Dict[str, Any],
    ) -> ReasoningTrace:
        retrieved = self._call_retrieve(query, top_k=3)
        evidence, _ = self._merge_evidence(evidence, retrieved)
        solver_traces = self._run_symbolic_solver(query, evidence)

        if evidence:
            raw_answer = " ".join(f"{d['text']} {d['citation_label']}" for d in evidence[:2])
        else:
            raw_answer = ""

        if solver_traces:
            raw_answer = f"{raw_answer} {' '.join(solver_traces)}".strip()

        clean_answer, removed = self._filter_ungrounded_numbers(
            raw_answer, evidence, solver_traces, query=query
        )
        if not clean_answer and solver_traces:
            clean_answer = " ".join(solver_traces)

        cits = [d["citation_label"] for d in evidence[:2]]
        verified = bool(clean_answer) and len(removed) == 0
        steps = [
            ReasoningStep(
                kind="conclusion",
                text=clean_answer or DROPPED_FACET_NOTE,
                citations=cits,
                verified=verified,
            )
        ]
        return ReasoningTrace(
            mode="fast",
            hops=1,
            rho_hat=rho_hat,
            rounds=1,
            conformal=conformal,
            steps=steps,
            facets=[],
            removed_claims=removed,
            answer=clean_answer or DROPPED_FACET_NOTE,
            solver_traces=solver_traces,
        )

    def _run_deep(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        rho_hat: float,
        conformal: Dict[str, Any],
        mode_name: str = "deep",
    ) -> ReasoningTrace:
        decomp = self._decomposer.decompose(query)
        sub_queries = [sq.text for sq in decomp.sub_queries] or [query]

        # Initial retrieval (round 1)
        for sq in sub_queries:
            evidence, _ = self._merge_evidence(evidence, self._call_retrieve(sq, top_k=4))
        rounds = 1

        # Premise validation guard
        premise_res = self._premise_validator.validate_premise(query, evidence)
        if premise_res.has_false_premise:
            ref_step = ReasoningStep(
                kind="premise",
                text=premise_res.refutation_text,
                citations=[premise_res.counter_citation] if premise_res.counter_citation else [],
                verified=True,
            )
            conc_step = ReasoningStep(
                kind="conclusion",
                text=premise_res.refutation_text,
                citations=[premise_res.counter_citation] if premise_res.counter_citation else [],
                verified=True,
            )
            return ReasoningTrace(
                mode=mode_name,
                hops=1,
                rho_hat=rho_hat,
                rounds=rounds,
                conformal=conformal,
                steps=[ref_step, conc_step],
                facets=[],
                removed_claims=[],
                answer=premise_res.refutation_text,
                solver_traces=[],
            )

        solver_traces = self._run_symbolic_solver(query, evidence)
        steps: List[ReasoningStep] = []
        removed_claims: List[str] = []

        def _build_draft(ev: List[Dict[str, Any]]) -> Tuple[str, int, List[ReasoningStep]]:
            dag = self._dag_builder.build_dag(ev, query=query)
            chain = self._dag_builder.find_optimal_evidence_chain(dag, max_hops=3)
            local_steps: List[ReasoningStep] = []
            for node in chain.nodes:
                if node.role in ("cause", "premise"):
                    kind = "premise"
                elif node.role == "limitation":
                    kind = "critique"
                else:
                    kind = "mechanism"
                local_steps.append(
                    ReasoningStep(
                        kind=kind,
                        text=node.text,
                        citations=[node.citation_label],
                        verified=True,
                    )
                )
            narrative = self._dag_builder.compose_chain_narrative(chain)
            if not narrative and ev:
                narrative = " ".join(f"{d['text']} {d['citation_label']}" for d in ev[:2])
            if solver_traces:
                for tr in solver_traces:
                    local_steps.append(
                        ReasoningStep(kind="formula", text=tr, citations=[], verified=True)
                    )
                narrative = f"{narrative} {' '.join(solver_traces)}".strip()
            return narrative, max(1, len(chain.nodes)), local_steps

        draft_answer, hops, steps = _build_draft(evidence)

        # Proposition 7: Bounded Critique-Retrieve loop (stops at E_{r+1} == E_r or rounds == max_deep_rounds)
        while True:
            report = self._critic.verify_answer(draft_answer, evidence)
            num_ok, _ = verify_numeric_provenance(
                draft_answer, [d["text"] for d in evidence], solver_trace=solver_traces
            )
            if report.is_faithful and num_ok:
                break
            if rounds >= self.max_deep_rounds:
                break

            # Retrieve supplementary evidence for unsupported claims or query
            followup_q = (
                report.unsupported_claims[0] if report.unsupported_claims else query
            )
            incoming = self._call_retrieve(followup_q, top_k=3)
            evidence, changed = self._merge_evidence(evidence, incoming)
            rounds += 1
            if not changed:
                # E_{r+1} == E_r -> finite termination
                break

            solver_traces = self._run_symbolic_solver(query, evidence)
            draft_answer, hops, steps = _build_draft(evidence)

        # Final critique & Numeric Provenance Invariant enforcement
        final_report = self._critic.verify_answer(draft_answer, evidence)
        clean_answer, num_removed = self._filter_ungrounded_numbers(
            draft_answer, evidence, solver_traces, query=query
        )
        removed_claims.extend(num_removed)

        if not final_report.is_faithful and final_report.unsupported_claims:
            steps.append(
                ReasoningStep(
                    kind="critique",
                    text=f"Kiểm định GroundingCritic: {len(final_report.unsupported_claims)} nhận định cần đối soát.",
                    citations=[],
                    verified=False,
                )
            )

        if not clean_answer and solver_traces:
            clean_answer = " ".join(solver_traces)

        all_cits = list(dict.fromkeys(d["citation_label"] for d in evidence[:4]))
        steps.append(
            ReasoningStep(
                kind="conclusion",
                text=clean_answer or DROPPED_FACET_NOTE,
                citations=all_cits,
                verified=len(removed_claims) == 0 and bool(clean_answer),
            )
        )

        return ReasoningTrace(
            mode=mode_name,
            hops=hops,
            rho_hat=rho_hat,
            rounds=rounds,
            conformal=conformal,
            steps=steps,
            facets=[],
            removed_claims=removed_claims,
            answer=clean_answer or DROPPED_FACET_NOTE,
            solver_traces=solver_traces,
        )

    def _run_broad(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        rho_hat: float,
        conformal: Dict[str, Any],
    ) -> ReasoningTrace:
        # 1. Gather candidates across 4 Haar resolutions
        for res in VALID_RESOLUTIONS:
            evidence, _ = self._merge_evidence(
                evidence, self._call_retrieve(query, top_k=3, resolution=res)
            )

        # 2. Cross-doc bridges (<= 2 hops) via ConceptHypergraph
        hg = ConceptHypergraph()
        for d in evidence:
            sents = _split_sentences(d["text"])
            hg.extract_from_sentences(
                sents, document_id=d["document_id"], section_title=d["section_title"]
            )
        bridge_texts: List[str] = []
        node_names = [n.name for n in list(hg.nodes.values())[:6]]
        if len(node_names) >= 2:
            for i in range(len(node_names)):
                for j in range(i + 1, len(node_names)):
                    path = hg.find_path(node_names[i], node_names[j], max_hops=2)
                    if path:
                        bridge_texts.append(
                            " -> ".join(f"{e.source_id} ({e.relation_type.value}) {e.target_id}" for e in path)
                        )
                        if len(bridge_texts) >= 2:
                            break
                if len(bridge_texts) >= 2:
                    break

        # 3. Boundary conditions via BoundaryInspector
        boundaries = self._boundary_inspector.extract_boundaries(evidence)

        # 4. Build submodular candidates over (5 facets x 4 resolutions)
        candidates: List[Dict[str, Any]] = []
        for d in evidence:
            facet = d.get("facet")
            if facet not in VALID_FACETS:
                facet = self._classify_facet(d["text"])
            res = d.get("resolution", "LEAF_PARAGRAPH")
            if res not in VALID_RESOLUTIONS:
                res = "LEAF_PARAGRAPH"
            cover: Set[Any] = {("facet", facet), ("resolution", res), ("doc", d["document_id"])}
            for tok in re.findall(r"\w{4,}", d["text"].lower())[:8]:
                cover.add(("tok", tok))
            candidates.append({**d, "facet": facet, "resolution": res, "cover": cover})

        selected = greedy_submodular_select(candidates, self.broad_budget)

        # Group selected candidates by facet
        by_facet: Dict[str, List[Dict[str, Any]]] = {f: [] for f in VALID_FACETS}
        for c in selected:
            by_facet[c["facet"]].append(c)

        facets_out: List[FacetEvidence] = []
        ok_texts: List[str] = []
        for facet in VALID_FACETS:
            facet_cands = by_facet[facet]
            if not facet_cands:
                facets_out.append(
                    FacetEvidence(
                        facet=facet,
                        resolution="LEAF_PARAGRAPH",
                        text=DROPPED_FACET_NOTE,
                        citations=[],
                        status="dropped",
                    )
                )
                continue

            passed = self._check_gate(facet, facet_cands)
            top_res = facet_cands[0]["resolution"]
            if not passed:
                facets_out.append(
                    FacetEvidence(
                        facet=facet,
                        resolution=top_res,
                        text=DROPPED_FACET_NOTE,
                        citations=[],
                        status="dropped",
                    )
                )
            else:
                combined_text = " ".join(
                    f"{fc['text']} {fc['citation_label']}" for fc in facet_cands
                )
                cits = [fc["citation_label"] for fc in facet_cands]
                facets_out.append(
                    FacetEvidence(
                        facet=facet,
                        resolution=top_res,
                        text=combined_text,
                        citations=cits,
                        status="ok",
                    )
                )
                ok_texts.append(combined_text)

        if boundaries and ok_texts:
            b0 = boundaries[0]
            ok_texts.append(f"Điều kiện biên: {b0.condition_text} {b0.citation_label}")
        if bridge_texts and ok_texts:
            ok_texts.append(f"Liên kết khái niệm: {bridge_texts[0]}")

        solver_traces = self._run_symbolic_solver(query, evidence)
        parts: List[str] = []
        if solver_traces:
            parts.extend(solver_traces)
        parts.extend(ok_texts)
        raw_answer = " ".join(parts).strip()
        clean_answer, removed = self._filter_ungrounded_numbers(
            raw_answer, evidence, solver_traces, query=query
        )
        if not clean_answer and solver_traces:
            clean_answer = " ".join(solver_traces)

        steps: List[ReasoningStep] = [
            ReasoningStep(kind="formula", text=tr, citations=[], verified=True)
            for tr in solver_traces
        ]
        steps.append(
            ReasoningStep(
                kind="conclusion",
                text=clean_answer or DROPPED_FACET_NOTE,
                citations=list(dict.fromkeys(c for f in facets_out if f.status == "ok" for c in f.citations)),
                verified=bool(clean_answer) and len(removed) == 0,
            )
        )
        return ReasoningTrace(
            mode="broad",
            hops=min(2, max(1, len(bridge_texts))),
            rho_hat=rho_hat,
            rounds=1,
            conformal=conformal,
            steps=steps,
            facets=facets_out,
            removed_claims=removed,
            answer=clean_answer or DROPPED_FACET_NOTE,
            solver_traces=solver_traces,
        )

    # ------------------------------------------------------------------
    # Public Entry Point
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        mode: str = "fast",
        context_docs: Optional[List[Dict[str, Any]]] = None,
        base_routing: Optional[Dict[str, Any]] = None,
    ) -> ReasoningTrace:
        routing = base_routing or {}
        rho_hat = float(routing.get("rho_hat", 0.0))
        conformal = dict(routing.get("conformal", {}))
        evidence = self._normalize_docs(context_docs)

        if mode == "fast":
            return self._run_fast(query, evidence, rho_hat, conformal)
        if mode == "deep":
            return self._run_deep(query, evidence, rho_hat, conformal, mode_name="deep")
        if mode == "broad":
            return self._run_broad(query, evidence, rho_hat, conformal)
        if mode == "deep_broad":
            broad_trace = self._run_broad(query, evidence, rho_hat, conformal)
            ok_facets = [f for f in broad_trace.facets if f.status == "ok"]
            ok_docs = [
                {
                    "passage_id": f"facet_{f.facet}",
                    "document_id": f"facet_{f.facet}",
                    "section_title": f.facet,
                    "citation_label": f.citations[0] if f.citations else f"[{i + 1}]",
                    "resolution": f.resolution,
                    "text": f.text,
                }
                for i, f in enumerate(ok_facets)
            ]
            deep_trace = self._run_deep(
                query, ok_docs, rho_hat, conformal, mode_name="deep_broad"
            )
            deep_trace.facets = broad_trace.facets
            deep_trace.removed_claims = list(
                dict.fromkeys(broad_trace.removed_claims + deep_trace.removed_claims)
            )
            return deep_trace

        raise ValueError(
            f"Unsupported reasoning mode '{mode}'. Expected 'fast', 'deep', 'broad', or 'deep_broad'."
        )
