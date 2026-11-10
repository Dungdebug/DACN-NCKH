"""
Neuro-Symbolic Multi-Hop Evidence DAG (Ticket 7.1).
Constructs a Directed Acyclic Graph (DAG) connecting multi-passage evidence
fragments across sections (Cause -> Mechanism -> Impact -> Metric -> Limitation),
discovers the optimal reasoning chain, and composes an articulate narrative.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
import re


@dataclass
class EvidenceNode:
    """Represents a single atomic claim or evidence sentence."""
    node_id: str
    text: str
    section_title: str
    citation_label: str
    role: str  # "cause", "mechanism", "impact", "metric", "limitation", "premise", "generic"
    score: float = 1.0
    entities: Set[str] = field(default_factory=set)
    passage_id: str = ""


@dataclass
class EvidenceEdge:
    """Directed connection representing logical or narrative flow."""
    source_id: str
    target_id: str
    edge_type: str  # "causes_or_leads_to", "quantifies", "constrains", "elaborates"
    weight: float = 1.0


@dataclass
class EvidenceChain:
    """Ordered multi-hop chain of evidence nodes forming an end-to-end explanation."""
    nodes: List[EvidenceNode] = field(default_factory=list)
    edges: List[EvidenceEdge] = field(default_factory=list)
    cumulative_score: float = 0.0


@dataclass
class EvidenceGraph:
    """The constructed evidence DAG."""
    nodes: Dict[str, EvidenceNode] = field(default_factory=dict)
    edges: List[EvidenceEdge] = field(default_factory=list)
    adjacency: Dict[str, List[Tuple[str, EvidenceEdge]]] = field(default_factory=dict)


class MultiHopEvidenceDAG:
    """
    Constructs and reasons over evidence DAGs.
    """

    ROLE_HIERARCHY = {
        "cause": 0,
        "premise": 0,
        "mechanism": 1,
        "generic": 1,
        "impact": 2,
        "metric": 3,
        "limitation": 4,
    }

    _METRIC_PATTERNS = [
        r"\d+([.,]\d+)?\s*%",
        r"\b\d+([.,]\d+)?\s*(triệu|tỷ|nghìn|lần|x|ms|s|tokens|mb|gb)\b",
        r"\b(tăng|giảm|đạt|đạt được|tiết kiệm|chiếm)\s+\d+",
        r"o\([n\^0-9]+\)",
    ]

    _LIMITATION_PATTERNS = [
        r"\b(tuy nhiên|mặc dù|nhưng|hạn chế|nhược điểm|thách thức|chi phí ban đầu|rủi ro)\b",
        r"\b(đòi hỏi|yêu cầu cao|chưa giải quyết|khó khăn)\b",
    ]

    _CAUSE_PATTERNS = [
        r"\b(nguyên nhân|do|bởi vì|dẫn đến|gây ra|lý do|vấn đề là)\b",
        r"\b(gặp vấn đề|hạn chế của|thách thức lớn)\b",
    ]

    _MECHANISM_PATTERNS = [
        r"\b(sử dụng|áp dụng|đề xuất|kiến trúc|cơ chế|phương pháp|thuật toán|hoạt động bằng cách)\b",
        r"\b(kết hợp|phân tầng|tích hợp|cập nhật|nén|lưu trữ)\b",
    ]

    _IMPACT_PATTERNS = [
        r"\b(giúp|cho phép|nhằm|tạo điều kiện|cải thiện|nâng cao|tối ưu hóa)\b",
        r"\b(kết quả chứng minh|kết quả thực nghiệm|chứng minh rằng)\b",
    ]

    def split_clean_sentences(self, text: str) -> List[str]:
        """Splits sentences while preserving decimals and abbreviations."""
        masked = re.sub(r"(\d+)[.,](\d+)", r"\1__DEC__\2", text)
        masked = re.sub(r"\b([A-Z])\.", r"\1__DOT__", masked)
        parts = re.split(r"[.!?]+(?:\s+|\s*$)", masked)
        cleaned = []
        for p in parts:
            p_restored = p.replace("__DEC__", ",").replace("__DOT__", ".").strip()
            if len(p_restored) >= 15:
                cleaned.append(p_restored)
        return cleaned

    def classify_role(self, sentence: str, section_title: str) -> str:
        """Classifies an atomic sentence into its cognitive role."""
        s_lower = sentence.lower()
        sec_lower = section_title.lower()

        # Limitation check
        if any(re.search(pat, s_lower) for pat in self._LIMITATION_PATTERNS):
            return "limitation"
        # Metric check
        if any(re.search(pat, s_lower) for pat in self._METRIC_PATTERNS):
            return "metric"
        # Cause / Problem check
        if any(re.search(pat, s_lower) for pat in self._CAUSE_PATTERNS) or "vấn đề" in sec_lower or "problem" in sec_lower:
            return "cause"
        # Impact check
        if any(re.search(pat, s_lower) for pat in self._IMPACT_PATTERNS) or "kết quả" in sec_lower:
            return "impact"
        # Mechanism check
        if any(re.search(pat, s_lower) for pat in self._MECHANISM_PATTERNS) or "phương pháp" in sec_lower or "kiến trúc" in sec_lower:
            return "mechanism"

        return "generic"

    def extract_entities(self, text: str) -> Set[str]:
        """Extracts acronyms, capitalized multi-word phrases, and domain keywords."""
        entities = set()
        # Acronyms (e.g. SA-CMS, RAG, LLM, O(N^2))
        for match in re.finditer(r"\b[A-Z0-9]{2,}(?:-[A-Z0-9]+)*\b", text):
            entities.add(match.group(0).lower())
        # Numbers & metrics
        for match in re.finditer(r"\d+([.,]\d+)?%?", text):
            entities.add(match.group(0).lower())
        # Words of length >= 4
        words = re.findall(r"\b[a-zA-Zà-ỹÀ-Ỹ0-9]{4,}\b", text.lower())
        for w in words:
            if w not in {"trong", "những", "chúng", "được", "nghiên", "nhiều", "phần"}:
                entities.add(w)
        return entities

    def build_dag(self, passages: List[Dict[str, Any]], query: str = "") -> EvidenceGraph:
        """Constructs an evidence graph with directed edges connecting complementary roles."""
        graph = EvidenceGraph()
        q_words = set(re.findall(r"\w+", query.lower())) if query else set()

        node_list: List[EvidenceNode] = []
        for p_idx, passage in enumerate(passages):
            raw_text = passage.get("text", "")
            sec_title = passage.get("section_title", f"Phần {p_idx + 1}")
            cit_label = passage.get("citation_label", f"[{p_idx + 1}]")
            p_id = passage.get("passage_id", f"p_{p_idx}")

            sentences = self.split_clean_sentences(raw_text)
            for s_idx, sent in enumerate(sentences):
                node_id = f"{p_id}::s{s_idx}"
                role = self.classify_role(sent, sec_title)
                entities = self.extract_entities(sent)
                
                # Compute query relevance score
                s_words = set(re.findall(r"\w+", sent.lower()))
                relevance = len(q_words & s_words) / max(1, len(q_words)) if q_words else 0.5
                role_bonus = 0.3 if role in ("metric", "mechanism", "impact") else 0.1
                score = round(relevance * 2.0 + role_bonus, 3)

                node = EvidenceNode(
                    node_id=node_id,
                    text=sent,
                    section_title=sec_title,
                    citation_label=cit_label,
                    role=role,
                    score=score,
                    entities=entities,
                    passage_id=p_id,
                )
                graph.nodes[node_id] = node
                node_list.append(node)
                graph.adjacency[node_id] = []

        # Create directed edges (u -> v) adhering to the DAG role hierarchy:
        # cause/premise -> mechanism -> impact/metric -> limitation
        for i, u in enumerate(node_list):
            u_level = self.ROLE_HIERARCHY.get(u.role, 1)
            for j, v in enumerate(node_list):
                if u.node_id == v.node_id:
                    continue
                v_level = self.ROLE_HIERARCHY.get(v.role, 1)

                # Strictly forward in cognitive hierarchy OR same hierarchy but sequential passage
                is_forward = (v_level > u_level) or (v_level == u_level and i < j and u.passage_id != v.passage_id)
                if not is_forward:
                    continue

                # Edge type determination
                if v.role == "metric":
                    edge_type = "quantifies"
                elif v.role == "limitation":
                    edge_type = "constrains"
                elif u.role in ("cause", "premise") and v.role in ("mechanism", "impact"):
                    edge_type = "causes_or_leads_to"
                else:
                    edge_type = "elaborates"

                # Edge weight based on entity overlap + compatibility
                overlap = len(u.entities & v.entities)
                weight = 1.0 + (overlap * 0.5)

                edge = EvidenceEdge(
                    source_id=u.node_id,
                    target_id=v.node_id,
                    edge_type=edge_type,
                    weight=weight,
                )
                graph.edges.append(edge)
                graph.adjacency[u.node_id].append((v.node_id, edge))

        return graph

    def find_optimal_evidence_chain(self, graph: EvidenceGraph, max_hops: int = 3) -> EvidenceChain:
        """Finds the path in the DAG with highest cumulative informative score."""
        if not graph.nodes:
            return EvidenceChain()

        if len(graph.nodes) == 1:
            only_node = next(iter(graph.nodes.values()))
            return EvidenceChain(nodes=[only_node], cumulative_score=only_node.score)

        # Roots: prefer originating nodes (cause, premise, mechanism) so reasoning flows forward
        originating_candidates = [
            n for n in graph.nodes.values()
            if n.role in ("cause", "premise", "mechanism", "generic")
        ]
        if not originating_candidates:
            originating_candidates = list(graph.nodes.values())

        originating_candidates.sort(key=lambda n: n.score, reverse=True)

        best_chain = EvidenceChain()
        best_score = -1.0

        def dfs(current_id: str, current_nodes: List[EvidenceNode], current_edges: List[EvidenceEdge], current_score: float, hops_left: int):
            nonlocal best_chain, best_score
            if current_score > best_score and len(current_nodes) >= 1:
                # Bonus if chain spans diverse roles, especially bridging mechanism/cause to metric/impact
                roles = {n.role for n in current_nodes}
                has_bridge = any(r in ("cause", "mechanism") for r in roles) and any(r in ("metric", "impact") for r in roles)
                bridge_bonus = 3.0 if has_bridge else 0.0
                diversity_bonus = len(roles) * 1.5 + bridge_bonus
                total = current_score + diversity_bonus
                if total > best_score:
                    best_score = total
                    best_chain = EvidenceChain(
                        nodes=list(current_nodes),
                        edges=list(current_edges),
                        cumulative_score=round(total, 3),
                    )

            if hops_left <= 0:
                return

            for next_id, edge in graph.adjacency.get(current_id, []):
                next_node = graph.nodes[next_id]
                if next_node in current_nodes:
                    continue  # prevent cycle
                dfs(
                    next_id,
                    current_nodes + [next_node],
                    current_edges + [edge],
                    current_score + next_node.score + edge.weight,
                    hops_left - 1,
                )

        # Search starting from top root candidates
        for start_node in originating_candidates[:4]:
            dfs(start_node.node_id, [start_node], [], start_node.score, max_hops - 1)

        if not best_chain.nodes:
            # Fallback to single top node
            all_sorted = sorted(graph.nodes.values(), key=lambda n: n.score, reverse=True)
            top_node = all_sorted[0]
            best_chain = EvidenceChain(nodes=[top_node], cumulative_score=top_node.score)

        return best_chain

    def compose_chain_narrative(self, chain: EvidenceChain) -> str:
        """Composes a unified multi-hop narrative with natural Vietnamese discourse markers."""
        if not chain or not chain.nodes:
            return ""

        if len(chain.nodes) == 1:
            node = chain.nodes[0]
            return f"{node.text} {node.citation_label}"

        parts = []
        for idx, node in enumerate(chain.nodes):
            text = node.text.rstrip(".")
            cit = node.citation_label

            if idx == 0:
                if node.role in ("cause", "premise"):
                    parts.append(f"Về bối cảnh và nguyên nhân, {text.lower() if not text.startswith(('C', 'K', 'N')) else text} {cit}.")
                elif node.role == "mechanism":
                    parts.append(f"Về cơ chế kỹ thuật, {text.lower() if not text.startswith(('S', 'K', 'N')) else text} {cit}.")
                else:
                    parts.append(f"{text} {cit}.")
            elif node.role == "mechanism":
                parts.append(f"Để hiện thực hóa, hệ thống sử dụng {text.lower() if not text.startswith(('S', 'H')) else text} {cit}.")
            elif node.role == "impact":
                parts.append(f"Cơ chế này trực tiếp dẫn đến: {text} {cit}.")
            elif node.role == "metric":
                parts.append(f"Đáng chú ý, kết quả thực nghiệm ghi nhận {text.lower() if not text.startswith(('K', 'S')) else text} {cit}.")
            elif node.role == "limitation":
                parts.append(f"Tuy nhiên, tài liệu cũng lưu ý rằng {text.lower() if not text.startswith(('T', 'M')) else text} {cit}.")
            else:
                parts.append(f"Đồng thời, {text.lower()} {cit}.")

        return " ".join(parts)
