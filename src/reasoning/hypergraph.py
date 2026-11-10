"""
Concept Hypergraph & Multi-Hop Reasoner for SA-CMS.
Bridges isolated passages and distinct documents into a unified semantic graph.
Allows multi-hop deduction across document boundaries (e.g., Fact 1 in Doc A -> Fact 2 in Doc B).
100% Strict Offline, Zero External APIs, Deterministic BFS Pathfinding.
"""

import re
import time
import unicodedata
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import deque


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for canonical node indexing."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


class RelationType(str, Enum):
    """Semantic relation types connecting concepts across documents."""
    CAUSAL = "causal"           # X dẫn đến / gây ra / nhờ đó Y
    SOLUTION = "solution"       # X giải quyết / khắc phục / tối ưu hóa Y
    PART_OF = "part_of"         # X bao gồm / là một phần của Y
    COMPARISON = "comparison"   # X vượt trội hơn / thay thế cho Y
    ATTRIBUTE = "attribute"     # X đạt mức / có giá trị là Y
    OPPOSITION = "opposition"   # X trái ngược / mâu thuẫn với Y


@dataclass
class ConceptNode:
    """A unique entity or conceptual subject in the hypergraph."""
    name: str
    canonical_id: str
    node_type: str = "entity"  # entity, method, metric, problem
    occurrences: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "canonical_id": self.canonical_id,
            "node_type": self.node_type,
            "occurrence_count": len(self.occurrences),
        }


@dataclass
class RelationEdge:
    """A directed semantic edge between two concepts with provenance."""
    source_id: str
    target_id: str
    relation_type: RelationType
    evidence_text: str
    document_id: str
    section_title: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation_type.value,
            "evidence": self.evidence_text,
            "document_id": self.document_id,
            "section_title": self.section_title,
            "confidence": self.confidence,
        }


@dataclass
class MultiHopChain:
    """A multi-hop deductive explanation bridging concepts across documents."""
    query: str
    start_concept: str
    end_concept: str
    hop_count: int
    edges: List[RelationEdge]
    reasoning_steps: List[str]
    synthesis: str
    documents_involved: List[str]
    citations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "start_concept": self.start_concept,
            "end_concept": self.end_concept,
            "hop_count": self.hop_count,
            "reasoning_steps": self.reasoning_steps,
            "synthesis": self.synthesis,
            "documents_involved": self.documents_involved,
            "citations": self.citations,
        }


class ConceptHypergraph:
    """
    Constructs and queries an in-memory knowledge hypergraph from document text.
    Supports multi-hop breadth-first search (BFS) reasoning across documents.
    """

    def __init__(self):
        self.nodes: Dict[str, ConceptNode] = {}
        self.adj_list: Dict[str, List[RelationEdge]] = {}
        self.edges: List[RelationEdge] = []

    def _canonicalize(self, text: str) -> str:
        s = strip_accents(text.lower().strip())
        s = re.sub(r"[^\w\s]", " ", s)
        return re.sub(r"\s+", "_", s).strip("_")

    def add_node(self, name: str, node_type: str = "entity", occurrence: Optional[Dict[str, Any]] = None) -> ConceptNode:
        cid = self._canonicalize(name)
        if not cid:
            cid = "unknown"
        if cid not in self.nodes:
            self.nodes[cid] = ConceptNode(name=name.strip(), canonical_id=cid, node_type=node_type)
            self.adj_list[cid] = []
        if occurrence:
            self.nodes[cid].occurrences.append(occurrence)
        return self.nodes[cid]

    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: RelationType,
        evidence_text: str,
        document_id: str,
        section_title: str = "",
        confidence: float = 1.0,
    ) -> Optional[RelationEdge]:
        s_node = self.add_node(source)
        t_node = self.add_node(target)
        if s_node.canonical_id == t_node.canonical_id:
            return None

        edge = RelationEdge(
            source_id=s_node.canonical_id,
            target_id=t_node.canonical_id,
            relation_type=relation_type,
            evidence_text=evidence_text.strip(),
            document_id=document_id,
            section_title=section_title,
            confidence=confidence,
        )
        self.edges.append(edge)
        self.adj_list[s_node.canonical_id].append(edge)
        return edge

    def extract_from_sentences(
        self,
        sentences: List[str],
        document_id: str,
        section_title: str = "",
    ) -> int:
        """
        Parses sentences and extracts relational edges using syntactic patterns.
        """
        extracted_count = 0

        # Regex pattern templates: (Subject) (Verb/Predicate) (Object)
        patterns = [
            # 1. Solution: X giải quyết / khắc phục / tối ưu hóa Y
            (
                re.compile(r"([\w\s\-\_]{2,35})\s+(?:giải quyết|khắc phục|tối ưu hóa|giảm thiểu|xử lý)\s+([\w\s\-\_]{2,40})", re.IGNORECASE),
                RelationType.SOLUTION,
            ),
            # 2. Causal: X dẫn đến / gây ra / khiến cho / nhờ đó Y
            (
                re.compile(r"([\w\s\-\_]{2,35})\s+(?:dẫn đến|gây ra|làm cho|nhờ đó|khiến cho)\s+([\w\s\-\_]{2,40})", re.IGNORECASE),
                RelationType.CAUSAL,
            ),
            # 3. Part-of: X bao gồm / gồm có / chứa Y
            (
                re.compile(r"([\w\s\-\_]{2,35})\s+(?:bao gồm|gồm có|chứa các thành phần|chia thành)\s+([\w\s\-\_]{2,40})", re.IGNORECASE),
                RelationType.PART_OF,
            ),
            # 4. Comparison: X vượt trội hơn / tốt hơn / thay thế cho Y
            (
                re.compile(r"([\w\s\-\_]{2,35})\s+(?:vượt trội hơn|tốt hơn|thay thế cho|tiết kiệm hơn)\s+([\w\s\-\_]{2,40})", re.IGNORECASE),
                RelationType.COMPARISON,
            ),
            # 5. Attribute: X đạt mức / chiếm tỷ lệ Y
            (
                re.compile(r"([\w\s\-\_]{2,35})\s+(?:đạt mức|chiếm tỷ lệ|đo được là|đạt)\s+([\w\s\-\_0-9.,%]{2,35})", re.IGNORECASE),
                RelationType.ATTRIBUTE,
            ),
        ]

        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 15:
                continue

            for pattern, rel_type in patterns:
                for match in pattern.finditer(s_clean):
                    subj = match.group(1).strip(" ,.;:-")
                    obj = match.group(2).strip(" ,.;:-")

                    # Filter out trivial stop subjects
                    subj_norm = strip_accents(subj.lower())
                    if any(subj_norm == w for w in ["dieu nay", "chung", "no", "he thong", "viec"]):
                        continue
                    if len(subj.split()) > 8 or len(obj.split()) > 10:
                        continue

                    edge = self.add_edge(
                        source=subj,
                        target=obj,
                        relation_type=rel_type,
                        evidence_text=s_clean,
                        document_id=document_id,
                        section_title=section_title,
                    )
                    if edge:
                        extracted_count += 1

        return extracted_count

    def find_path(
        self,
        start_concept: str,
        target_concept: str,
        max_hops: int = 4,
    ) -> Optional[List[RelationEdge]]:
        """
        Executes Breadth-First Search (BFS) to find the shortest reasoning path.
        """
        start_id = self._canonicalize(start_concept)
        target_id = self._canonicalize(target_concept)

        if start_id not in self.nodes or target_id not in self.nodes:
            # Fuzzy match to closest canonical node
            start_candidates = [nid for nid in self.nodes if start_id in nid or nid in start_id]
            target_candidates = [nid for nid in self.nodes if target_id in nid or nid in target_id]
            if not start_candidates or not target_candidates:
                return None
            start_id = start_candidates[0]
            target_id = target_candidates[0]

        if start_id == target_id:
            return []

        queue = deque([(start_id, [])])
        visited: Set[str] = {start_id}

        while queue:
            curr_node, path = queue.popleft()
            if len(path) >= max_hops:
                continue

            for edge in self.adj_list.get(curr_node, []):
                next_node = edge.target_id
                if next_node == target_id:
                    return path + [edge]

                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path + [edge]))

        return None

    def synthesize_multi_hop_deduction(
        self,
        query: str,
        start_concept: str,
        target_concept: str,
        max_hops: int = 4,
    ) -> Optional[MultiHopChain]:
        """
        Finds reasoning path and synthesizes multi-hop logical explanation with provenance.
        """
        path = self.find_path(start_concept, target_concept, max_hops=max_hops)
        if path is None or len(path) == 0:
            return None

        reasoning_steps = []
        docs_involved = set()
        citations = []

        for idx, edge in enumerate(path, start=1):
            s_name = self.nodes.get(edge.source_id, ConceptNode(edge.source_id, edge.source_id)).name
            t_name = self.nodes.get(edge.target_id, ConceptNode(edge.target_id, edge.target_id)).name

            rel_text = {
                RelationType.CAUSAL: "dẫn đến / làm cho",
                RelationType.SOLUTION: "giải quyết / tối ưu hóa",
                RelationType.PART_OF: "bao gồm thành phần",
                RelationType.COMPARISON: "vượt trội / thay thế",
                RelationType.ATTRIBUTE: "đạt chỉ số",
            }.get(edge.relation_type, "liên kết với")

            step_desc = f"Bước {idx}: **{s_name}** {rel_text} **{t_name}** [{edge.document_id}]"
            reasoning_steps.append(step_desc)
            docs_involved.add(edge.document_id)
            citations.append({
                "hop": idx,
                "document_id": edge.document_id,
                "section_title": edge.section_title,
                "evidence": edge.evidence_text,
            })

        # Build natural synthesis statement
        chain_summary = " -> ".join([
            self.nodes.get(edge.source_id, ConceptNode(edge.source_id, edge.source_id)).name
            for edge in path
        ] + [self.nodes.get(path[-1].target_id, ConceptNode(path[-1].target_id, path[-1].target_id)).name])

        synthesis = (
            f"Qua đối chiếu liên kết giữa các tài liệu ({', '.join(docs_involved)}), "
            f"chuỗi suy luận được xác lập: {chain_summary}.\n"
            + "\n".join([f"- {s}" for s in reasoning_steps])
        )

        return MultiHopChain(
            query=query,
            start_concept=start_concept,
            end_concept=target_concept,
            hop_count=len(path),
            edges=path,
            reasoning_steps=reasoning_steps,
            synthesis=synthesis,
            documents_involved=sorted(list(docs_involved)),
            citations=citations,
        )

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "relations_by_type": {
                t.value: sum(1 for e in self.edges if e.relation_type == t)
                for t in RelationType
            },
        }
