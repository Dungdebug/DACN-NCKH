"""
QAPairExtractor — Rule-based grounded QA extraction from document structures.
Implements Proposal Section 6.4: "Tự học bằng câu hỏi tự sinh".
100% Strict Offline, Zero External API, Pure Python Regex & Structural Parsing.
"""

import re
import time
import unicodedata
from typing import List, Dict, Any, Optional, Set, Tuple

from src.self_study.types import QARuleType, SelfStudyQAPair, SelfStudyReport
from src.document_structure.types import DocumentStructure, StructuralSpan, BoundaryType
from src.document_structure.parser import DocumentStructureParser


def normalize_text_key(text: str) -> str:
    """Normalizes text for deduplication: lowercase, strip accents, strip punctuation."""
    norm = unicodedata.normalize("NFD", text.lower())
    clean = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    clean = re.sub(r"[^\w\s]", "", clean)
    return " ".join(clean.split())


class QAPairExtractor:
    """
    Extracts high-precision, document-grounded question-answer pairs using
    syntactic boundary rules on hierarchical document structures.
    
    Guarantees:
    1. Zero External API / Zero Cloud AI.
    2. Exact Substring Grounding: Answer is guaranteed to be in source text.
    3. Hierarchy Mapping: Aligned to SA-CMS levels (1=Paragraph, 2=Section, 3=Document).
    """

    # Common Vietnamese abbreviations to protect during sentence splitting
    ABBREVIATIONS = {
        "ts", "pgs", "gs", "ths", "ks", "bs", "tp", "tt", "tx", "hđ", "ubnd",
        "fig", "eq", "sec", "ref", "dr", "prof", "vol", "no", "pp", "al", "etc",
    }

    # Stopwords for filtering vacuous questions
    STOPWORDS = {
        "là", "của", "và", "các", "những", "cho", "với", "trong", "được", "có",
        "này", "đó", "khi", "thì", "mà", "từ", "ở", "về", "như", "một", "nhiều",
    }

    def __init__(self, min_question_words: int = 3, max_answer_words: int = 80):
        self.min_question_words = min_question_words
        self.max_answer_words = max_answer_words
        self.parser = DocumentStructureParser()

    def split_sentences(self, text: str) -> List[str]:
        """Splits text into sentences while protecting abbreviations, decimals, and lines."""
        if not text:
            return []
        
        # Protect decimal numbers (e.g., 0.154 -> 0<DECIMAL>154)
        protected = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL>\2', text)
        
        # Split on sentence punctuation or distinct linebreaks
        raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', protected)
        
        sentences = []
        for s in raw_sentences:
            cleaned = s.replace("<DECIMAL>", ".").strip()
            # Skip empty or trivially short lines (< 12 chars)
            if len(cleaned) >= 12:
                sentences.append(cleaned)
        return sentences

    def extract_from_text(
        self,
        text: str,
        document_id: str = "DOC_INLINE",
        max_pairs_per_section: int = 5,
        max_pairs: Optional[int] = None,
    ) -> SelfStudyReport:
        """Parses text into hierarchical structure and extracts QA pairs across levels."""
        start_time = time.time()
        doc_structure = self.parser.parse_text(text)
        report = self.extract_from_structure(
            doc_structure=doc_structure,
            document_id=document_id,
            max_pairs_per_section=max_pairs_per_section,
            start_time=start_time,
        )
        if max_pairs is not None and len(report.pairs) > max_pairs:
            report.pairs = report.pairs[:max_pairs]
        return report

    def extract_from_structure(
        self,
        doc_structure: DocumentStructure,
        document_id: str = "DOC_001",
        max_pairs_per_section: int = 5,
        start_time: Optional[float] = None,
    ) -> SelfStudyReport:
        """
        Extracts QA pairs from pre-parsed DocumentStructure:
        - Level 1: Paragraph-grounded pairs (factual details, metrics)
        - Level 2: Section-grounded pairs (mechanisms, enumerations, summaries)
        - Level 3: Document-grounded pairs (high-level definitions, overarching purpose)
        """
        if start_time is None:
            start_time = time.time()

        pairs: List[SelfStudyQAPair] = []
        seen_question_keys: Set[str] = set()
        pairs_by_rule: Dict[str, int] = {r.value: 0 for r in QARuleType}

        # Level 1: Process Paragraphs
        for p_span in doc_structure.paragraphs:
            p_text = p_span.text.strip()
            if not p_text:
                continue
            p_pairs = self._extract_from_span(
                text=p_text,
                span_offset=p_span.start_char,
                document_id=document_id,
                level=1,
                section_title=p_span.title,
                seen_keys=seen_question_keys,
            )
            for p in p_pairs:
                pairs.append(p)
                pairs_by_rule[p.rule_type.value] += 1

        # Level 2: Process Sections (Multi-paragraph integration)
        for s_span in doc_structure.sections:
            s_text = s_span.text.strip()
            if not s_text:
                continue
            s_pairs = self._extract_from_span(
                text=s_text,
                span_offset=s_span.start_char,
                document_id=document_id,
                level=2,
                section_title=s_span.title,
                seen_keys=seen_question_keys,
                limit=max_pairs_per_section,
            )
            for p in s_pairs:
                pairs.append(p)
                pairs_by_rule[p.rule_type.value] += 1

        # Level 3: Document-wide synthesis (first section / abstract / overall purpose)
        doc_text = doc_structure.text.strip()
        if doc_text and doc_structure.sections:
            lead_section = doc_structure.sections[0]
            lead_pairs = self._extract_from_span(
                text=lead_section.text.strip(),
                span_offset=lead_section.start_char,
                document_id=document_id,
                level=3,
                section_title="Toàn văn tài liệu",
                seen_keys=seen_question_keys,
                limit=3,
            )
            for p in lead_pairs:
                pairs.append(p)
                pairs_by_rule[p.rule_type.value] += 1

        elapsed_ms = (time.time() - start_time) * 1000.0
        return SelfStudyReport(
            document_id=document_id,
            total_pairs=len(pairs),
            pairs=pairs,
            pairs_by_rule=pairs_by_rule,
            extraction_time_ms=round(elapsed_ms, 2),
            metadata={"total_tokens": doc_structure.total_tokens},
        )

    def _extract_from_span(
        self,
        text: str,
        span_offset: int,
        document_id: str,
        level: int,
        section_title: str,
        seen_keys: Set[str],
        limit: int = 10,
    ) -> List[SelfStudyQAPair]:
        """Applies 6 syntactic rule matchers to sentences in a structural span."""
        span_pairs: List[SelfStudyQAPair] = []
        sentences = self.split_sentences(text)

        for sent in sentences:
            if len(span_pairs) >= limit:
                break

            # 1. Limitation Rule (Checked first to avoid definition collision):
            # "Hạn chế của X là Y", "Điểm yếu của X là Y"
            p_lim = self._match_limitation(sent, text, span_offset, document_id, level, section_title)
            if p_lim and self._is_valid_and_novel(p_lim, seen_keys):
                span_pairs.append(p_lim)
                continue

            # 2. Mechanism Rule: "X hoạt động nhờ/qua Y"
            p_mec = self._match_mechanism(sent, text, span_offset, document_id, level, section_title)
            if p_mec and self._is_valid_and_novel(p_mec, seen_keys):
                span_pairs.append(p_mec)
                continue

            # 3. Metric Rule: "F1 đạt 0.154" / "tiêu thụ 1.2 GB" / "32 tuần"
            p_met = self._match_metric(sent, text, span_offset, document_id, level, section_title)
            if p_met and self._is_valid_and_novel(p_met, seen_keys):
                span_pairs.append(p_met)
                continue

            # 4. Enumeration Rule: "bao gồm: A, B, C", "gồm 3 mức"
            p_enum = self._match_enumeration(sent, text, span_offset, document_id, level, section_title)
            if p_enum and self._is_valid_and_novel(p_enum, seen_keys):
                span_pairs.append(p_enum)
                continue

            # 5. Comparison Rule: "X tốt hơn Y" / "P1 hơn B5" / "SA-CMS quên ít hơn"
            p_comp = self._match_comparison(sent, text, span_offset, document_id, level, section_title)
            if p_comp and self._is_valid_and_novel(p_comp, seen_keys):
                span_pairs.append(p_comp)
                continue

            # 6. Definition Rule: "X là Y" / "X được định nghĩa là Y" / "X: Y"
            p_def = self._match_definition(sent, text, span_offset, document_id, level, section_title)
            if p_def and self._is_valid_and_novel(p_def, seen_keys):
                span_pairs.append(p_def)
                continue

        return span_pairs

    # =========================================================================
    # RULE MATCHERS (Deterministic, Grounded Substring Regex)
    # =========================================================================

    def _match_limitation(
        self, sent: str, text: str, span_offset: int, doc_id: str, level: int, title: str
    ) -> Optional[SelfStudyQAPair]:
        """Matches: 'Hạn chế/Nhược điểm/Thách thức/Điểm yếu của <Subject> là <Predicate>'."""
        pattern = r"^(?:[-*•]\s*)?(?:Hạn chế|Nhược điểm|Điểm yếu|Thách thức|Giới hạn)\s+(?:của\s+(?P<subject>[A-ZÀ-Ỹ0-9][\w\s\-\.\(\)]{2,40})\s+)?(?:là|nằm ở việc|gồm)\s+(?P<predicate>[^.,;\n]{8,220})"
        match = re.search(pattern, sent, re.IGNORECASE)
        if not match:
            return None

        subj = match.group("subject")
        subj_str = subj.strip() if subj else "phương pháp"
        pred = match.group("predicate").strip()

        question = f"Hạn chế của {subj_str} là gì?"
        answer = pred
        return self._build_pair(sent, answer, question, text, span_offset, doc_id, level, title, QARuleType.LIMITATION, 0.94)

    def _match_mechanism(
        self, sent: str, text: str, span_offset: int, doc_id: str, level: int, title: str
    ) -> Optional[SelfStudyQAPair]:
        """Matches: '<Subject> hoạt động nhờ/bằng/qua/theo <Mechanism>'."""
        pattern = r"(?P<subject>[A-ZÀ-Ỹ0-9][\w\s\-\.\(\)]{2,45})\s+(?:hoạt động nhờ|hoạt động bằng|hoạt động qua|vận hành dựa trên|thực hiện thông qua)\s+(?P<predicate>[^.,;\n]{8,220})"
        match = re.search(pattern, sent, re.IGNORECASE)
        if not match:
            return None

        subj = match.group("subject").strip(" :-*•")
        pred = match.group("predicate").strip()

        question = f"{subj} hoạt động như thế nào?"
        answer = pred
        return self._build_pair(sent, answer, question, text, span_offset, doc_id, level, title, QARuleType.MECHANISM, 0.92)

    def _match_metric(
        self, sent: str, text: str, span_offset: int, doc_id: str, level: int, title: str
    ) -> Optional[SelfStudyQAPair]:
        """Matches: '<Metric_Name> đạt/chiếm/ở mức/tiêu thụ/khoảng/là <Value>'."""
        pattern = r"(?P<metric>[A-ZÀ-Ỹ0-9][\w\s\-\.\(\)]{2,40})\s+(?:đạt|chiếm|ở mức|giảm|tăng|tiêu thụ|được đo là|khoảng|là)\s+(?P<value>\d+(?:[\.,\-\u2013–]\d+)?\s*(?:%|ms|tokens?|GB|MB|params?|s|giây|phút|tuần|hồ sơ|câu hỏi|tài liệu|tỷ|triệu\s+token|F1|EM))"
        match = re.search(pattern, sent, re.IGNORECASE)
        if not match:
            return None

        metric = match.group("metric").strip(" :-*•")
        val = match.group("value").strip()

        # Reject common invalid words as metric name
        if metric.lower() in {"đây", "đó", "nó", "điều này", "kết quả", "bài báo", "thời gian"} or "bao nhiêu" in metric.lower():
            return None

        question = f"{metric} là bao nhiêu?"
        answer = f"{metric} đạt {val}" if f"{metric} đạt {val}" in sent else val
        if answer not in sent:
            answer = val

        return self._build_pair(sent, answer, question, text, span_offset, doc_id, level, title, QARuleType.METRIC, 0.96)

    def _match_enumeration(
        self, sent: str, text: str, span_offset: int, doc_id: str, level: int, title: str
    ) -> Optional[SelfStudyQAPair]:
        """Matches: '<Subject> bao gồm/gồm có/chia thành: <Items>'."""
        pattern = r"(?P<subject>[A-ZÀ-Ỹ0-9][\w\s\-\.\(\)]{2,45})(?:,\s*|\s+)(?:bao gồm|gồm có|gồm|chia thành|cấu thành từ)\s*:?\s*(?P<items>[^.\n]{10,250})"
        match = re.search(pattern, sent, re.IGNORECASE)
        if not match:
            return None

        subj = match.group("subject").strip(" :-*•")
        items = match.group("items").strip()

        question = f"{subj} bao gồm những thành phần nào?"
        answer = items
        return self._build_pair(sent, answer, question, text, span_offset, doc_id, level, title, QARuleType.ENUMERATION, 0.90)

    def _match_comparison(
        self, sent: str, text: str, span_offset: int, doc_id: str, level: int, title: str
    ) -> Optional[SelfStudyQAPair]:
        """Matches: '<Subject> tốt hơn/cao hơn/vượt trội hơn <Target>'."""
        pattern = r"(?P<subject>[A-ZÀ-Ỹ0-9][\w\s\-\.\(\)]{2,40})\s+(?P<rel>tốt hơn|cao hơn|vượt trội hơn|vượt trội so với|quên ít hơn)\s+(?P<target>[A-ZÀ-Ỹ0-9][\w\s\-\.\(\)]{2,40})"
        match = re.search(pattern, sent, re.IGNORECASE)
        if not match:
            return None

        subj = match.group("subject").strip(" :-*•")
        rel = match.group("rel").strip()
        target = match.group("target").strip()

        question = f"{subj} so với {target} như thế nào?"
        answer = f"{subj} {rel} {target}"
        if answer not in sent:
            answer = f"{rel} {target}"

        return self._build_pair(sent, answer, question, text, span_offset, doc_id, level, title, QARuleType.COMPARISON, 0.85)

    def _match_definition(
        self, sent: str, text: str, span_offset: int, doc_id: str, level: int, title: str
    ) -> Optional[SelfStudyQAPair]:
        """Matches: '<Subject> là / chính là / được định nghĩa là <Predicate>' OR '<Subject>: <Predicate>'."""
        # 1. Standard "X là Y" / "X được định nghĩa là Y"
        pat1 = r"^(?:[-*•]\s*)?(?P<subject>[A-ZÀ-Ỹ0-9][\w\s\-\.\(\)]{2,45})\s+(?:là|được định nghĩa là|chính là)\s+(?P<predicate>[^.,;\n]{8,220})"
        m1 = re.search(pat1, sent, re.IGNORECASE)
        if m1:
            subj = m1.group("subject").strip(" :-*•")
            pred = m1.group("predicate").strip()
            # Filter non-informative subjects
            if subj.lower() not in {"đây", "đó", "nó", "điều này", "kết quả", "bài toán"} and "bao nhiêu" not in subj.lower():
                question = f"{subj} là gì?"
                answer = pred
                return self._build_pair(sent, answer, question, text, span_offset, doc_id, level, title, QARuleType.DEFINITION, 0.95)

        # 2. Academic Colon Definition: "X: Y"
        pat2 = r"^(?:[-*•]\s*)?(?P<subject>[A-ZÀ-Ỹ0-9][\w\s\-\.\(\)]{2,35}):\s+(?P<predicate>[^.\n]{8,200})"
        m2 = re.search(pat2, sent)
        if m2:
            subj = m2.group("subject").strip(" :-*•")
            pred = m2.group("predicate").strip()
            if subj.lower() not in {"chú thích", "ghi chú", "lưu ý", "nguồn", "thời gian", "mục", "phần", "chương"} and "bài báo" not in subj.lower():
                question = f"{subj} là gì?"
                answer = pred
                return self._build_pair(sent, answer, question, text, span_offset, doc_id, level, title, QARuleType.DEFINITION, 0.91)

        return None

    def _build_pair(
        self,
        sentence: str,
        answer: str,
        question: str,
        full_text: str,
        span_offset: int,
        doc_id: str,
        level: int,
        title: str,
        rule_type: QARuleType,
        confidence: float,
    ) -> Optional[SelfStudyQAPair]:
        """Builds SelfStudyQAPair while rigorously enforcing character span offsets."""
        ans_clean = answer.strip()
        if not ans_clean or ans_clean not in sentence:
            return None

        # Locate exact position in full_text
        ans_idx = full_text.find(ans_clean)
        if ans_idx == -1:
            ans_idx = 0
        char_start = span_offset + ans_idx
        char_end = char_start + len(ans_clean)

        try:
            return SelfStudyQAPair(
                question=question,
                answer=ans_clean,
                span_start=char_start,
                span_end=char_end,
                source_text=sentence,
                document_id=doc_id,
                level=level,
                rule_type=rule_type,
                confidence=confidence,
                section_title=title,
                metadata={"matched_sentence": sentence},
            )
        except ValueError:
            return None

    def _is_valid_and_novel(self, pair: SelfStudyQAPair, seen_keys: Set[str]) -> bool:
        """Enforces quality constraints: length, words count, deduplication."""
        q_words = pair.question.split()
        if len(q_words) < self.min_question_words:
            return False

        a_words = pair.answer.split()
        if len(a_words) < 2 or len(a_words) > self.max_answer_words:
            return False

        q_key = normalize_text_key(pair.question)
        if q_key in seen_keys:
            return False

        seen_keys.add(q_key)
        return True
