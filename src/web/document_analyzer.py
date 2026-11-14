"""
Phase 5.9 Extension — Document Intelligence & Auto Question Generator.
Extracts structured executive summaries, key takeaways, and generates
diverse, relevant, document-grounded questions from hierarchical SA-CMS structures.
100% Local Offline, Zero External API, Zero Network Egress.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

from src.utils.vietnamese_nlp import VietnameseNLP
from src.web.token_optimizer import clean_sentence_abstractive


def strip_accents(s: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive matching."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


class DocumentAnalyzer:
    """
    Analyzes document structures to synthesize key points and generate
    insightful, grounded questions tailored to the document's content.
    """

    STOPWORDS_VI = {
        "và", "của", "các", "những", "trong", "để", "với", "cho", "tại", "là",
        "có", "được", "khi", "này", "đó", "theo", "về", "như", "một", "nhiều",
        "thì", "mà", "từ", "ở", "ra", "đã", "sẽ", "đang", "bởi", "do", "bằng",
    }

    @staticmethod
    def _clean_text(text: str) -> str:
        """Removes citation markers and excess whitespace."""
        clean = re.sub(r"\[?[A-Z0-9_\-]+::P\d+\]?", "", text)
        clean = re.sub(r"_[A-Z0-9_]+::P\d+\]?", "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @classmethod
    def _split_clean_sentences(cls, text: str) -> List[str]:
        """Splits text into meaningful sentences."""
        clean = cls._clean_text(text)
        raw_sents = re.split(r"(?<=[.!?])\s+|\n+", clean)
        sentences = []
        for s in raw_sents:
            s_stripped = s.strip()
            # Ignore numbered headers like "1. ", "1.1 "
            s_stripped = re.sub(r"^\d+(\.\d+)*\.?\s*", "", s_stripped)
            if len(s_stripped) >= 20 and not s_stripped.startswith("#"):
                sentences.append(s_stripped)
        return sentences

    @classmethod
    def extract_key_points_and_summary(
        cls,
        raw_text: str,
        title: str = "Tài liệu",
        max_points: int = 4,
    ) -> Dict[str, Any]:
        """
        Extracts high-level executive summary and 3-4 key takeaways.
        """
        all_sentences = cls._split_clean_sentences(raw_text)
        if not all_sentences:
            return {
                "overview": f"Tài liệu {title} đã được nạp thành công vào hệ thống bộ nhớ tham số SA-CMS.",
                "key_points": [f"Nội dung tài liệu '{title}' bao gồm các phần kiến trúc và dữ kiện kỹ thuật liên quan."],
            }

        key_points = []
        try:
            from src.brain.textrank_salience import select_key_points
            key_points = select_key_points(all_sentences, k=max_points)
        except Exception:
            key_points = []

        if not key_points:
            # Fallback to legacy heuristic scoring + distinct prefix selection
            scored_sentences = []
            for s in all_sentences:
                score = 0.0
                s_lower = s.lower()
                # Informational keywords
                if any(k in s_lower for k in ["mục tiêu", "nghiên cứu", "đề xuất", "hệ thống", "kiến trúc", "mô hình"]):
                    score += 3.0
                if any(k in s_lower for k in ["kết quả", "hiệu năng", "đạt", "đánh giá", "tối ưu"]):
                    score += 2.5
                if any(k in s_lower for k in ["hạn chế", "thách thức", "vấn đề", "khó khăn"]):
                    score += 2.0
                if re.search(r"\d+(\.\d+)?\s*(%|token|ms|gb|mb|triệu|nghìn)", s_lower):
                    score += 2.0
                # Length sanity check (prefer 40 - 200 chars)
                if 40 <= len(s) <= 220:
                    score += 1.0

                scored_sentences.append((score, s))

            scored_sentences.sort(key=lambda x: x[0], reverse=True)

            # Pick top distinct points
            seen = set()
            for _, s in scored_sentences:
                first_few = " ".join(s.split()[:4]).lower()
                if first_few not in seen:
                    seen.add(first_few)
                    key_points.append(s)
                if len(key_points) >= max_points:
                    break

            if not key_points:
                key_points = all_sentences[:max_points]

        lead = key_points[0] if key_points else "Tài liệu kỹ thuật chuyên ngành."
        overview = f"Tài liệu '{title}' tập trung vào các nội dung trọng tâm: {lead}"

        return {
            "overview": overview,
            "key_points": key_points,
        }

    @classmethod
    def build_document_knowledge_blueprint(
        cls,
        raw_text: str,
        title: str = "Tài liệu",
    ) -> str:
        """
        Synthesizes a comprehensive VIP Multi-Tier Document Knowledge Blueprint:
        - L1 TL;DR & L2 Executive Synthesis (via ProgressiveSummarizer)
        - Key structural points & headings
        - Extracted quantitative metrics & benchmarks (% / MB / GB / ms / parameters)
        Uses zero raw '###' or single-asterisk '*italic*' so frontend renders cleanly.
        """
        summary_data = cls.extract_key_points_and_summary(raw_text, title=title, max_points=6)
        overview = summary_data.get("overview", "")
        key_points = summary_data.get("key_points", [])

        l1_tldr = overview
        l2_exec = ""
        try:
            from src.brain.progressive_summarizer import ProgressiveSummarizer
            ps = ProgressiveSummarizer().generate_digest(raw_text, doc_title=title)
            if ps.l1_tldr:
                l1_tldr = ps.l1_tldr
            if ps.l2_executive:
                l2_exec = ps.l2_executive
        except Exception:
            pass

        # Extract section headings if present
        headings = []
        for line in raw_text.splitlines():
            l_str = line.strip()
            if l_str.startswith("#"):
                h_clean = l_str.lstrip("#").strip()
                if h_clean and h_clean not in headings:
                    headings.append(h_clean)

        # Extract quantitative metrics
        metrics_found = []
        for m in re.finditer(
            r"\b(\d+(?:[.,]\d+)?\s*(?:%|(?:MB|GB|KB|ms|s|tokens?|triệu(?:\s+tham\s+số)?|tỷ|tầng|lớp)\b))",
            raw_text,
            re.IGNORECASE,
        ):
            val_str = m.group(1).strip()
            if val_str not in metrics_found:
                metrics_found.append(val_str)

        blocks = [
            f"📖 **BẢN ĐỒ TRI THỨC TÀI LIỆU: {title}**",
            f"🎯 **Tóm lược cốt lõi (TL;DR):** {l1_tldr}",
        ]
        if l2_exec and l2_exec.strip() != l1_tldr.strip():
            blocks.append(f"🧭 **Tổng hợp điều hành:**\n{l2_exec}")

        if headings:
            h_badges = " • ".join(f"`{h}`" for h in headings[:6])
            blocks.append(f"🗂️ **Cấu trúc chương mục:** {h_badges}")

        if key_points:
            pts_str = "\n".join(f"- {pt}" for pt in key_points[:6])
            blocks.append(f"📌 **Các luận điểm & phát hiện trọng tâm:**\n{pts_str}")

        if metrics_found:
            m_badges = ", ".join(f"**{m}**" for m in metrics_found[:8])
            blocks.append(f"📊 **Chỉ số & thông số định lượng nổi bật:** {m_badges}")

        return "\n\n".join(blocks)

    @classmethod
    def generate_suggested_questions(
        cls,
        raw_text: str,
        title: str = "Tài liệu",
        max_questions: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Automatically generates diverse, relevant, document-grounded questions.
        Covers definitions, mechanisms, limitations, quantitative facts, and overviews.
        """
        questions = []
        seen_queries = set()

        def add_q(q_text: str, category: str, sample_ans: str = ""):
            clean_q = q_text.strip()
            if clean_q and clean_q.lower() not in seen_queries:
                seen_queries.add(clean_q.lower())
                questions.append({
                    "question": clean_q,
                    "category": category,
                    "sample_answer_excerpt": sample_ans[:160] if sample_ans else "",
                })

        # Pattern 0: Specific Topics / Modules (e.g. "Topic 1: ...", "Topic 11: ...", "Chủ đề 2:")
        topic_pattern = re.compile(
            r"^\s*(Topic|Chủ đề|Bài|Mục|Section)\s*(\d+)[:\-\.\s]+([^\n\.\?]{3,65})",
            re.IGNORECASE | re.MULTILINE,
        )
        for tm in topic_pattern.finditer(raw_text):
            pfx, num_str, t_name = tm.group(1).title(), tm.group(2), tm.group(3).strip()
            add_q(f"Yêu cầu và nội dung chính của {pfx} {num_str} ({t_name[:45]}) là gì?", "Chủ đề bài học", tm.group(0))
            if len(questions) >= 3:
                break

        # Pattern 1: Section-based Questions (Level 2 SA-CMS structure - highest fidelity)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for line in lines:
            # Matches: "## 1. Quy hoạch Điện VIII", "### 2. Kiến trúc", "1. Đặt vấn đề"
            h_match = re.match(r"^(#{1,4}\s*)?(\d+(\.\d+)*\.?\s*)?([A-ZÀ-Ỹ][^\n:]{4,55})$", line)
            if h_match:
                sec_name = h_match.group(4).strip()
                # Skip generic headers
                if any(g in sec_name.lower() for g in ["tài liệu tham khảo", "mục lục", "references", "phụ lục", "overview"]):
                    continue
                q = f"Thông tin cốt lõi được trình bày tại mục '{sec_name}' là gì?"
                add_q(q, "Tra cứu Chương mục", sec_name)
            if len(questions) >= 3:
                break

        # Pattern 2: Concept & Definitions ("là gì", "được định nghĩa")
        def_pattern = re.compile(
            r"([A-ZÀ-Ỹa-zà-ỹ0-9_\-\s]{3,35})\s+(là\s+gì|là\s+một|được\s+(xác định|hiểu|định nghĩa)\s+là|đóng\s+vai\s+trò\s+là)\s+([^.!?]+)",
            re.IGNORECASE,
        )
        for m in def_pattern.finditer(raw_text):
            subject = m.group(1).strip()
            subject = re.sub(r"^(ở|tại|theo|trong|với|khi)\s+", "", subject, flags=re.IGNORECASE).strip()
            cleaned_subject = VietnameseNLP.clean_extracted_term(subject)
            if 1 <= len(cleaned_subject.split()) <= 6 and not cleaned_subject.startswith("#"):
                q = f"{cleaned_subject} là gì và có vai trò như thế nào?"
                add_q(q, "Định nghĩa & Khái niệm", m.group(0))
            if len(questions) >= 4:
                break

        # Pattern 3: Limitations / Problems ("Hạn chế", "Nhược điểm", "Thách thức")
        limit_pattern = re.compile(
            r"(hạn\s+chế|nhược\s+điểm|thách\s+thức|khó\s+khăn|vấn\s+đề)\s+(ở\s+đây|của\s+hệ\s+thống|chính|được\s+xác\s+định|trong)?\s*[:\s]+([^.!?]+)",
            re.IGNORECASE,
        )
        for m in limit_pattern.finditer(raw_text):
            excerpt = m.group(0).strip()
            add_q(f"Những hạn chế hoặc thách thức nào được chỉ rõ trong '{title}'?", "Hạn chế & Thách thức", excerpt)
            break

        # Pattern 4: Mechanism / Architecture / Solutions ("mô hình", "kiến trúc", "phương pháp", "giải pháp")
        mech_pattern = re.compile(
            r"(hệ\s+thống|mô\s+hình|kiến\s+trúc|phương\s+pháp|thuật\s+toán)\s+([A-ZÀ-Ỹa-zà-ỹ0-9_\-\s]{4,30})\s+(hoạt\s+động|kết\s+hợp|áp\s+dụng|đạt\s+được|bao\s+gồm)",
            re.IGNORECASE,
        )
        for m in mech_pattern.finditer(raw_text):
            name = m.group(2).strip()
            if 2 <= len(name.split()) <= 6:
                q = f"Kiến trúc hoặc cơ chế hoạt động của {name} như thế nào?"
                add_q(q, "Kiến trúc & Cơ chế", m.group(0))
            if len(questions) >= 5:
                break

        # Pattern 5: Quantitative Metrics / Numbers ("%", "ms", "MB", "GB", "tokens")
        num_pattern = re.compile(
            r"([^.!?\n]{15,90}\b\d+(\.\d+)?\s*(%|token|ms|gb|mb|mw|triệu|nghìn)\b[^.!?\n]{10,60})",
            re.IGNORECASE,
        )
        for m in num_pattern.finditer(raw_text):
            excerpt = m.group(1).strip()
            add_q(f"Tài liệu ghi nhận những kết quả hoặc số liệu định lượng cụ thể nào?", "Số liệu & Định lượng", excerpt)
            break

        # Fallback questions with guaranteed document relevance
        defaults = [
            (f"Tài liệu '{title}' tóm tắt những nội dung chính nào?", "Tổng quan"),
            (f"Các giải pháp kỹ thuật chính được đề xuất trong '{title}' là gì?", "Giải pháp"),
            (f"Mục tiêu trọng tâm của '{title}' được xác định như thế nào?", "Mục tiêu"),
        ]
        for def_q, def_cat in defaults:
            if len(questions) < max_questions:
                add_q(def_q, def_cat)

        return questions[:max_questions]

    @classmethod
    def find_best_grounded_answer(
        cls,
        query: str,
        evidence_passages: List[Any],
        answer_mode: str = "balanced",
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Locates the exact span answering the query from the evidence passages.

        Two invariants are enforced:
        1. Precision — the returned core is the minimal clause that answers the
           question (answering "bao nhiêu %" never returns the whole paragraph).
        2. Question tail — every answer ends with a self-study question block.
        """
        from src.hybrid_qa.answer_span import (
            ensure_trailing_question,
            extract_answer_span,
            generate_followup_questions,
            relevance_score,
            select_best_sentence,
            split_sentences,
            strip_question_block,
        )

        MIN_RELEVANCE = 0.34

        from src.self_study.persona.academic_persona import (
            AcademicIntent,
            get_academic_persona_engine,
        )
        persona = get_academic_persona_engine()
        persona_intent = persona.classify_intent(query)
        if persona_intent in (
            AcademicIntent.GREETING,
            AcademicIntent.FAREWELL,
            AcademicIntent.GRATITUDE,
            AcademicIntent.SYSTEM_CAPABILITY,
        ):
            return persona.handle_social_query(query, intent=persona_intent), []

        if not evidence_passages:
            return "Không tìm thấy đủ thông tin bằng chứng trong tài liệu để trả lời câu hỏi này.", []

        # Collect every sentence from every evidence passage.
        sentence_pool: List[str] = []
        sentence_sections: List[str] = []
        for passage in evidence_passages:
            passage_text = getattr(passage, "text", "") if hasattr(passage, "text") else str(passage)
            passage_section = getattr(passage, "section_title", "") or ""
            for sentence in split_sentences(passage_text):
                sentence_pool.append(sentence)
                sentence_sections.append(passage_section)

        from src.web.token_optimizer import expand_query_semantics
        exp_query, _ = expand_query_semantics(query)

        best_sentence = select_best_sentence(exp_query, sentence_pool, sentence_sections)
        best_passage_idx = 0
        if best_sentence:
            for idx, passage in enumerate(evidence_passages):
                passage_text = getattr(passage, "text", "") if hasattr(passage, "text") else str(passage)
                if best_sentence[:40] in cls._clean_text(passage_text):
                    best_passage_idx = idx
                    break
        elif sentence_pool:
            best_sentence = sentence_pool[0]

        # Relevance gate: refuse instead of answering from an unrelated passage.
        matched_section = ""
        if best_sentence in sentence_pool:
            matched_section = sentence_sections[sentence_pool.index(best_sentence)]
        combined_pool_text = " ".join(sentence_pool[:4])
        combined_sec_text = " ".join(sentence_sections[:4])
        rel = max(
            relevance_score(query, best_sentence, extra=matched_section),
            relevance_score(exp_query, best_sentence, extra=matched_section),
            relevance_score(query, combined_pool_text, extra=combined_sec_text) * 0.85,
            relevance_score(exp_query, combined_pool_text, extra=combined_sec_text) * 0.85,
        )
        if not best_sentence or rel < MIN_RELEVANCE:
            return "Không tìm thấy đủ thông tin bằng chứng trong tài liệu để trả lời câu hỏi này.", []

        # Deduplicate citations: one citation per distinct document & section.
        dedup_citations = []
        seen_keys = set()
        passage_to_cit_label: List[str] = []
        for idx, p in enumerate(evidence_passages, start=1):
            doc_id = getattr(p, "document_id", "DOC")
            doc_title = getattr(p, "document_title", doc_id) or doc_id
            sec_title = getattr(p, "section_title", "Phần nội dung") or "Phần nội dung"
            para_idx = getattr(p, "paragraph_index", idx)

            key = (doc_id, sec_title, para_idx)
            if key not in seen_keys:
                seen_keys.add(key)
                cit_label = f"[{len(dedup_citations) + 1}]"
                dedup_citations.append({
                    "citation_index": len(dedup_citations) + 1,
                    "citation_label": cit_label,
                    "passage_id": getattr(p, "passage_id", f"PASSAGE_{idx}"),
                    "document_id": doc_id,
                    "document_title": doc_title,
                    "section_title": sec_title,
                    "paragraph_index": para_idx,
                    "score": round(float(getattr(p, "score", 1.0)), 4),
                    "text": getattr(p, "text", "")[:400],
                })
                passage_to_cit_label.append(cit_label)
            else:
                # Find existing label for this key
                matched_lbl = "[1]"
                for dc in dedup_citations:
                    if (dc["document_id"], dc["section_title"], dc["paragraph_index"]) == key:
                        matched_lbl = dc["citation_label"]
                        break
                passage_to_cit_label.append(matched_lbl)

        # Map each sentence in sentence_pool to its passage citation label
        sentence_cit_labels: List[str] = []
        for p_idx, passage in enumerate(evidence_passages):
            passage_text = getattr(passage, "text", "") if hasattr(passage, "text") else str(passage)
            p_lbl = passage_to_cit_label[p_idx] if p_idx < len(passage_to_cit_label) else "[1]"
            for _ in split_sentences(passage_text):
                sentence_cit_labels.append(p_lbl)

        # Check compound query decomposition so multi-part questions cover all sub-queries
        sub_query_sentences: List[Tuple[str, str]] = []
        try:
            from src.reasoning.query_decomposer import QueryDecomposer
            decomp = QueryDecomposer().decompose(query)
            if decomp.is_compound and len(decomp.sub_queries) >= 2:
                for sq in decomp.sub_queries:
                    sq_exp, _ = expand_query_semantics(sq.text)
                    sq_best = select_best_sentence(sq_exp, sentence_pool, sentence_sections)
                    if sq_best and sq_best in sentence_pool:
                        s_idx = sentence_pool.index(sq_best)
                        s_lbl = sentence_cit_labels[s_idx] if s_idx < len(sentence_cit_labels) else "[1]"
                        if all(sq_best != existing_s for existing_s, _ in sub_query_sentences):
                            sub_query_sentences.append((sq_best, s_lbl))
                if sub_query_sentences:
                    best_sentence = sub_query_sentences[0][0]
        except Exception:
            pass

        cit_1 = dedup_citations[0]["citation_label"] if dedup_citations else "[1]"
        if best_sentence in sentence_pool:
            b_idx = sentence_pool.index(best_sentence)
            if b_idx < len(sentence_cit_labels):
                cit_1 = sentence_cit_labels[b_idx]

        q_norm = strip_accents((query or "").lower())
        general_section_markers = (
            "noi dung chi tiet", "thong tin cot loi", "yeu cau cua muc",
            "tong quan", "tom tat", "gioi thieu chung", "noi dung chinh",
            "noi dung cua muc", "trinh bay muc", "bao gom nhung gi",
        )
        is_general_section_query = any(m in q_norm for m in general_section_markers)
        aspect_concept = any(k in q_norm for k in ("khai niem", "dinh nghia", "la gi", "y nghia", "vai tro", "muc dich"))
        aspect_mechanism = any(k in q_norm for k in (
            "nguyen ly", "co che", "cong thuc", "dac diem", "tinh chat",
            "cach hoat dong", "hoat dong nhu the nao", "hoat dong the nao",
            "quy trinh", "thuat toan", "ky thuat", "trien khai",
        ))
        aspect_example = any(k in q_norm for k in ("vi du", "minh hoa", "bai toan", "truong hop", "ap dung"))
        aspect_eval = any(k in q_norm for k in (
            "uu diem", "nhuoc diem", "han che", "loi ich", "ket qua",
            "so lieu", "hieu nang", "hieu qua", "do chinh xac", "toc do", "bao nhieu", "%",
        ))
        has_specific_aspect = aspect_concept or aspect_mechanism or aspect_example or aspect_eval

        q_intent_tokens = {
            t for t in re.findall(r"[a-z0-9_]+", q_norm)
            if len(t) >= 2 and t not in {
                "la", "gi", "cua", "muc", "trong", "nay", "nao", "nhu", "the", "cho",
                "biet", "ve", "co", "duoc", "cac", "nhung", "mot", "va", "voi", "toi", "minh", "ta", "ho",
            }
        }
        exp_norm = strip_accents((exp_query or "").lower())
        exp_tokens = {
            t for t in re.findall(r"[a-z0-9_]+", exp_norm)
            if len(t) >= 2 and t not in q_intent_tokens
        }

        def _is_heading_sent(sent: str, sec: str = "") -> bool:
            st = (sent or "").strip()
            if not st:
                return True
            if re.match(
                r"^\s*(?:#{1,6}\s+|(?:\b(?:topic|chủ đề|chu de|bài|bai|mục|muc|chương|chuong|phần|phan|section|chapter|lesson|unit)\s*\d+[:.\-–]?)|(?:\d+(?:\.\d+)*\.\s+[A-ZÀ-Ỹ]))",
                st,
                re.IGNORECASE,
            ):
                return True
            if sec and (
                st.lower() == sec.strip().lower()
                or (len(st) >= 4 and st.rstrip(".:") in sec.strip())
            ):
                return True
            return False

        def _score_body_sentence(sent: str) -> float:
            s_norm = strip_accents(sent.lower())
            s_tokens = set(re.findall(r"[a-z0-9_]+", s_norm))
            score = len(q_intent_tokens & s_tokens) * 2.0 + len(exp_tokens & s_tokens) * 0.75
            is_ex_sent = any(c in s_norm for c in ("vi du", "minh hoa", "gia su", "cho p =", "ta co", "chang han"))
            if aspect_concept:
                if not is_ex_sent and any(c in s_norm for c in (" la ", "duoc dinh nghia", "khai niem", "phuong phap", "he thong", "dung de", "nham", "bao gom")):
                    score += 6.0
                if is_ex_sent or any(c in s_norm for c in ("cho p =", "ta co:", "mod 26")):
                    score -= 4.0
            if aspect_mechanism:
                mech_cues = ("nguyen ly", "co che", "hoat dong", "cong thuc", "dac diem", "tinh chat", "quy trinh", "buoc", "thuat toan", "thong qua", "dua tren", "=", "mod ")
                mech_hits = sum(1 for c in mech_cues if c in s_norm)
                if mech_hits:
                    score += min(9.0, 4.5 + mech_hits * 2.0)
                if "cong thuc" in q_norm and ("cong thuc" in s_norm or "=" in s_norm):
                    score += 4.0
            if aspect_example:
                if is_ex_sent or any(c in s_norm for c in ("ap dung", "bai toan", "truong hop")) or re.search(r"\d", s_norm):
                    score += 6.5
            if aspect_eval:
                if any(c in s_norm for c in ("uu diem", "loi ich", "nhuoc diem", "han che", "kho khan", "rui ro", "ket qua", "dat ", "cai thien", "tang", "giam", "%")) or re.search(r"\d+(?:[.,]\d+)?\s*%", s_norm):
                    score += 6.5
            return score

        # Check if best_sentence is a heading or equals section_title
        is_heading = _is_heading_sent(best_sentence, matched_section)

        best_passage = evidence_passages[best_passage_idx]
        best_passage_text = getattr(best_passage, "text", "") if hasattr(best_passage, "text") else str(best_passage)
        passage_sents = split_sentences(best_passage_text)

        body_sents = [
            s.strip() for s in passage_sents
            if s.strip() and not _is_heading_sent(s, matched_section)
        ]
        is_topic_or_sec_query = bool(re.search(r"\b(?:topic|chủ đề|chu de|bài|bai|mục|muc|chương|chuong|phần|phan|section|chapter)\s*\d+\b", q_norm))
        if is_heading:
            if not is_topic_or_sec_query and not is_general_section_query and len(body_sents) >= 2 and (has_specific_aspect or len(q_intent_tokens) >= 2):
                ranked_body = sorted(
                    enumerate(body_sents),
                    key=lambda item: (_score_body_sentence(item[1]), -item[0]),
                    reverse=True,
                )
                best_sentence = ranked_body[0][1]
                is_heading = False
        elif has_specific_aspect and not is_general_section_query and len(body_sents) >= 2:
            ranked_body = sorted(
                enumerate(body_sents),
                key=lambda item: (_score_body_sentence(item[1]), -item[0]),
                reverse=True,
            )
            top_body = ranked_body[0][1]
            if _score_body_sentence(top_body) >= _score_body_sentence(best_sentence) + 3.0:
                best_sentence = top_body

        if is_heading:
            # If best sentence is a heading, answer with heading plus next 2-3 body sentences
            idx_in_passage = -1
            for i, s in enumerate(passage_sents):
                if best_sentence[:40] in s or s[:40] in best_sentence:
                    idx_in_passage = i
                    break
            if idx_in_passage >= 0:
                heading_part = passage_sents[idx_in_passage].strip()
                next_sents = [s.strip() for s in passage_sents[idx_in_passage + 1 : idx_in_passage + 4] if s.strip()]
                if next_sents:
                    h_end = "" if heading_part.endswith((".", "!", "?", ":")) else "."
                    answer_core = f"{heading_part}{h_end} {' '.join(next_sents)}"
                else:
                    answer_core = heading_part
            else:
                answer_core = best_sentence
        else:
            # Precision cut: minimal answering span in concise mode, full rich sentence in balanced/detailed
            answer_core = extract_answer_span(exp_query, best_sentence) if best_sentence else ""
            if (
                not answer_core
                or len(answer_core.split()) < 4
                or answer_core.strip().endswith(":")
                or ":" in best_sentence
                or (answer_mode != "concise" and len(answer_core) < len(best_sentence) * 0.75)
            ):
                answer_core = best_sentence

        if answer_core and not answer_core.endswith((".", "!", "?")):
            answer_core += "."

        # Follow-up questions grounded in the evidence that was actually used.
        target_sec_name = dedup_citations[0]["section_title"] if dedup_citations else "chương mục"
        all_context_text = "\n".join(
            getattr(p, "text", "") if hasattr(p, "text") else str(p)
            for p in evidence_passages[:3]
        )
        if model is not None and tokenizer is not None:
            from src.hybrid_qa.answer_span import generate_followup_questions_llm
            follow_ups = generate_followup_questions_llm(
                query=query,
                context=all_context_text,
                model=model,
                tokenizer=tokenizer,
                count=3,
            )
        else:
            follow_ups = generate_followup_questions(
                query=query,
                context=all_context_text,
                section_title=target_sec_name,
            )

        if answer_mode == "concise":
            body = f"{answer_core} {cit_1}".strip()
            active_citations = [c for c in dedup_citations if c["citation_label"] == cit_1]
            if not active_citations:
                active_citations = dedup_citations[:1]
        else:
            doc_display_title = dedup_citations[0]["document_title"] if dedup_citations else "tài liệu"
            is_comparison = bool(re.search(
                r"\b(?:so sánh|đối chiếu|khác biệt|khác nhau|so với|compare|comparison|versus|\bvs\b)\b",
                query,
                re.IGNORECASE,
            ))
            has_sub_queries = bool(sub_query_sentences and len(sub_query_sentences) >= 2)

            if is_comparison:
                # Comparison queries: support cross-passage delta reasoning
                cross_parts = [f"{answer_core} {cit_1}"]
                for sq_s, sq_lbl in sub_query_sentences:
                    if sq_s.strip() not in answer_core:
                        s_fmt = sq_s.strip() if sq_s.strip().endswith((".", "!", "?")) else f"{sq_s.strip()}."
                        cross_parts.append(f"{s_fmt} {sq_lbl}")

                delta_bullet = None
                all_used_pairs = [(best_sentence, cit_1)] + [(s, sentence_cit_labels[i]) for i, s in enumerate(sentence_pool) if s != best_sentence]
                unit_buckets: Dict[str, List[Tuple[float, str, str]]] = {}
                for u_sent, u_lbl in all_used_pairs:
                    for nm in re.finditer(
                        r"(\d+(?:[.,]\d+)?)\s*(%|(?:mb|gb|ms|s|tokens?|triệu|tỷ)\b)",
                        u_sent,
                        re.IGNORECASE,
                    ):
                        raw_num = nm.group(1).replace(",", ".")
                        unit_key = nm.group(2).lower()
                        try:
                            num_val = float(raw_num)
                        except ValueError:
                            continue
                        unit_buckets.setdefault(unit_key, [])
                        if all(abs(num_val - existing[0]) > 1e-6 for existing in unit_buckets[unit_key]):
                            unit_buckets[unit_key].append((num_val, nm.group(0).strip(), u_lbl))

                for unit_key, entries in unit_buckets.items():
                    if len(entries) >= 2:
                        v1, raw1, lbl1 = entries[0]
                        v2, raw2, lbl2 = entries[1]
                        delta_val = round(abs(v1 - v2), 2)
                        delta_str = f"{delta_val:g}"
                        unit_disp = "%" if unit_key == "%" else f" {entries[0][1].split()[-1] if ' ' in entries[0][1] else unit_key}"
                        cit_pair = lbl1 if lbl1 == lbl2 else f"{lbl1} {lbl2}"
                        delta_bullet = (
                            f"- **Đối chiếu định lượng tự động**: Mức chênh lệch giữa hai mốc số liệu được ghi nhận là **{delta_str}{unit_disp}** ({raw1} so với {raw2}). {cit_pair}"
                        )
                        break

                body = " ".join(cross_parts)
                if delta_bullet:
                    body = f"{body}\n\n{delta_bullet}"
                active_citations = dedup_citations
            elif has_sub_queries:
                # Decomposed compound queries: include answers for all sub-queries
                primary_supports = []
                if answer_mode != "concise":
                    for s in passage_sents:
                        s_c = clean_sentence_abstractive(s.strip())
                        if s_c and s_c not in answer_core and not any(s_c in sq[0] for sq in sub_query_sentences):
                            primary_supports.append(s_c)
                            if len(primary_supports) >= (3 if answer_mode in ("detailed", "comprehensive") else 1):
                                break

                main_prose = [answer_core] + primary_supports
                fmt_main = [s if s.endswith((".", "!", "?")) else f"{s}." for s in main_prose]
                cross_parts = [f"{' '.join(fmt_main)} {cit_1}"]
                for sq_s, sq_lbl in sub_query_sentences:
                    if sq_s.strip() not in answer_core and not any(sq_s.strip() in p for p in primary_supports):
                        s_fmt = sq_s.strip() if sq_s.strip().endswith((".", "!", "?")) else f"{sq_s.strip()}."
                        cross_parts.append(f"{s_fmt} {sq_lbl}")
                body = " ".join(cross_parts)
                active_citations = dedup_citations
            else:
                # Comprehensive Grounded Prose & Multi-Passage Synthesis
                if answer_mode in ("detailed", "comprehensive"):
                    max_supporting = 10
                    supporting_sents = []
                    for s in passage_sents:
                        s_clean = s.strip()
                        if not s_clean:
                            continue
                        if s_clean in answer_core or answer_core.startswith(s_clean):
                            continue
                        if s_clean not in supporting_sents:
                            supporting_sents.append(s_clean)
                            if len(supporting_sents) >= max_supporting:
                                break
                else:
                    cand_sents = []
                    seen_cands = set()
                    for idx_s, s in enumerate(passage_sents):
                        s_clean = s.strip()
                        if not s_clean:
                            continue
                        if s_clean in answer_core or answer_core.startswith(s_clean):
                            continue
                        if s_clean not in seen_cands:
                            seen_cands.add(s_clean)
                            cand_sents.append((idx_s, s_clean))

                    if is_general_section_query:
                        supporting_sents = [s_clean for _, s_clean in cand_sents[:6]]
                    else:
                        scored_cands = sorted(
                            cand_sents,
                            key=lambda item: (_score_body_sentence(item[1]), -item[0]),
                            reverse=True,
                        )
                        relevant_cands = [s_clean for _, s_clean in scored_cands if _score_body_sentence(s_clean) > 0]
                        supporting_sents = relevant_cands[:4] if relevant_cands else [s_clean for _, s_clean in cand_sents[:2]]

                clean_core = clean_sentence_abstractive(answer_core)
                clean_supports = [clean_sentence_abstractive(s) for s in supporting_sents]
                clean_supports = [s for s in clean_supports if s and s != clean_core and s not in clean_core]

                all_sents = [clean_core] + clean_supports
                fmt_sents = [s if s.endswith((".", "!", "?")) else f"{s}." for s in all_sents]
                joined_prose = " ".join(fmt_sents)
                body = f"{joined_prose} {cit_1}".strip()
                active_citations = [c for c in dedup_citations if c["citation_label"] == cit_1]
                if not active_citations:
                    active_citations = dedup_citations[:1]

            # Multi-facet & multi-passage synthesis across all evidence passages
            q_norm = strip_accents((query or "").lower())
            q_nums = re.findall(r"\b(?:topic|chủ đề|chu de|bài|bai|mục|muc|chương|chuong|phần|phan|section)\s*(\d+)\b", q_norm)
            query_target_num = q_nums[0] if q_nums else None

            multi_facet_bullets: List[str] = []
            used_citation_labels: Set[str] = {c["citation_label"] for c in active_citations}
            seen_bullet_texts: Set[str] = set()

            def _add_bullet(b_text: str, cit_label: str):
                b_clean = b_text.strip()
                b_norm = strip_accents(b_clean.lower())
                if not b_clean or any(b_norm in s or s in b_norm for s in seen_bullet_texts):
                    return
                if b_clean in body or b_norm in strip_accents(body.lower()):
                    return
                seen_bullet_texts.add(b_norm)
                multi_facet_bullets.append(b_clean)
                used_citation_labels.add(cit_label)

            # Scan passages for complementary grounded facets
            for p_idx, p in enumerate(evidence_passages):
                p_text = getattr(p, "text", "") if hasattr(p, "text") else str(p)
                p_sec = getattr(p, "section_title", "") or ""
                p_cit = passage_to_cit_label[p_idx] if p_idx < len(passage_to_cit_label) else f"[{p_idx+1}]"

                # Check topic isolation: skip conflicting topic numbers
                if query_target_num:
                    p_nums = re.findall(r"\b(?:topic|chủ đề|chu de|bài|bai|mục|muc|chương|chuong|phần|phan|section)\s*(\d+)\b", (p_sec + " " + p_text).lower())
                    if p_nums and query_target_num not in p_nums:
                        continue

                lines = [l.strip() for l in p_text.splitlines() if l.strip()]
                p_sents = split_sentences(p_text)
                unit_candidates = lines + [s for s in p_sents if s not in lines]

                # 1. Named sub-methods, components, specific categories
                for u in unit_candidates:
                    u_clean = clean_sentence_abstractive(u)
                    m_label = re.match(r"^([A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9_\-\s]{2,35}):\s*(.+)", u_clean)
                    if m_label:
                        label_name = m_label.group(1).strip()
                        label_content = m_label.group(2).strip()
                        norm_lbl = strip_accents(label_name.lower())
                        if norm_lbl in ("vi du", "vi du minh hoa"):
                            _add_bullet(f"- **Ví dụ minh họa**: {label_content} {p_cit}", p_cit)
                        elif norm_lbl in ("cong thuc", "cong thuc ma hoa", "cong thuc giai ma"):
                            _add_bullet(f"- **Công thức toán học**: {label_content} {p_cit}", p_cit)
                        elif len(label_content.split()) >= 4 and not any(skip in norm_lbl for skip in ["nguon", "tai lieu", "chu de", "topic", "trang"]):
                            prefix_kw = "" if any(w in norm_lbl for w in ["phuong phap", "he ma", "thanh phan", "ky thuat", "giai thuat", "mo hinh", "kien truc"]) else "Phương pháp / Thành phần "
                            _add_bullet(f"- **{prefix_kw}{label_name}**: {label_content} {p_cit}", p_cit)

                    m_sys = re.match(r"^(?:\d+\.)?\s*(Hệ mã\s+[^\n:]+|HỆ MÃ\s+[^\n:]+|Mã hóa\s+[^\n:]+)", u, re.IGNORECASE)
                    if m_sys:
                        raw_sys = clean_sentence_abstractive(m_sys.group(1).strip().rstrip(".:"))
                        _add_bullet(f"- **Hệ thống chuyên biệt**: {raw_sys} {p_cit}", p_cit)

                # 2. Mathematical formulas & concrete examples
                for s in p_sents:
                    s_clean = clean_sentence_abstractive(s)
                    s_norm = strip_accents(s_clean.lower())
                    is_formula = any(k in s_norm for k in ["cong thuc", "ta co: zi =", "p = c = k", "mod 26", "mod "]) or bool(re.search(r"\b[CPEK]\s*=\s*\(?[^.!?;]+\)?\s*mod", s_clean))
                    is_ex = any(k in s_norm for k in ["cho p =", "vi du", "chang han", "gia su"])
                    if is_formula and not any(s_clean in b for b in multi_facet_bullets):
                        _add_bullet(f"- **Công thức & Quy tắc**: {s_clean} {p_cit}", p_cit)
                    elif is_ex and len(s_clean.split()) >= 4 and not any(s_clean in b for b in multi_facet_bullets):
                        _add_bullet(f"- **Ví dụ minh họa**: {s_clean} {p_cit}", p_cit)

                # 3. Quantitative metrics & empirical benchmarks
                p_sents = split_sentences(p_text)
                for s in p_sents:
                    s_clean = clean_sentence_abstractive(s)
                    s_norm = strip_accents(s_clean.lower())
                    has_metric = bool(re.search(r"\b\d+(?:[.,]\d+)?\s*(?:%|mb|gb|ms|s|token|triệu|tỷ|mw|kw)\b", s_norm))
                    is_eval_phrase = any(w in s_norm for w in ["thuc nghiem", "hieu nang", "dat duoc", "giam", "tang", "tiet kiem", "chinh xac", "f1", "chi phi"])
                    if has_metric and is_eval_phrase and s_clean not in body:
                        _add_bullet(f"- **Số liệu & Thực nghiệm**: {s_clean} {p_cit}", p_cit)

                # 4. Limitations, challenges & security considerations
                for s in p_sents:
                    s_clean = clean_sentence_abstractive(s)
                    s_norm = strip_accents(s_clean.lower())
                    is_limit = any(w in s_norm for w in ["han che", "nhuoc diem", "thach thuc", "kho khan", "lo hong", "rui ro", "diem yeu", "khong gian khoa qua nho"])
                    if is_limit and len(s_clean.split()) >= 5 and s_clean not in body:
                        _add_bullet(f"- **Đặc điểm & Hạn chế**: {s_clean} {p_cit}", p_cit)

            if multi_facet_bullets:
                max_b = 8 if answer_mode in ("detailed", "comprehensive") else 4
                body = f"{body}\n\n" + "\n".join(multi_facet_bullets[:max_b])
                active_citations = [c for c in dedup_citations if c["citation_label"] in used_citation_labels]
                if not active_citations:
                    active_citations = dedup_citations
            elif answer_mode != "concise" and len(evidence_passages) >= 2:
                # Complementary multi-passage sentences if no distinct bullet facets were extracted
                extra_cross_sents = []
                for p_idx, p in enumerate(evidence_passages):
                    if p_idx == best_passage_idx:
                        continue
                    if query_target_num:
                        p_sec = getattr(p, "section_title", "") or ""
                        p_nums = re.findall(r"\b(?:topic|chủ đề|chu de|bài|bai|mục|muc|chương|chuong|phần|phan|section)\s*(\d+)\b", (p_sec + " " + getattr(p, "text", "")).lower())
                        if p_nums and query_target_num not in p_nums:
                            continue
                    p_sents = split_sentences(getattr(p, "text", ""))
                    p_cit = passage_to_cit_label[p_idx] if p_idx < len(passage_to_cit_label) else f"[{p_idx+1}]"
                    for s in p_sents:
                        s_clean = clean_sentence_abstractive(s)
                        if len(s_clean.split()) >= 6 and s_clean not in body and _score_body_sentence(s_clean) > 0:
                            s_fmt = s_clean if s_clean.endswith((".", "!", "?")) else f"{s_clean}."
                            extra_cross_sents.append(f"{s_fmt} {p_cit}")
                            used_citation_labels.add(p_cit)
                            if len(extra_cross_sents) >= (4 if answer_mode in ("detailed", "comprehensive") else 2):
                                break
                    if len(extra_cross_sents) >= 4:
                        break
                if extra_cross_sents:
                    body = f"{body} " + " ".join(extra_cross_sents)
                    active_citations = [c for c in dedup_citations if c["citation_label"] in used_citation_labels]

            if answer_mode == "detailed":
                # Defense Strategist integration (Wave 3)
                try:
                    from src.brain.defense_strategist import DefenseStrategist
                    strategist = DefenseStrategist()
                    plan = strategist.plan(query)
                    if plan.archetype != "GENERIC" and plan.confidence >= 0.35:
                        cand_dicts = [
                            {
                                "sentence": s,
                                "citation_label": lbl,
                                "section_title": sec,
                            }
                            for s, lbl, sec in zip(sentence_pool, sentence_cit_labels, sentence_sections)
                        ]
                        arg_block = strategist.assemble_argument(plan, cand_dicts)
                        if arg_block and "- **" in arg_block:
                            body = f"{body}\n\n{arg_block}"
                except Exception:
                    pass

                sec_list = sorted({c["section_title"] for c in dedup_citations if c.get("section_title")})
                sec_summary = ", ".join(f"`{s}`" for s in sec_list[:3]) if sec_list else f"**{doc_display_title}**"
                cit_badges = " ".join(c["citation_label"] for c in dedup_citations[:3])
                body = (
                    f"{body}\n\n"
                    f"💡 **Tổng hợp & Đúc kết chuyên sâu:**\n"
                    f"Nội dung trên được tổng hợp đối chiếu trực tiếp từ {len(dedup_citations)} đoạn chứng cứ thuộc {sec_summary} "
                    f"trong tài liệu **{doc_display_title}** {cit_badges}, bảo đảm tính toàn vẹn ngữ cảnh và không suy diễn ngoài văn bản."
                )

        dedup_citations = active_citations

        # Sources first, then the mandatory trailing question block.
        sources_lines = [
            f"{c['citation_label']} **{c['document_title']}** — {c['section_title']} — Đoạn {c['paragraph_index']}"
            for c in dedup_citations[:3]
        ]

        body = strip_question_block(body)
        if sources_lines:
            body = f"{body}\n\n### Nguồn tham khảo\n" + "\n\n".join(sources_lines)

        full_answer = ensure_trailing_question(body, follow_ups)
        return full_answer, dedup_citations
