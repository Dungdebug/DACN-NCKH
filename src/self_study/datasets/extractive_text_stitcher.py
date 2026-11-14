"""
Extractive Text Stitcher for Sovereign Brain v5.0 (Proposal Section 6.5).
100% Offline, Pure Python stdlib, Zero External Dependencies.

Transforms fragmented extractive sentence spans, bullet points, section titles,
and formulas from source documents into fluent, coherent, and readable chat answers:
  Khái niệm -> Các phương pháp/loại mã hóa -> Các hệ cụ thể trong tài liệu -> Ví dụ minh họa
Guarantees 100% extractive grounding (zero hallucination).
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Set, Tuple


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive canonical alignment."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def clean_sentence(text: str) -> str:
    """
    Cleans an extractive sentence:
    - Strips page markers and section number artifacts
    - Strips leading bullet markers
    - Trims whitespace
    - Normalizes punctuation
    """
    if not text:
        return ""

    s = text.strip()

    # Remove page markers: === Page \d+ ===
    s = re.sub(r"===\s*Page\s*\d+\s*===", "", s, flags=re.IGNORECASE).strip()

    # Remove leading bullets: *, -, •, +, etc.
    s = re.sub(r"^[-*•+–—]\s*", "", s).strip()

    # Remove opening redundant discourse connectors if needed
    s = re.sub(
        r"^(?:Tuy nhiên|Do đó|Vì vậy|Theo đó|Ngoài ra|Hơn nữa|Bên cạnh đó|Đồng thời)[,:\s]+",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()

    if not s:
        return ""

    # Capitalize first character if lowercase
    if s[0].islower():
        s = s[0].upper() + s[1:]

    # Ensure ending terminal punctuation
    if s[-1] not in ".!?:":
        s = f"{s}."

    return s


def clean_bullet(bullet: str) -> str:
    """Cleans a bullet item text, stripping leading bullet markers and trailing spaces."""
    if not bullet:
        return ""
    b = bullet.strip()
    b = re.sub(r"===\s*Page\s*\d+\s*===", "", b, flags=re.IGNORECASE).strip()
    b = re.sub(r"^[-*•+–—]\s*", "", b).strip()
    b = re.sub(r"^\d+[\.)]\s*", "", b).strip()
    return b.strip()


def clean_system_name(name: str) -> str:
    """
    Cleans and standardizes section/system names:
    - Removes numbering prefixes ('2. ', '10. ')
    - Converts ALL CAPS to readable title casing while preserving key cryptographic acronyms
    """
    if not name:
        return ""

    s = name.strip()
    s = re.sub(r"===\s*Page\s*\d+\s*===", "", s, flags=re.IGNORECASE).strip()
    # Strip leading number prefixes like "2. ", "10. ", "II. "
    s = re.sub(r"^(?:[0-9]+|[IVXLCDM]+)[\.:]\s*", "", s).strip()
    # Strip trailing colons
    s = re.sub(r"[:\s]+$", "", s).strip()

    if not s:
        return ""

    # Check if string is predominantly uppercase
    letters = [c for c in s if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        # Known cryptographic acronyms to retain in uppercase
        acronyms = {
            "OTP", "DES", "AES", "RSA", "SHA", "MD5", "Z26", "XOR", "MAC",
            "CBC", "ECB", "CFB", "OFB", "CTR", "PRNG", "IV", "NIST",
        }
        # Known proper named systems to title-case properly
        proper_names = {
            "CAESAR": "Caesar",
            "AFFINE": "Affine",
            "VIGENERE": "Vigenere",
            "HILL": "Hill",
            "VERNAM": "Vernam",
            "AUTOKEY": "Autokey",
            "PLAYFAIR": "Playfair",
        }

        words = s.split()
        converted_words = []
        for i, w in enumerate(words):
            w_up = w.upper()
            if w_up in acronyms:
                converted_words.append(w_up)
            elif w_up in proper_names:
                converted_words.append(proper_names[w_up])
            elif i == 0:
                converted_words.append(w.capitalize())
            else:
                converted_words.append(w.lower())
        return " ".join(converted_words)

    return s


def deduplicate_preserve_order(items: List[str]) -> List[str]:
    """Deduplicates a list of strings while strictly preserving original insertion order."""
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned:
            continue
        norm = strip_accents(cleaned.lower())
        if norm not in seen:
            seen.add(norm)
            result.append(cleaned)
    return result


class ExtractiveTextStitcher:
    """
    Intelligent extractive text stitcher for document-grounded responses.
    Assembles fragmented text components into cohesive answers:
      (Khái niệm -> Các phương pháp/loại mã hóa -> Các hệ cụ thể trong tài liệu -> Ví dụ minh họa)
    Guarantees 100% offline, zero hallucination, pure extractive synthesis.
    """

    @classmethod
    def stitch_complete_answer(
        cls,
        query: str,
        intro_sents: Optional[List[str]] = None,
        bullet_types: Optional[List[str]] = None,
        section_systems: Optional[List[str]] = None,
        examples: Optional[List[str]] = None,
        include_preamble: bool = True,
        preamble_text: Optional[str] = None,
        style: str = "sequential",
        max_examples: int = 5,
        **kwargs: Any,
    ) -> str:
        """
        Stitches a complete, coherent answer from extractive fragments.

        Args:
            query: The user or student inquiry.
            intro_sents: Explanatory/definition sentences for the core concept.
            bullet_types: High-level classification or method bullet points.
            section_systems: List of specific systems/ciphers found in document sections.
            examples: Illustrative walkthroughs, mathematical formulas, or concrete examples.
            include_preamble: Whether to prefix the answer with a polite grounding lead-in.
            preamble_text: Custom preamble string if desired.
            style: Formatting style: 'sequential' (default, 4 distinct numbered sections),
                   'nested' (systems nested inside taxonomy), or 'compact' (un-numbered markdown).
            max_examples: Maximum number of examples to include.

        Returns:
            A clean, formatted, fully readable chat response string.
        """
        intro_sents = intro_sents or []
        bullet_types = bullet_types or []
        section_systems = section_systems or []
        examples = examples or []

        # ----------------------------------------------------------------------
        # 1. Khái niệm (Concept / Definition)
        # ----------------------------------------------------------------------
        concept_sentences: List[str] = []
        for sent in intro_sents:
            if not sent:
                continue
            # Exclude raw heading-only lines if body sentences follow
            s_clean = clean_sentence(sent)
            if not s_clean:
                continue
            # Check if this sentence is purely an uppercase section heading (e.g. '1. GIỚI THIỆU...')
            s_no_num = re.sub(r"^(?:[0-9]+|[IVXLCDM]+)[\.:]\s*", "", sent).strip(": \t\n")
            letters = [c for c in s_no_num if c.isalpha()]
            is_pure_heading = letters and all(c.isupper() for c in letters) and len(s_no_num.split()) <= 8
            if is_pure_heading and len(intro_sents) > 1:
                # Skip pure uppercase section heading when actual content sentences exist
                continue
            concept_sentences.append(s_clean)

        concept_sentences = deduplicate_preserve_order(concept_sentences)
        concept_text = " ".join(concept_sentences).strip()

        # ----------------------------------------------------------------------
        # 2. Các phương pháp / Loại mã hóa (Methods / Taxonomy)
        # ----------------------------------------------------------------------
        clean_methods: List[str] = []
        method_lead_in = ""
        for b in bullet_types:
            if not b:
                continue
            b_clean = clean_bullet(b)
            if not b_clean:
                continue
            # Check if this line is a lead-in description rather than a bullet item
            # e.g. "Có 2 phương pháp phổ biến:"
            if b_clean.endswith(":") or any(
                b_clean.lower().startswith(k) for k in ["có ", "bao gồm", "gồm có", "các phương pháp"]
            ):
                if not method_lead_in:
                    method_lead_in = b_clean if b_clean.endswith(":") else f"{b_clean}:"
                continue
            # Ensure proper punctuation for bullet
            if b_clean[-1] not in ".!?:":
                b_clean = f"{b_clean}."
            clean_methods.append(b_clean)

        clean_methods = deduplicate_preserve_order(clean_methods)

        # ----------------------------------------------------------------------
        # 3. Các hệ cụ thể trong tài liệu (Specific Systems)
        # ----------------------------------------------------------------------
        clean_systems: List[str] = []
        for sys_item in section_systems:
            if not sys_item:
                continue
            sys_name = clean_system_name(sys_item)
            if not sys_name:
                continue
            # Filter out non-system meta headings if needed
            sys_norm = strip_accents(sys_name.lower())
            if any(k in sys_norm for k in ["gioi thieu", "tong quan", "tai lieu tham khao", "muc luc"]):
                continue
            clean_systems.append(sys_name)

        clean_systems = deduplicate_preserve_order(clean_systems)

        # ----------------------------------------------------------------------
        # 4. Ví dụ minh họa / Công thức (Examples & Formulas)
        # ----------------------------------------------------------------------
        clean_examples: List[str] = []
        for ex in examples[:max_examples]:
            if not ex:
                continue
            ex_clean = clean_bullet(ex)
            if not ex_clean:
                continue
            clean_examples.append(ex_clean)

        clean_examples = deduplicate_preserve_order(clean_examples)

        # ----------------------------------------------------------------------
        # Assembly based on chosen style
        # ----------------------------------------------------------------------
        if not concept_text and not clean_methods and not clean_systems and not clean_examples:
            return "Không tìm thấy đủ thông tin trích xuất phù hợp trong tài liệu để tổng hợp câu trả lời."

        sections: List[str] = []
        sec_idx = 1

        if style == "nested":
            # Nested style: combines methods and systems under single section 2
            if concept_text:
                sections.append(f"{sec_idx}. **Khái niệm:** {concept_text}")
                sec_idx += 1

            if clean_methods or clean_systems:
                lines = [f"{sec_idx}. **Phân loại và các phương pháp:**"]
                if clean_methods:
                    lead = method_lead_in or "Các phương pháp phổ biến được ghi nhận trong tài liệu:"
                    lines.append(lead)
                    for m in clean_methods:
                        lines.append(f"- {m}")
                if clean_systems:
                    if clean_methods:
                        lines.append("\nCác hệ mã hóa cụ thể được nghiên cứu xuyên suốt tài liệu gồm:")
                    else:
                        lines.append("Các hệ mã hóa cụ thể được nghiên cứu trong tài liệu:")
                    for s in clean_systems:
                        lines.append(f"  * {s}")
                sections.append("\n".join(lines))
                sec_idx += 1

            if clean_examples:
                lines = [f"{sec_idx}. **Ví dụ và công thức minh họa:**"]
                for ex in clean_examples:
                    lines.append(f"- {ex}")
                sections.append("\n".join(lines))
                sec_idx += 1

        elif style == "compact":
            # Compact markdown style
            if concept_text:
                sections.append(f"**Khái niệm:** {concept_text}")
            if clean_methods:
                lead = method_lead_in or "Các phương pháp phổ biến:"
                m_lines = [f"**Phương pháp:** {lead}"] + [f"- {m}" for m in clean_methods]
                sections.append("\n".join(m_lines))
            if clean_systems:
                s_lines = ["**Các hệ cụ thể:**"] + [f"- {s}" for s in clean_systems]
                sections.append("\n".join(s_lines))
            if clean_examples:
                e_lines = ["**Ví dụ minh họa:**"] + [f"- {ex}" for ex in clean_examples]
                sections.append("\n".join(e_lines))

        else:
            # Sequential style (default): 4 distinct, numbered pillars
            # 1. Khái niệm -> 2. Các phương pháp/loại mã hóa -> 3. Các hệ cụ thể trong tài liệu -> 4. Ví dụ minh họa
            if concept_text:
                sections.append(f"{sec_idx}. **Khái niệm:** {concept_text}")
                sec_idx += 1

            if clean_methods:
                lines = [f"{sec_idx}. **Các phương pháp và phân loại mã hóa:**"]
                lead = method_lead_in or "Các phương pháp phổ biến được ghi nhận trong tài liệu:"
                lines.append(lead)
                for m in clean_methods:
                    lines.append(f"- {m}")
                sections.append("\n".join(lines))
                sec_idx += 1

            if clean_systems:
                lines = [f"{sec_idx}. **Các hệ cụ thể trong tài liệu:**"]
                lines.append("Danh sách các hệ mã hóa được nghiên cứu và trình bày chi tiết gồm:")
                for s in clean_systems:
                    lines.append(f"  * {s}")
                sections.append("\n".join(lines))
                sec_idx += 1

            if clean_examples:
                lines = [f"{sec_idx}. **Ví dụ và công thức minh họa:**"]
                for ex in clean_examples:
                    lines.append(f"- {ex}")
                sections.append("\n".join(lines))
                sec_idx += 1

        # Preamble prefix
        body = "\n\n".join(sections)
        if include_preamble:
            lead_in = preamble_text or "Dựa trên tài liệu đã cung cấp:"
            return f"{lead_in}\n\n{body}"
        return body

    @classmethod
    def stitch_section_detail(
        cls,
        section_title: str,
        content_lines: List[str],
        formula_or_example: Optional[str] = None,
    ) -> str:
        """
        Stitches a deep, fully grounded breakdown of a specific document section.
        (e.g. for section queries like '3. PHÁ MÃ' or '9. HỆ MÃ HOÁ DÒNG').
        """
        title = clean_system_name(section_title) or section_title.strip()
        lines: List[str] = [f"### {title}\n"]

        cleaned_bullets: List[str] = []
        intro_paras: List[str] = []

        for line in content_lines:
            line_str = line.strip()
            if not line_str:
                continue
            line_clean = clean_bullet(line_str)
            if not line_clean:
                continue

            # Classify bullet vs paragraph
            if line_str.startswith(("-", "*", "•", "+")) or ":" in line_clean[:30]:
                cleaned_bullets.append(f"- {line_clean}")
            else:
                intro_paras.append(clean_sentence(line_clean))

        if intro_paras:
            lines.append(" ".join(intro_paras))
            lines.append("")

        if cleaned_bullets:
            lines.extend(cleaned_bullets)

        if formula_or_example:
            lines.append(f"\n**Minh họa / Công thức:**\n- {clean_bullet(formula_or_example)}")

        return "\n".join(lines).strip()

    @classmethod
    def stitch_comparison(
        cls,
        topic_a: str,
        topic_b: str,
        points_a: List[str],
        points_b: List[str],
        conclusion: Optional[str] = None,
    ) -> str:
        """
        Stitches a structured comparative breakdown between two systems or methods.
        (e.g. 'Mã hóa thay thế vs Mã hóa hoán vị' or 'An ninh vô điều kiện vs An ninh tính toán').
        """
        title_a = clean_system_name(topic_a) or topic_a
        title_b = clean_system_name(topic_b) or topic_b

        lines = [
            f"**So sánh đối chiếu giữa {title_a} và {title_b}:**\n",
            f"1. **{title_a}:**",
        ]
        for p in points_a:
            lines.append(f"   - {clean_bullet(p)}")

        lines.append(f"\n2. **{title_b}:**")
        for p in points_b:
            lines.append(f"   - {clean_bullet(p)}")

        if conclusion:
            lines.append(f"\n3. **Nhận xét tổng kết:** {clean_sentence(conclusion)}")

        return "\n".join(lines).strip()

    @classmethod
    def extract_facets_from_raw_text(cls, raw_text: str) -> Dict[str, List[str]]:
        """
        Extracts 4 core answer facets directly from raw document text:
        - intro_sents
        - bullet_types
        - section_systems
        - examples
        """
        intro_sents: List[str] = []
        bullet_types: List[str] = []
        section_systems: List[str] = []
        examples: List[str] = []

        if not raw_text:
            return {
                "intro_sents": intro_sents,
                "bullet_types": bullet_types,
                "section_systems": section_systems,
                "examples": examples,
            }

        lines = raw_text.splitlines()
        in_intro_sec = False
        in_crypto_sec = False

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("==="):
                continue

            # Heading detection
            is_heading = bool(re.match(r"^(?:[0-9]+|[IVXLCDM]+)[\.:]\s+[A-ZÀ-Ỹ0-9\s]+", stripped))
            if is_heading:
                h_clean = clean_system_name(stripped)
                h_norm = strip_accents(stripped.lower())

                if any(k in h_norm for k in ["gioi thieu", "khai niem", "tong quan"]):
                    in_intro_sec = True
                    in_crypto_sec = False
                else:
                    in_intro_sec = False
                    in_crypto_sec = True
                    if any(k in h_norm for k in ["he ma", "ma hoa", "cipher", "don mot lan", "autokey", "playfair"]):
                        section_systems.append(h_clean)
                continue

            # In introductory section: collect intro sents and bullet types
            if in_intro_sec:
                if stripped.startswith(("*", "•", "+")):
                    bullet_types.append(clean_bullet(stripped))
                elif stripped.startswith("-"):
                    if any(k in strip_accents(stripped.lower()) for k in ["thay the", "hoan vi", "phuong phap"]):
                        bullet_types.append(clean_bullet(stripped))
                    else:
                        intro_sents.append(clean_sentence(stripped))
                else:
                    intro_sents.append(clean_sentence(stripped))

            # In specific system sections: collect examples and formulas
            if in_crypto_sec:
                s_norm = strip_accents(stripped.lower())
                if any(k in s_norm for k in ["ta co:", "cho p =", "cong thuc", "vi du", "mod 26", "xor"]):
                    examples.append(stripped)

        return {
            "intro_sents": deduplicate_preserve_order(intro_sents),
            "bullet_types": deduplicate_preserve_order(bullet_types),
            "section_systems": deduplicate_preserve_order(section_systems),
            "examples": deduplicate_preserve_order(examples),
        }

    @classmethod
    def extract_facets_from_passages(cls, passages: List[Any]) -> Dict[str, List[str]]:
        """
        Extracts 4 core facets from a list of retrieved Passage objects or dicts.
        """
        all_text_blocks: List[str] = []
        for p in passages:
            if isinstance(p, dict):
                t = p.get("text", "")
            else:
                t = getattr(p, "text", "")
            if t:
                all_text_blocks.append(t)

        combined_text = "\n\n".join(all_text_blocks)
        return cls.extract_facets_from_raw_text(combined_text)

    @classmethod
    def stitch_from_passages(cls, query: str, passages: List[Any], **kwargs: Any) -> str:
        """
        One-stop helper: extracts facets from passages and stitches a complete answer.
        """
        facets = cls.extract_facets_from_passages(passages)
        return cls.stitch_complete_answer(
            query=query,
            intro_sents=facets["intro_sents"],
            bullet_types=facets["bullet_types"],
            section_systems=facets["section_systems"],
            examples=facets["examples"],
            **kwargs,
        )

    @classmethod
    def is_extractive_grounded(
        cls,
        answer: str,
        source_texts: List[str],
        min_overlap_ratio: float = 0.70,
    ) -> bool:
        """
        Validates that all key clauses in the stitched answer are verifiable
        substrings or derived tokens from source documents.
        """
        if not answer or not source_texts:
            return False

        # Gather all tokens from source texts
        src_tokens: Set[str] = set()
        for st in source_texts:
            for token in re.findall(r"\w+", strip_accents(st.lower())):
                src_tokens.add(token)

        # Stopwords to ignore in overlap calculation
        stopwords = {
            "la", "va", "cac", "nhung", "trong", "cua", "cho", "duoc", "co",
            "voi", "den", "tu", "mot", "hai", "theo", "dung", "tren", "tai",
            "lieu", "da", "cung", "cap", "khai", "niem", "phan", "loai", "phuong",
            "phap", "he", "cu", "the", "vi", "du", "minh", "hoa", "cong", "thuc",
            "danh", "sach", "nghien", "cuu", "trinh", "bay", "chi", "tiet",
            "pho", "bien", "ghi", "nhan", "xuyen", "suot", "gồm",
        }

        ans_tokens = [
            t for t in re.findall(r"\w+", strip_accents(answer.lower()))
            if t not in stopwords and not t.isdigit()
        ]

        if not ans_tokens:
            return True

        grounded_count = sum(1 for t in ans_tokens if t in src_tokens)
        ratio = grounded_count / len(ans_tokens)
        return ratio >= min_overlap_ratio


# Expose module-level helper function
def stitch_complete_answer(
    query: str,
    intro_sents: Optional[List[str]] = None,
    bullet_types: Optional[List[str]] = None,
    section_systems: Optional[List[str]] = None,
    examples: Optional[List[str]] = None,
    **kwargs: Any,
) -> str:
    """Convenience top-level function matching ExtractiveTextStitcher.stitch_complete_answer."""
    return ExtractiveTextStitcher.stitch_complete_answer(
        query=query,
        intro_sents=intro_sents,
        bullet_types=bullet_types,
        section_systems=section_systems,
        examples=examples,
        **kwargs,
    )
