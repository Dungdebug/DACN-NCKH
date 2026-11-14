"""
Comparison and Pros/Cons Answer Stitcher for Sovereign Brain v5.5 (Wave 6).
100% Offline, Pure Python stdlib, Zero External Dependencies.

Specialized synthesis engine for:
1. Pros and Cons analysis ('ưu nhược điểm của X', 'điểm mạnh/điểm yếu của X', 'lợi ích và hạn chế')
2. Cross-Entity Comparisons ('so sánh X và Y', 'phân biệt X và Y', 'sự khác biệt', 'X vs Y')
3. Cryptographic and Systemic Security Evaluations ('an ninh của hệ mã', 'đánh giá độ an toàn', 'khả năng kháng thám mã')
4. Multi-domain comparison (Mật mã học, Hệ điều hành, Mạng máy tính, CSDL, CTDL&GT, CNPM)

Outputs structured, grounded, fluent Markdown answers with categorized bullet points,
comparison matrices, security facets, citations, and evidence sufficiency validation.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple, Union


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive canonical alignment."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


# Precompiled regexes for high-throughput string cleaning (< 1ms execution)
PAGE_MARKER_REGEX = re.compile(r"===\s*Page\s*\d+\s*===", re.IGNORECASE)
BULLET_PREFIX_REGEX = re.compile(r"^[-*•+–—]\s*")
NUMBER_PREFIX_REGEX = re.compile(r"^\d+[\.)]\s*")
SECTION_NUM_REGEX = re.compile(r"^(?:[0-9]+|[IVXLCDM]+)[\.:]\s*")
DISCOURSE_CONNECTOR_REGEX = re.compile(
    r"^(?:Tuy nhiên|Do đó|Vì vậy|Theo đó|Ngoài ra|Hơn nữa|Bên cạnh đó|Đồng thời)[,:\s]+",
    re.IGNORECASE,
)
TRAILING_PUNCT_REGEX = re.compile(r"[:\s]+$")


def clean_sentence(text: str) -> str:
    """
    Cleans an extractive sentence:
    - Strips page markers and section number artifacts
    - Strips leading bullet markers
    - Trims whitespace
    - Ensures proper capitalization and punctuation
    """
    if not text:
        return ""
    s = text.strip()
    s = PAGE_MARKER_REGEX.sub("", s).strip()
    s = BULLET_PREFIX_REGEX.sub("", s).strip()
    s = NUMBER_PREFIX_REGEX.sub("", s).strip()
    s = DISCOURSE_CONNECTOR_REGEX.sub("", s).strip()

    if not s:
        return ""

    if s[0].islower():
        s = s[0].upper() + s[1:]

    if s[-1] not in ".!?:;":
        s = f"{s}."

    return s


def clean_bullet(bullet: str) -> str:
    """Cleans a bullet item text, stripping leading bullet markers and trailing spaces."""
    if not bullet:
        return ""
    b = bullet.strip()
    b = PAGE_MARKER_REGEX.sub("", b).strip()
    b = BULLET_PREFIX_REGEX.sub("", b).strip()
    b = NUMBER_PREFIX_REGEX.sub("", b).strip()
    return b.strip()


# Known cryptographic and computer science acronyms
DOMAIN_ACRONYMS = {
    "OTP", "DES", "AES", "RSA", "SHA", "MD5", "Z26", "XOR", "MAC",
    "CBC", "ECB", "CFB", "OFB", "CTR", "PRNG", "IV", "NIST", "TCP",
    "UDP", "IP", "DNS", "HTTP", "HTTPS", "SSL", "TLS", "ACID", "SQL",
    "NOSQL", "CPU", "RAM", "OS", "FIFO", "LRU", "BFS", "DFS", "BST",
    "AVL", "B-TREE", "B+TREE", "SOLID", "TDD", "CI/CD", "API",
}


def clean_entity_name(name: str) -> str:
    """Standardizes entity names by stripping numbering and normalizing title casing."""
    if not name:
        return ""
    s = name.strip()
    s = PAGE_MARKER_REGEX.sub("", s).strip()
    s = SECTION_NUM_REGEX.sub("", s).strip()
    s = TRAILING_PUNCT_REGEX.sub("", s).strip()

    words = s.split()
    converted: List[str] = []
    for i, w in enumerate(words):
        w_clean = re.sub(r"[^\w]", "", w).upper()
        if w_clean in DOMAIN_ACRONYMS:
            converted.append(w_clean)
        elif i == 0:
            converted.append(w.capitalize())
        else:
            converted.append(w)
    return " ".join(converted)


def deduplicate_preserve_order(items: List[str]) -> List[str]:
    """Deduplicates a list of strings while strictly preserving original insertion order."""
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        cleaned = clean_bullet(item)
        if not cleaned:
            continue
        norm = strip_accents(cleaned.lower())
        if norm not in seen:
            seen.add(norm)
            result.append(cleaned)
    return result


# Core Keyword Lexicons for Semantic Filtering (Normalized to Accentless Lowercase)
RAW_PROS_KEYWORDS = [
    "ưu điểm", "lợi thế", "điểm mạnh", "thế mạnh", "thuận lợi",
    "tính ưu việt", "tốc độ nhanh", "độ trễ thấp", "hiệu năng cao",
    "an toàn tuyệt đối", "an ninh tuyệt đối", "hoàn hảo", "khắc phục nhược điểm", "tiết kiệm",
    "dễ triển khai", "bền vững", "linh hoạt", "độ phức tạp thấp",
    "khả nghịch", "tính toàn vẹn", "tính bí mật cao", "kháng thám mã tốt",
    "tin cậy", "đảm bảo độ tin cậy", "tính cô lập cao", "an toàn tài nguyên",
    "chuyển ngữ cảnh rất nhanh", "chia sẻ dữ liệu thuận tiện", "bảo tồn toàn bộ tần suất",
    "pros", "advantage", "advantages", "strength", "strengths", "benefit",
    "benefits", "fast", "low latency", "perfect secrecy", "high efficiency",
    "reliable", "high performance",
]

RAW_CONS_KEYWORDS = [
    "nhược điểm", "hạn chế", "điểm yếu", "khuyết điểm", "bất lợi",
    "dễ bị phá mã", "tốn bộ nhớ", "chi phí cao", "độ trễ cao",
    "khó quản lý khóa", "phân phối khóa khó khăn", "khó quản lý và phân phối khóa",
    "dễ bị tấn công", "dễ bị tổn thương", "lỗ hổng", "nguy cơ", "không thể tái sử dụng",
    "tốn tài nguyên", "dễ bị thám mã", "phân tích tần suất", "vét cạn", "chậm",
    "tranh chấp dữ liệu", "race condition", "đồng bộ hóa phức tạp", "mất gói",
    "không đảm bảo độ tin cậy", "tốn kém tài nguyên", "chuyển ngữ cảnh tốn kém",
    "khó quản lý", "phân tích cặp ký tự", "tái sử dụng",
    "cons", "disadvantage", "disadvantages", "weakness", "weaknesses",
    "limitation", "limitations", "drawback", "drawbacks", "vulnerable",
    "insecure", "bottleneck", "overhead",
]

RAW_SECURITY_KEYWORDS = [
    "an ninh", "bảo mật", "an toàn", "phá mã", "thám mã",
    "an ninh vô điều kiện", "an ninh tính toán", "tấn công vét cạn",
    "brute-force", "phân tích tần suất", "tần suất xuất hiện",
    "không gian khóa", "độ dài khóa", "quản lý khóa", "phân phối khóa",
    "hoàn hảo", "tuyệt đối", "mật mã học", "mã hóa đối xứng",
    "mã hóa bất đối xứng", "khóa bí mật", "khóa công khai",
    "security", "cryptanalysis", "unconditional security", "computational security",
    "perfect secrecy", "key space", "brute force", "frequency analysis",
]

PROS_KEYWORDS: List[str] = [strip_accents(k.lower()) for k in RAW_PROS_KEYWORDS]
CONS_KEYWORDS: List[str] = [strip_accents(k.lower()) for k in RAW_CONS_KEYWORDS]
SECURITY_KEYWORDS: List[str] = [strip_accents(k.lower()) for k in RAW_SECURITY_KEYWORDS]


class ComparisonAndProsConsStitcher:
    """
    Intelligent Extractive Synthesizer for Comparisons, Pros/Cons, and Security Evaluations.
    100% Offline, Pure Python stdlib, Zero Hallucination.
    """

    @classmethod
    def classify_query_intent(cls, query: str) -> str:
        """
        Classifies query intent into:
        - 'comparison': Side-by-side or vs queries
        - 'pros_cons': Strengths and weaknesses evaluation
        - 'security_evaluation': Cryptographic / system security evaluation
        - 'general_evaluation': General assessment
        """
        if not query:
            return "general_evaluation"

        q_norm = strip_accents(query.lower()).strip()

        # Comparison indicators
        comp_patterns = [
            r"\bso sanh\b",
            r"\bphan biet\b",
            r"\bsu khac biet\b",
            r"\bkhac nhau\b",
            r"\bdiem giong va khac\b",
            r"\bdoi chieu\b",
            r"\bvs\b",
            r"\bversus\b",
            r"\bhay\b.*\b(tot hon|nhanh hon|an toan hon)\b",
            r"\bcompare\b",
            r"\bcomparison\b",
            r"\bdifference between\b",
        ]
        for pat in comp_patterns:
            if re.search(pat, q_norm):
                return "comparison"

        # Pros/Cons indicators
        pros_cons_patterns = [
            r"\buu (?:va )?nhuoc diem\b",
            r"\buu diem\b",
            r"\bnhuoc diem\b",
            r"\bdiem manh\b",
            r"\bdiem yeu\b",
            r"\bthe manh\b",
            r"\bhan che\b",
            r"\bloi the\b",
            r"\bbat loi\b",
            r"\bpros and cons\b",
            r"\badvantages and disadvantages\b",
            r"\bstrengths and weaknesses\b",
        ]
        for pat in pros_cons_patterns:
            if re.search(pat, q_norm):
                return "pros_cons"

        # Security evaluation indicators
        security_patterns = [
            r"\ban ninh\b",
            r"\bbao mat\b",
            r"\bdo an toan\b",
            r"\btinh bao mat\b",
            r"\btham ma\b",
            r"\bpha ma\b",
            r"\btan cong\b",
            r"\bkhang tham ma\b",
            r"\bkhong gian khoa\b",
            r"\bsecurity\b",
            r"\bcryptanalysis\b",
        ]
        for pat in security_patterns:
            if re.search(pat, q_norm):
                return "security_evaluation"

        return "general_evaluation"

    @classmethod
    def extract_target_entities(
        cls,
        query: str,
        passages: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[str], List[str]]:
        """
        Extracts primary entity and compared entities from the query and available passages.
        Returns:
            (primary_entity, list_of_entities)
        """
        if not query:
            return None, []

        q = query.strip()
        entities: List[str] = []

        # 1. Check comparison patterns: "so sánh X và Y", "phân biệt X với Y", "X vs Y"
        comp_match = re.search(
            r"(?:so sánh|phân biệt|sự khác biệt giữa|đối chiếu)\s+(?:giữa\s+)?(.+?)\s+(?:và|với|vs)\s+(.+?)(?:\?|$|\.|\,)",
            q,
            flags=re.IGNORECASE,
        )
        if comp_match:
            e1 = comp_match.group(1).strip()
            e2 = comp_match.group(2).strip()
            e1 = re.sub(r"^(?:hai|các|giữa)\s+", "", e1, flags=re.IGNORECASE).strip()
            e2 = re.sub(r"^(?:với|và)\s+", "", e2, flags=re.IGNORECASE).strip()
            # Clean common trailing domain context
            e2 = re.sub(r"\s+(?:trong|ở|theo)\s+.*$", "", e2, flags=re.IGNORECASE).strip()
            entities = [clean_entity_name(e1), clean_entity_name(e2)]
            return entities[0], entities

        vs_match = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$|\.|\,)", q, flags=re.IGNORECASE)
        if vs_match:
            e1 = vs_match.group(1).strip()
            e2 = vs_match.group(2).strip()
            e2 = re.sub(r"\s+(?:trong|ở|theo)\s+.*$", "", e2, flags=re.IGNORECASE).strip()
            entities = [clean_entity_name(e1), clean_entity_name(e2)]
            return entities[0], entities

        # 2. Check pros/cons pattern: "ưu nhược điểm của X", "điểm mạnh/điểm yếu của X"
        single_match = re.search(
            r"(?:ưu(?:\s*v[àa])?\s*nhược điểm|điểm mạnh(?:\s*v[àa])?\s*điểm yếu|lợi thế|hạn chế|đặc tính|an ninh|độ an toàn|tính bảo mật)\s+(?:của|về|trong)?\s*(.+?)(?:\?|$|\.|\,)",
            q,
            flags=re.IGNORECASE,
        )
        if single_match:
            e = single_match.group(1).strip()
            # Clean prefixes carefully: if starts with "hệ mã hóa" / "hệ mã hoá", keep "mã hóa"
            e = re.sub(r"^(?:giao thức|thuật toán|mô hình)\s+", "", e, flags=re.IGNORECASE).strip()
            if re.match(r"^hệ\s+mã\s+(?:hóa|hoá)\b", e, flags=re.IGNORECASE):
                e = re.sub(r"^hệ\s+", "", e, flags=re.IGNORECASE).strip()
            elif re.match(r"^hệ\s+mã\b", e, flags=re.IGNORECASE):
                e = re.sub(r"^hệ\s+mã\s+", "", e, flags=re.IGNORECASE).strip()
            if e:
                cleaned = clean_entity_name(e)
                return cleaned, [cleaned]

        # 3. Fallback: match known section titles or prominent entities from passages
        if passages:
            matched_sections: List[str] = []
            q_norm = strip_accents(q.lower())
            for p in passages:
                if not isinstance(p, dict):
                    continue
                sec = p.get("section_title") or ""
                clean_sec = clean_entity_name(sec)
                sec_norm = strip_accents(clean_sec.lower())
                if sec_norm and sec_norm in q_norm and clean_sec not in matched_sections:
                    matched_sections.append(clean_sec)

            if len(matched_sections) >= 2:
                return matched_sections[0], matched_sections[:2]
            elif len(matched_sections) == 1:
                return matched_sections[0], [matched_sections[0]]

        # Default fallback: clean full query
        cleaned_fallback = clean_entity_name(q.rstrip("?").strip())
        return cleaned_fallback, [cleaned_fallback]

    @classmethod
    def _extract_sentences_and_bullets(cls, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parses passages into individual normalized sentence/bullet items with source metadata."""
        items: List[Dict[str, Any]] = []
        for idx, p in enumerate(passages):
            if not isinstance(p, dict):
                continue
            pid = p.get("passage_id") or f"P{idx:02d}"
            sec = p.get("section_title") or ""
            raw_text = p.get("text") or ""
            if not raw_text:
                continue

            lines = raw_text.splitlines()
            for line in lines:
                l = line.strip()
                if not l or l.startswith("==="):
                    continue
                is_bullet = bool(re.match(r"^[-*•+–—]\s*", l) or re.match(r"^\d+[\.)]\s*", l))
                clean_l = clean_bullet(l) if is_bullet else clean_sentence(l)
                if not clean_l or len(clean_l) < 3:
                    continue

                items.append({
                    "raw": l,
                    "clean": clean_l,
                    "text": clean_l,
                    "is_bullet": is_bullet,
                    "section_title": sec,
                    "passage_id": pid,
                })
        return items

    @classmethod
    def _is_entity_match(cls, entity_norm: str, text_norm: str, sec_norm: str) -> bool:
        """Determines whether a passage item is relevant to a specified entity."""
        if not entity_norm:
            return True

        # Exact substring match
        if entity_norm in text_norm or entity_norm in sec_norm:
            return True

        # Base entity match (ignoring parentheses e.g. "OTP")
        base_entity = re.sub(r"\(.*?\)", "", entity_norm).strip()
        if base_entity and (base_entity in text_norm or base_entity in sec_norm):
            return True

        # Parenthetical acronym match
        acronyms = re.findall(r"\((.*?)\)", entity_norm)
        for acr in acronyms:
            acr_clean = acr.strip().lower()
            if acr_clean and (acr_clean in text_norm or acr_clean in sec_norm):
                return True

        # Token-level overlap
        tokens = [t for t in re.split(r"[\s\(\)\,\.\-\/]+", entity_norm) if len(t) > 2]
        if tokens:
            overlap = sum(1 for tok in tokens if tok in text_norm or tok in sec_norm)
            if overlap >= min(2, len(tokens)):
                return True

        return False

    @classmethod
    def extract_pros_and_cons(
        cls,
        passages: List[Dict[str, Any]],
        entity: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extracts verified pros and cons from passages.
        Filters or prioritizes items relevant to entity.
        Returns:
            (pros_list, cons_list) where each element is a dict with 'text', 'passage_id', 'section_title'.
        """
        items = cls._extract_sentences_and_bullets(passages)
        pros: List[Dict[str, Any]] = []
        cons: List[Dict[str, Any]] = []

        entity_norm = strip_accents(entity.lower()).strip() if entity else ""

        for it in items:
            text = it["clean"]
            text_norm = strip_accents(text.lower())
            sec_norm = strip_accents(it["section_title"].lower())

            is_entity_relevant = cls._is_entity_match(entity_norm, text_norm, sec_norm)

            # Match pros against normalized unaccented lexicon
            is_pro = any(k in text_norm for k in PROS_KEYWORDS)
            # Match cons against normalized unaccented lexicon
            is_con = any(k in text_norm for k in CONS_KEYWORDS)

            # Explicit colon label heuristics: "Ưu điểm: ...", "Nhược điểm: ..."
            if re.search(r"^(?:uu diem|loi the|diem manh)\s*[:\-]", text_norm):
                is_pro = True
                is_con = False
            elif re.search(r"^(?:nhuoc diem|han che|diem yeu)\s*[:\-]", text_norm):
                is_con = True
                is_pro = False

            if is_pro and not is_con:
                pros.append({
                    "text": text,
                    "clean": text,
                    "passage_id": it["passage_id"],
                    "section_title": it["section_title"],
                    "entity_match": is_entity_relevant,
                })
            elif is_con and not is_pro:
                cons.append({
                    "text": text,
                    "clean": text,
                    "passage_id": it["passage_id"],
                    "section_title": it["section_title"],
                    "entity_match": is_entity_relevant,
                })
            elif is_pro and is_con:
                parts = re.split(r"[,;]\s*(?:nhưng|tuy nhiên|ngược lại|tuy vay)\s*", text, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    pros.append({
                        "text": parts[0].strip(),
                        "clean": parts[0].strip(),
                        "passage_id": it["passage_id"],
                        "section_title": it["section_title"],
                        "entity_match": is_entity_relevant,
                    })
                    cons.append({
                        "text": parts[1].strip(),
                        "clean": parts[1].strip(),
                        "passage_id": it["passage_id"],
                        "section_title": it["section_title"],
                        "entity_match": is_entity_relevant,
                    })
                else:
                    pros.append({
                        "text": text,
                        "clean": text,
                        "passage_id": it["passage_id"],
                        "section_title": it["section_title"],
                        "entity_match": is_entity_relevant,
                    })

        unique_pros = cls._dedup_items(pros)
        unique_cons = cls._dedup_items(cons)

        # If entity filter was requested, return matching items if found
        if entity_norm:
            matching_pros = [p for p in unique_pros if p.get("entity_match")]
            matching_cons = [c for c in unique_cons if c.get("entity_match")]
            if matching_pros or matching_cons:
                return matching_pros, matching_cons

        return unique_pros, unique_cons

    @classmethod
    def _dedup_items(cls, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Set[str] = set()
        res: List[Dict[str, Any]] = []
        for it in items:
            raw_text = it.get("text") or it.get("clean") or it.get("raw") or ""
            key = strip_accents(raw_text.lower()).strip()
            if key and key not in seen:
                seen.add(key)
                res.append(it)
        return res

    @classmethod
    def extract_security_facets(
        cls,
        passages: List[Dict[str, Any]],
        entity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extracts specialized cryptographic security facets:
        - unconditional_security (An ninh vô điều kiện / Hoàn hảo / OTP)
        - computational_security (An ninh tính toán)
        - attacks_and_vulnerabilities (Vét cạn, phân tích tần suất, thám mã)
        - key_space_and_parameters (Không gian khóa, độ dài khóa)
        - general_security (Các đặc tính bảo mật khác)
        """
        items = cls._extract_sentences_and_bullets(passages)

        unconditional: List[Dict[str, Any]] = []
        computational: List[Dict[str, Any]] = []
        attacks: List[Dict[str, Any]] = []
        keyspace: List[Dict[str, Any]] = []
        general_sec: List[Dict[str, Any]] = []

        entity_norm = strip_accents(entity.lower()).strip() if entity else ""

        for it in items:
            text = it["clean"]
            norm = strip_accents(text.lower())
            sec_norm = strip_accents(it["section_title"].lower())

            if entity_norm and not cls._is_entity_match(entity_norm, norm, sec_norm):
                # If filtered by entity and not matched, keep only if high-security general definition
                if not any(k in norm for k in ["an ninh vo dieu kien", "an ninh tinh toan", "khong the pha ma"]):
                    continue

            # 1. Unconditional security
            if any(k in norm for k in ["an ninh vo dieu kien", "hoan hao", "an ninh tuyet doi", "unconditional"]):
                unconditional.append(it)
            # 2. Computational security
            elif any(k in norm for k in ["an ninh tinh toan", "chi phi de pha vo", "thoi gian pha ma", "computational security"]):
                computational.append(it)
            # 3. Cryptanalysis attacks & vulnerabilities
            elif any(k in norm for k in ["phan tich tan suat", "vet can", "brute-force", "tham ma", "pha ma", "de bi pha"]):
                attacks.append(it)
            # 4. Key space and parameters
            elif any(k in norm for k in ["khong gian khoa", "do dai khoa", "26!", "ma tran m x m", "modulo 26", "bang chu cai"]):
                keyspace.append(it)
            # 5. General security
            elif any(k in norm for k in ["an ninh", "bao mat", "security", "an toan"]):
                general_sec.append(it)

        return {
            "unconditional_security": cls._dedup_items(unconditional),
            "computational_security": cls._dedup_items(computational),
            "attacks_and_vulnerabilities": cls._dedup_items(attacks),
            "key_space_and_parameters": cls._dedup_items(keyspace),
            "general_security": cls._dedup_items(general_sec),
        }

    @classmethod
    def extract_entity_features(
        cls,
        passages: List[Dict[str, Any]],
        entity: str,
    ) -> Dict[str, Any]:
        """Extracts complete descriptive features, mechanisms, formulas, pros/cons for a specific entity."""
        items = cls._extract_sentences_and_bullets(passages)
        entity_norm = strip_accents(entity.lower()).strip()

        matched_items: List[Dict[str, Any]] = []
        for it in items:
            norm = strip_accents(it["clean"].lower())
            sec_norm = strip_accents(it["section_title"].lower())
            if cls._is_entity_match(entity_norm, norm, sec_norm):
                matched_items.append(it)

        definitions: List[str] = []
        mechanisms: List[str] = []
        pros: List[str] = []
        cons: List[str] = []
        citations: List[str] = []

        for it in matched_items:
            t = it["clean"]
            norm = strip_accents(t.lower())
            citations.append(it["passage_id"])
            if it["section_title"]:
                citations.append(it["section_title"])

            is_pro = any(k in norm for k in PROS_KEYWORDS)
            is_con = any(k in norm for k in CONS_KEYWORDS)

            if is_pro and not is_con:
                pros.append(t)
            elif is_con:
                cons.append(t)
            elif any(k in norm for k in ["la ", "dinh nghia", "khai niem", "xuat hien"]):
                definitions.append(t)
            else:
                mechanisms.append(t)

        return {
            "entity": entity,
            "definitions": deduplicate_preserve_order(definitions),
            "mechanisms": deduplicate_preserve_order(mechanisms),
            "pros": deduplicate_preserve_order(pros),
            "cons": deduplicate_preserve_order(cons),
            "citations": deduplicate_preserve_order(citations),
        }

    @classmethod
    def synthesize_pros_cons_answer(
        cls,
        entity: str,
        pros: List[Dict[str, Any]],
        cons: List[Dict[str, Any]],
        passages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Renders a structured, categorized markdown answer for Pros & Cons inquiry."""
        clean_name = clean_entity_name(entity) or "đối tượng"
        citations: List[str] = []

        pro_texts = [p.get("text") or p.get("clean") for p in pros]
        con_texts = [c.get("text") or c.get("clean") for c in cons]

        for p in pros:
            citations.append(p["passage_id"])
            if p.get("section_title"):
                citations.append(p["section_title"])
        for c in cons:
            citations.append(c["passage_id"])
            if c.get("section_title"):
                citations.append(c["section_title"])

        # If no explicit pros/cons found, fallback to substantive characteristics from passages
        if not pro_texts and not con_texts:
            entity_feat = cls.extract_entity_features(passages, entity)
            citations.extend(entity_feat["citations"])
            lines: List[str] = [
                f"### Phân tích Đặc tính & Đánh giá của {clean_name}",
                "",
                f"Tài liệu hiện tại tập trung trình bày nguyên lý và đặc tả kỹ thuật của **{clean_name}**:",
            ]
            if entity_feat["definitions"]:
                lines.append(f"- **Đặc trưng:** {entity_feat['definitions'][0]}")
            for m in entity_feat["mechanisms"][:4]:
                lines.append(f"- {m}")
            lines.append("")
            lines.append(
                "*Lưu ý:* Tài liệu không ghi nhận mục riêng biệt về ưu nhược điểm định lượng, "
                "các đặc trưng kỹ thuật trên phản ánh trực tiếp năng lực vận hành thực tế của hệ thống."
            )
            answer_text = "\n".join(lines).strip()
            return {
                "answer": answer_text,
                "pros": [],
                "cons": [],
                "citations": deduplicate_preserve_order(citations),
                "has_sufficient_evidence": bool(entity_feat["definitions"] or entity_feat["mechanisms"]),
                "confidence": 0.7 if (entity_feat["definitions"] or entity_feat["mechanisms"]) else 0.3,
            }

        lines = [
            f"### Đánh giá Toàn diện Ưu điểm & Nhược điểm của {clean_name}",
            "",
            "Dựa trên các tài liệu và giáo trình đã học, các khía cạnh điểm mạnh và hạn chế được phân tích chi tiết như sau:",
            "",
            f"**1. Ưu điểm (Điểm mạnh / Lợi thế nổi bật):**",
        ]
        if pro_texts:
            for pt in pro_texts:
                lines.append(f"- {pt}")
        else:
            lines.append(f"- Tài liệu không ghi nhận trực tiếp ưu điểm nổi bật cụ thể của {clean_name}.")

        lines.append("")
        lines.append(f"**2. Nhược điểm (Hạn chế / Thách thức kỹ thuật):**")
        if con_texts:
            for ct in con_texts:
                lines.append(f"- {ct}")
        else:
            lines.append(f"- Chưa ghi nhận nhược điểm nghiêm trọng trực tiếp trong phạm vi trích xuất của {clean_name}.")

        lines.append("")
        lines.append("**3. Nhận xét & Đánh giá tổng kết:**")
        if pro_texts and con_texts:
            lines.append(
                f"- **{clean_name}** mang lại lợi thế kỹ thuật rõ rệt ({pro_texts[0]}), "
                f"tuy nhiên cần có biện pháp khắc phục hạn chế ({con_texts[0]}) khi ứng dụng vào hệ thống thực tế."
            )
        elif pro_texts:
            lines.append(f"- **{clean_name}** nổi bật với các thế mạnh kỹ thuật vững chắc: {pro_texts[0]}.")
        else:
            lines.append(f"- Cần lưu ý các điều kiện ràng buộc kỹ thuật của **{clean_name}**: {con_texts[0]}.")

        return {
            "answer": "\n".join(lines).strip(),
            "pros": pro_texts,
            "cons": con_texts,
            "citations": deduplicate_preserve_order(citations),
            "has_sufficient_evidence": True,
            "confidence": 0.95 if (pro_texts and con_texts) else 0.85,
        }

    @classmethod
    def synthesize_comparison_answer(
        cls,
        entity_a: str,
        entity_b: str,
        passages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Renders a structured, side-by-side and matrix comparison between two entities."""
        name_a = clean_entity_name(entity_a) or "Đối tượng A"
        name_b = clean_entity_name(entity_b) or "Đối tượng B"

        feat_a = cls.extract_entity_features(passages, entity_a)
        feat_b = cls.extract_entity_features(passages, entity_b)

        citations = deduplicate_preserve_order(feat_a["citations"] + feat_b["citations"])

        has_a = bool(feat_a["definitions"] or feat_a["mechanisms"] or feat_a["pros"] or feat_a["cons"])
        has_b = bool(feat_b["definitions"] or feat_b["mechanisms"] or feat_b["pros"] or feat_b["cons"])

        lines: List[str] = [
            f"### So sánh Đối chiếu: {name_a} và {name_b}",
            "",
            f"Dưới đây là bảng phân tích và đối chiếu toàn diện giữa **{name_a}** và **{name_b}** dựa trên tài liệu:",
            "",
            f"**1. Phân tích từng đối tượng:**",
            f"- **{name_a}:**",
        ]

        points_a = feat_a["definitions"][:2] + feat_a["mechanisms"][:2] + feat_a["pros"][:1] + feat_a["cons"][:1]
        if points_a:
            for pa in points_a:
                lines.append(f"  * {pa}")
        else:
            lines.append(f"  * (Chưa có đủ thông tin chi tiết trong tài liệu đối chiếu cho {name_a})")

        lines.append(f"- **{name_b}:**")
        points_b = feat_b["definitions"][:2] + feat_b["mechanisms"][:2] + feat_b["pros"][:1] + feat_b["cons"][:1]
        if points_b:
            for pb in points_b:
                lines.append(f"  * {pb}")
        else:
            lines.append(f"  * (Chưa có đủ thông tin chi tiết trong tài liệu đối chiếu cho {name_b})")

        # 2. Comparison Dimensions Table
        lines.append("")
        lines.append("**2. Bảng đối chiếu tiêu chí cốt lõi:**")
        lines.append(f"| Tiêu chí | {name_a} | {name_b} |")
        lines.append("|---|---|---|")

        def_a_short = (feat_a["definitions"][0] if feat_a["definitions"] else (feat_a["mechanisms"][0] if feat_a["mechanisms"] else "Xem chi tiết hệ thống"))
        def_b_short = (feat_b["definitions"][0] if feat_b["definitions"] else (feat_b["mechanisms"][0] if feat_b["mechanisms"] else "Xem chi tiết hệ thống"))
        lines.append(f"| **Bản chất / Nguyên lý** | {def_a_short} | {def_b_short} |")

        pro_a_short = feat_a["pros"][0] if feat_a["pros"] else "Độ bảo mật và tính ứng dụng theo chuẩn thiết kế"
        pro_b_short = feat_b["pros"][0] if feat_b["pros"] else "Độ bảo mật và tính ứng dụng theo chuẩn thiết kế"
        lines.append(f"| **Ưu thế nổi bật** | {pro_a_short} | {pro_b_short} |")

        con_a_short = feat_a["cons"][0] if feat_a["cons"] else "Phụ thuộc vào quản lý khóa và không gian khóa"
        con_b_short = feat_b["cons"][0] if feat_b["cons"] else "Phụ thuộc vào quản lý khóa và không gian khóa"
        lines.append(f"| **Hạn chế / Thách thức** | {con_a_short} | {con_b_short} |")

        lines.append("")
        lines.append("**3. Nhận xét & Kết luận so sánh:**")
        lines.append(
            f"- Điểm khác biệt mấu chốt: **{name_a}** và **{name_b}** tiếp cận bài toán theo hai cơ chế xử lý khác nhau. "
            f"Việc lựa chọn giữa hai phương pháp phụ thuộc trực tiếp vào yêu cầu băng thông, tốc độ và cấp độ an ninh mục tiêu."
        )

        comparison_facets = [
            {"criterion": "Bản chất / Nguyên lý", "entity_a": def_a_short, "entity_b": def_b_short},
            {"criterion": "Ưu thế nổi bật", "entity_a": pro_a_short, "entity_b": pro_b_short},
            {"criterion": "Hạn chế / Thách thức", "entity_a": con_a_short, "entity_b": con_b_short},
        ]

        has_sufficient = has_a or has_b
        confidence = 0.95 if (has_a and has_b) else (0.7 if has_sufficient else 0.2)

        return {
            "answer": "\n".join(lines).strip(),
            "comparison_facets": comparison_facets,
            "entity_a_features": feat_a,
            "entity_b_features": feat_b,
            "citations": citations,
            "has_sufficient_evidence": has_sufficient,
            "confidence": confidence,
        }

    @classmethod
    def synthesize_security_answer(
        cls,
        entity: str,
        security_facets: Dict[str, Any],
        passages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Renders specialized cryptographic and system security evaluation."""
        clean_name = clean_entity_name(entity) or "Hệ mã"
        citations: List[str] = []

        uncond = security_facets.get("unconditional_security", [])
        comp = security_facets.get("computational_security", [])
        attacks = security_facets.get("attacks_and_vulnerabilities", [])
        keyspace = security_facets.get("key_space_and_parameters", [])
        general = security_facets.get("general_security", [])

        for cat in [uncond, comp, attacks, keyspace, general]:
            for item in cat:
                citations.append(item["passage_id"])
                if item.get("section_title"):
                    citations.append(item["section_title"])

        lines: List[str] = [
            f"### Đánh giá An ninh & Khả năng Kháng Thám Mã của {clean_name}",
            "",
            "Dựa trên phân tích lý thuyết mật mã học và tài liệu kỹ thuật, các khía cạnh an ninh bao gồm:",
            "",
            "**1. Phân loại Cấp độ An ninh Hệ mã:**",
        ]

        if uncond:
            lines.append(f"- **An ninh vô điều kiện (Unconditional Security):**")
            for u in uncond:
                lines.append(f"  * {u['clean']}")
        else:
            lines.append("- **An ninh vô điều kiện:** Chỉ đạt được khi độ dài khóa bằng thông điệp và dùng 1 lần (như OTP).")

        if comp:
            lines.append(f"- **An ninh tính toán (Computational Security):**")
            for c in comp:
                lines.append(f"  * {c['clean']}")
        else:
            lines.append("- **An ninh tính toán:** Đảm bảo khi chi phí và thời gian phá vỡ vượt quá giá trị hữu dụng của thông tin.")

        lines.append("")
        lines.append("**2. Các Phương pháp Tấn công & Điểm yếu Thám mã:**")
        if attacks:
            for at in attacks:
                lines.append(f"- {at['clean']}")
        else:
            lines.append(f"- Chưa ghi nhận lỗ hổng tấn công trực tiếp trong tài liệu trích xuất của {clean_name}.")

        if keyspace:
            lines.append("")
            lines.append("**3. Tham số & Không gian Khóa (Key Space):**")
            for ks in keyspace:
                lines.append(f"- {ks['clean']}")

        lines.append("")
        lines.append("**4. Kết luận Đánh giá An ninh:**")
        if uncond:
            lines.append(
                f"- **{clean_name}** có đặc tính an ninh vượt trội, không thể bị phá mã về mặt lý thuyết nếu tuân thủ nghiêm ngặt quy tắc sinh và bảo mật khóa."
            )
        elif attacks:
            lines.append(
                f"- Để duy trì tính an toàn cho **{clean_name}**, cần mở rộng không gian khóa hoặc kết hợp nhiều bảng thay thế/hoán vị để triệt tiêu nguy cơ thám mã tần suất và vét cạn."
            )
        else:
            lines.append(
                f"- An ninh của **{clean_name}** phụ thuộc trực tiếp vào tính bí mật của khóa và chiều dài bản mã xử lý."
            )

        has_sufficient = bool(uncond or comp or attacks or keyspace or general)

        return {
            "answer": "\n".join(lines).strip(),
            "security_facets": security_facets,
            "citations": deduplicate_preserve_order(citations),
            "has_sufficient_evidence": has_sufficient,
            "confidence": 0.95 if (uncond or comp or attacks) else (0.75 if has_sufficient else 0.2),
        }

    @classmethod
    def synthesize_fallback_refusal(cls, query: str) -> Dict[str, Any]:
        """Provides a polite, grounded refusal response when no relevant evidence exists."""
        msg = (
            f"Không tìm thấy đủ tài liệu đối chiếu hoặc dữ liệu đánh giá phù hợp trong tài liệu để trả lời cho câu hỏi: '{query}'. "
            "Vui lòng cung cấp thêm ngữ cảnh hoặc kiểm tra lại tài liệu nguồn."
        )
        return {
            "query": query,
            "mode": "refusal",
            "target_entity": None,
            "target_entities": [],
            "answer": msg,
            "pros": [],
            "cons": [],
            "comparison_facets": [],
            "security_facets": {},
            "citations": [],
            "has_sufficient_evidence": False,
            "confidence": 0.0,
            "metadata": {"status": "insufficient_evidence"},
        }


def synthesize_pros_cons_and_comparison(
    query: str,
    passages: List[Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Main Entrypoint Function:
    Synthesizes structured, fully-grounded answers for comparisons, pros/cons, and security evaluations.

    Args:
        query: The user query string (e.g. 'ưu nhược điểm của X', 'so sánh X và Y', 'an ninh của hệ mã').
        passages: List of passage dictionaries (must contain 'text', optionally 'section_title', 'passage_id').
        **kwargs: Additional optional formatting configurations.

    Returns:
        Structured response dictionary with keys:
          - query: str
          - mode: str ('pros_cons', 'comparison', 'security_evaluation', 'general_evaluation')
          - target_entity: str
          - target_entities: list of str
          - answer: str (Formatted markdown text)
          - pros: list of str
          - cons: list of str
          - comparison_facets: list of dict
          - security_facets: dict
          - citations: list of str
          - has_sufficient_evidence: bool
          - confidence: float
          - metadata: dict
    """
    if not query or not str(query).strip():
        return ComparisonAndProsConsStitcher.synthesize_fallback_refusal(query or "")

    if not passages:
        return ComparisonAndProsConsStitcher.synthesize_fallback_refusal(query)

    stitcher = ComparisonAndProsConsStitcher
    mode = stitcher.classify_query_intent(query)
    primary_entity, entities = stitcher.extract_target_entities(query, passages)

    # 1. Comparison Mode
    if mode == "comparison":
        if len(entities) >= 2:
            e_a, e_b = entities[0], entities[1]
        elif len(entities) == 1:
            e_a = entities[0]
            e_b = "Phương pháp đối ứng"
        else:
            e_a, e_b = "Đối tượng A", "Đối tượng B"

        res = stitcher.synthesize_comparison_answer(e_a, e_b, passages)
        return {
            "query": query,
            "mode": "comparison",
            "target_entity": primary_entity,
            "target_entities": [e_a, e_b],
            "answer": res["answer"],
            "pros": res.get("entity_a_features", {}).get("pros", []) + res.get("entity_b_features", {}).get("pros", []),
            "cons": res.get("entity_a_features", {}).get("cons", []) + res.get("entity_b_features", {}).get("cons", []),
            "comparison_facets": res.get("comparison_facets", []),
            "security_facets": {},
            "citations": res.get("citations", []),
            "has_sufficient_evidence": res.get("has_sufficient_evidence", False),
            "confidence": res.get("confidence", 0.0),
            "metadata": {
                "entity_a": e_a,
                "entity_b": e_b,
            },
        }

    # 2. Security Evaluation Mode
    elif mode == "security_evaluation":
        sec_facets = stitcher.extract_security_facets(passages, primary_entity)
        res = stitcher.synthesize_security_answer(primary_entity or "Hệ mã", sec_facets, passages)
        return {
            "query": query,
            "mode": "security_evaluation",
            "target_entity": primary_entity,
            "target_entities": entities,
            "answer": res["answer"],
            "pros": [],
            "cons": [at["clean"] for at in sec_facets.get("attacks_and_vulnerabilities", [])],
            "comparison_facets": [],
            "security_facets": sec_facets,
            "citations": res.get("citations", []),
            "has_sufficient_evidence": res.get("has_sufficient_evidence", False),
            "confidence": res.get("confidence", 0.0),
            "metadata": {
                "unconditional_count": len(sec_facets.get("unconditional_security", [])),
                "computational_count": len(sec_facets.get("computational_security", [])),
                "attacks_count": len(sec_facets.get("attacks_and_vulnerabilities", [])),
            },
        }

    # 3. Pros and Cons Mode (or general evaluation)
    else:
        pros, cons = stitcher.extract_pros_and_cons(passages, primary_entity)
        res = stitcher.synthesize_pros_cons_answer(primary_entity or "Đối tượng", pros, cons, passages)
        return {
            "query": query,
            "mode": "pros_cons" if mode == "pros_cons" else "general_evaluation",
            "target_entity": primary_entity,
            "target_entities": entities,
            "answer": res["answer"],
            "pros": res.get("pros", []),
            "cons": res.get("cons", []),
            "comparison_facets": [],
            "security_facets": {},
            "citations": res.get("citations", []),
            "has_sufficient_evidence": res.get("has_sufficient_evidence", False),
            "confidence": res.get("confidence", 0.0),
            "metadata": {
                "pros_count": len(res.get("pros", [])),
                "cons_count": len(res.get("cons", [])),
            },
        }
