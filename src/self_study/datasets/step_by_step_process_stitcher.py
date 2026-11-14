"""
Step-by-Step Process & Algorithm Stitcher for Sovereign Brain v5.5.
Pure Python stdlib, 100% Strict Offline, Zero External API, Sub-10ms Execution.

Specializes in synthesizing coherent, logically ordered step-by-step answers for:
- 'các bước thực hiện' (Procedural Steps)
- 'quy trình mã hóa và giải mã' (Cryptographic Workflow: Key Gen -> Encryption -> Decryption)
- 'cách hoạt động của thuật toán' (Algorithm Lifecycle & Execution Mechanism)

Transforms fragmented extractive passages into structured, grounded responses formatted
as Bước 1, Bước 2, Bước 3... with clear phase separation and zero hallucination.
"""

from __future__ import annotations

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


def clean_text_line(text: str) -> str:
    """
    Cleans a single line or span:
    - Strips page markers like '=== Page \\d+ ==='
    - Normalizes internal whitespaces
    - Removes dangling bullet dashes at line starts
    """
    if not text:
        return ""
    s = text.strip()
    s = re.sub(r"===\s*Page\s*\d+\s*===", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^[•\-*+–—]\s*", "", s).strip()
    return s


# ------------------------------------------------------------------------------
# STEP & PHASE REGEX PATTERNS
# ------------------------------------------------------------------------------

# Detects explicit step numbering: "Bước 1: ...", "Step 2. ...", "Giai đoạn 3 - ..."
STEP_PREFIX_PATTERN = re.compile(
    r"^(?:bước|buoc|step|giai đoạn|giai doan|pha|phase|thao tác|thao tac)\s*"
    r"(\d+|[ivxlcdm]+)"
    r"[\s:.\-–—]+(?:\(([^\)]+)\)[\s:.\-–—]*)?(.*)$",
    re.IGNORECASE,
)

# Numbered bullet list: "1. ...", "2) ...", "(3) ..."
NUMBERED_LIST_PATTERN = re.compile(
    r"^(?:\(?(\d+)\)?[\.\)]|\[(\d+)\])\s+(.*)$",
    re.IGNORECASE,
)

# Lettered bullet list: "a. ...", "b) ...", "(c) ..."
LETTERED_LIST_PATTERN = re.compile(
    r"^(?:\(?([a-z])\)?[\.\)]|\[([a-z])\])\s+(.*)$",
    re.IGNORECASE,
)

# Roman numerals list: "I. ...", "ii. ..."
ROMAN_LIST_PATTERN = re.compile(
    r"^(?:\(?([ivxlcdm]+)\)?[\.\)]|\[([ivxlcdm]+)\])\s+(.*)$",
    re.IGNORECASE,
)

# Sequential transition markers indicating steps
TRANSITION_MARKERS = [
    (re.compile(r"^(?:trước hết|đầu tiên|ban đầu|khởi đầu|bước đầu tiên|firstly|first)\s*[:,\-–—]?\s*(.*)$", re.IGNORECASE), 1, "Khởi tạo"),
    (re.compile(r"^(?:thứ nhất|thứ 1|1 là)\s*[:,\-–—]?\s*(.*)$", re.IGNORECASE), 1, "Thứ nhất"),
    (re.compile(r"^(?:thứ hai|thứ 2|2 là)\s*[:,\-–—]?\s*(.*)$", re.IGNORECASE), 2, "Thứ hai"),
    (re.compile(r"^(?:thứ ba|thứ 3|3 là)\s*[:,\-–—]?\s*(.*)$", re.IGNORECASE), 3, "Thứ ba"),
    (re.compile(r"^(?:thứ tư|thứ 4|4 là)\s*[:,\-–—]?\s*(.*)$", re.IGNORECASE), 4, "Thứ tư"),
    (re.compile(r"^(?:thứ năm|thứ 5|5 là)\s*[:,\-–—]?\s*(.*)$", re.IGNORECASE), 5, "Thứ năm"),
    (re.compile(r"^(?:tiếp theo|kế tiếp|bước tiếp theo|sau đó|then|next)\s*[:,\-–—]?\s*(.*)$", re.IGNORECASE), 0, "Tiếp theo"),
    (re.compile(r"^(?:cuối cùng|sau cùng|bước cuối|finally|lastly)\s*[:,\-–—]?\s*(.*)$", re.IGNORECASE), 999, "Kết thúc"),
]

# Phase / Sub-process classification headers for cryptographic and multi-stage processes
PHASE_HEADER_PATTERNS = {
    "key_generation": [
        r"^(?:#+\s*)?(?:sinh khóa|tạo khóa|khởi tạo khóa|thiết lập khóa|key generation|key setup)(?:\s*\([^\)]*\))?[:\s]*$",
        r"^(?:#+\s*)?(?:quy trình sinh khóa|quá trình sinh khóa|các bước sinh khóa)[:\s]*$",
    ],
    "encryption": [
        r"^(?:#+\s*)?(?:quy trình mã hóa|quá trình mã hóa|các bước mã hóa|mã hóa bản rõ|phép mã hóa|encryption process|encryption)[:\s]*$",
        r"^(?:#+\s*)?(?:hàm mã hóa|công thức mã hóa|mã hóa)[:\s]*$",
    ],
    "decryption": [
        r"^(?:#+\s*)?(?:quy trình giải mã|quá trình giải mã|các bước giải mã|giải mã bản mã|phép giải mã|decryption process|decryption)[:\s]*$",
        r"^(?:#+\s*)?(?:hàm giải mã|công thức giải mã|giải mã)[:\s]*$",
    ],
}

# Action verbs in Vietnamese technical text that indicate procedural steps
ACTION_VERB_PREFIXES = [
    "chọn", "tính", "gán", "duyệt", "lặp", "gửi", "nhận", "chia", "so sánh",
    "cập nhật", "chuyển", "chuyển đổi", "thay thế", "hoán vị", "áp dụng",
    "xác định", "tìm", "kiểm tra", "đọc", "ghi", "mở rộng", "kết hợp",
    "thực hiện", "phân hoạch", "sắp xếp", "khởi tạo", "trả về", "lưu trữ",
]


def extract_topic_from_query(query: str) -> str:
    """
    Extracts the core topic or subject name from a procedural query.
    e.g. 'Các bước thực hiện thuật toán Dijkstra' -> 'Thuật toán Dijkstra'
         'Quy trình mã hóa và giải mã RSA' -> 'Hệ mã RSA'
         'Cách hoạt động của thuật toán QuickSort' -> 'Thuật toán QuickSort'
    """
    if not query:
        return ""

    q = query.strip()
    clean_q = re.sub(
        r"^(?:cho biết|hãy cho biết|hãy nêu|hãy giải thích|trình bày|nêu|giải thích|chi tiết)?\s*",
        "",
        q,
        flags=re.IGNORECASE,
    ).strip()

    patterns = [
        r"(?:các bước thực hiện|các bước tiến hành|quy trình thực hiện|quy trình tiến hành)\s+(?:của\s+)?([^\?\.\n:]+)",
        r"(?:quy trình mã hóa và giải mã|quy trình mã hoá và giải mã|quy trình mã hóa|quy trình giải mã)\s+(?:của\s+)?(?:hệ mã\s+|thuật toán\s+)?([^\?\.\n:]+)",
        r"(?:cách hoạt động|cách thức hoạt động|nguyên lý hoạt động|cơ chế hoạt động)\s+(?:của\s+)?(?:thuật toán\s+|giao thức\s+|hệ mã\s+)?([^\?\.\n:]+)",
        r"(?:thuật toán|giao thức|hệ mã)\s+([^\?\.\n:]+)\s+(?:hoạt động như thế nào|hoạt động ra sao|được thực hiện như thế nào)",
        r"(?:các bước|quy trình|cách thức)\s+([^\?\.\n:]+)",
    ]

    for p in patterns:
        m = re.search(p, clean_q, re.IGNORECASE)
        if m:
            topic = m.group(1).strip()
            topic = re.sub(r"[\?\.\!:,]+$", "", topic).strip()
            if topic:
                return topic

    fallback = re.sub(
        r"(?:các bước thực hiện|quy trình mã hóa và giải mã|cách hoạt động của thuật toán|hoạt động như thế nào|như thế nào|cho ví dụ|\?|\.)",
        "",
        clean_q,
        flags=re.IGNORECASE,
    ).strip()
    return fallback if fallback else q


def detect_process_intent(query: str) -> Dict[str, Any]:
    """
    Analyzes query intent, extracting process type, topic, and expected phases.
    """
    if not query:
        return {
            "is_process_query": False,
            "process_type": "unknown",
            "topic": "",
            "expected_phases": ["main"],
        }

    q_lower = query.lower()
    q_norm = strip_accents(q_lower)

    # 1. Encryption & Decryption process
    is_enc_dec = (
        ("ma hoa va giai ma" in q_norm or "giai ma va ma hoa" in q_norm)
        or ("encryption and decryption" in q_lower)
    )

    # 2. Algorithm operation / Mechanism
    is_algo = (
        ("cach hoat dong" in q_norm)
        or ("hoat dong nhu the nao" in q_norm)
        or ("nguyen ly hoat dong" in q_norm)
        or ("cach thuc hoat dong" in q_norm)
        or ("thuat toan" in q_norm and "hoat dong" in q_norm)
        or ("how does it work" in q_lower)
        or ("algorithm" in q_lower and "works" in q_lower)
    )

    # 3. Step-by-step / General process
    is_step = (
        ("cac buoc" in q_norm)
        or ("quy trinh" in q_norm)
        or ("tien trinh" in q_norm)
        or ("trinh tu" in q_norm)
        or ("giai doan" in q_norm)
        or ("huong dan tung buoc" in q_norm)
        or ("step by step" in q_lower)
        or ("steps to" in q_lower)
        or ("workflow" in q_lower)
    )

    topic = extract_topic_from_query(query)

    if is_enc_dec:
        process_type = "encryption_decryption"
        expected_phases = ["key_generation", "encryption", "decryption"]
    elif is_algo:
        process_type = "algorithm_workflow"
        expected_phases = ["main"]
    elif is_step:
        process_type = "step_by_step"
        expected_phases = ["main"]
    else:
        process_type = "general_process"
        expected_phases = ["main"]

    is_process_query = is_enc_dec or is_algo or is_step

    return {
        "is_process_query": is_process_query,
        "process_type": process_type,
        "topic": topic,
        "expected_phases": expected_phases,
    }


def is_phase_header_line(line: str) -> Optional[str]:
    """
    Checks if a line is a standalone phase / sub-process section header.
    Strictly verifies that the line is NOT a step item and has header structure.
    """
    l = clean_text_line(line)
    if not l:
        return None

    # Step markers or list lines are NEVER standalone phase headers
    if STEP_PREFIX_PATTERN.match(l) or NUMBERED_LIST_PATTERN.match(l) or LETTERED_LIST_PATTERN.match(l):
        return None
    for trans_regex, _, _ in TRANSITION_MARKERS:
        if trans_regex.match(l):
            return None

    # Must have heading characteristics (ends with ':', starts with '#', or short title without period)
    words = l.split()
    is_header_format = (
        l.endswith(":")
        or l.startswith("#")
        or (len(words) <= 7 and not l.endswith("."))
    )
    if not is_header_format:
        return None

    l_norm = strip_accents(l.lower())
    for phase, regex_list in PHASE_HEADER_PATTERNS.items():
        for pat in regex_list:
            pat_norm = strip_accents(pat)
            if re.search(pat_norm, l_norm):
                return phase

    return None


class StepByStepProcessStitcher:
    """
    Robust Extractive Step-by-Step Process & Algorithm Synthesizer.
    Pure Python stdlib, 100% strict offline.
    """

    def __init__(
        self,
        step_prefix: str = "Bước",
        include_intro: bool = True,
        include_summary: bool = True,
        deduplicate: bool = True,
        format_markdown: bool = True,
    ):
        self.step_prefix = step_prefix
        self.include_intro = include_intro
        self.include_summary = include_summary
        self.deduplicate = deduplicate
        self.format_markdown = format_markdown

    def _normalize_passages(
        self, passages: Union[List[Dict[str, Any]], List[Any]]
    ) -> List[Dict[str, Any]]:
        """Normalizes diverse passage formats (dict, objects, strings) into standard dicts."""
        normalized: List[Dict[str, Any]] = []
        if not passages:
            return normalized

        for idx, p in enumerate(passages):
            if isinstance(p, dict):
                pid = p.get("passage_id") or p.get("id") or f"P{idx + 1:03d}"
                text = p.get("text") or p.get("content") or ""
                sec = p.get("section_title") or p.get("title") or ""
                meta = p.get("metadata") or {}
            elif isinstance(p, str):
                pid = f"P{idx + 1:03d}"
                text = p
                sec = ""
                meta = {}
            else:
                pid = getattr(p, "passage_id", getattr(p, "id", f"P{idx + 1:03d}"))
                text = getattr(p, "text", getattr(p, "content", getattr(p, "page_content", "")))
                sec = getattr(p, "section_title", getattr(p, "title", ""))
                meta = getattr(p, "metadata", {})

            if text and str(text).strip():
                normalized.append({
                    "passage_id": str(pid),
                    "text": str(text),
                    "section_title": str(sec),
                    "metadata": meta,
                })

        return normalized

    def _parse_passage_into_steps(
        self, passage: Dict[str, Any], process_type: str = "general_process"
    ) -> List[Dict[str, Any]]:
        """
        Parses text lines of a single passage and extracts ordered step structures.
        Accurately identifies step items and phase boundaries.
        """
        raw_text = passage["text"]
        pid = passage["passage_id"]
        sec_title = passage["section_title"]

        lines = raw_text.splitlines()
        extracted_steps: List[Dict[str, Any]] = []

        current_phase = is_phase_header_line(sec_title) or "main"
        consecutive_step_counter = 0

        for line in lines:
            line_str = clean_text_line(line)
            if not line_str:
                continue

            # 1. Check explicit Step prefix: "Bước 1: ...", "Step 2: ..."
            m_step = STEP_PREFIX_PATTERN.match(line_str)
            if m_step:
                num_str, title_paren, content = m_step.groups()
                try:
                    num = int(num_str)
                except ValueError:
                    num = consecutive_step_counter + 1

                step_title = (title_paren or "").strip()
                step_content = (content or "").strip()

                if not step_title and (" - " in step_content or ": " in step_content):
                    sep = " - " if " - " in step_content else ": "
                    parts = step_content.split(sep, 1)
                    if len(parts[0].split()) <= 6 and not parts[0].endswith("."):
                        step_title = parts[0].strip()
                        step_content = parts[1].strip()

                consecutive_step_counter = num
                extracted_steps.append({
                    "step_number": num,
                    "title": step_title,
                    "content": step_content or step_title,
                    "phase": current_phase,
                    "source_passage_id": pid,
                    "raw_line": line_str,
                })
                continue

            # 2. Check numbered lists: "1. ...", "2) ..."
            m_num = NUMBERED_LIST_PATTERN.match(line_str)
            if m_num:
                num_1, num_2, content = m_num.groups()
                num = int(num_1 or num_2)
                step_content = content.strip()
                step_title = ""

                if ": " in step_content or " - " in step_content:
                    sep = ": " if ": " in step_content else " - "
                    parts = step_content.split(sep, 1)
                    if len(parts[0].split()) <= 6 and not parts[0].endswith("."):
                        step_title = parts[0].strip()
                        step_content = parts[1].strip()

                # Infer phase if inline step mentions key gen / enc / dec
                step_phase = current_phase
                if process_type == "encryption_decryption" and current_phase == "main":
                    c_norm = strip_accents(step_content.lower())
                    if any(k in c_norm for k in ["sinh khoa", "tao khoa", "chon khoa", "khoi tao khoa"]):
                        step_phase = "key_generation"
                    elif any(k in c_norm for k in ["ma hoa", "ban ma c"]):
                        step_phase = "encryption"
                    elif any(k in c_norm for k in ["giai ma", "ban ro m"]):
                        step_phase = "decryption"

                consecutive_step_counter = num
                extracted_steps.append({
                    "step_number": num,
                    "title": step_title,
                    "content": step_content or step_title,
                    "phase": step_phase,
                    "source_passage_id": pid,
                    "raw_line": line_str,
                })
                continue

            # 3. Check transition markers: "Đầu tiên, ...", "Tiếp theo, ..."
            transition_matched = False
            for trans_regex, trans_num, trans_label in TRANSITION_MARKERS:
                m_trans = trans_regex.match(line_str)
                if m_trans:
                    content = m_trans.group(1).strip()
                    if trans_num == 0 or trans_num == 999:
                        consecutive_step_counter += 1
                        step_num = consecutive_step_counter
                    else:
                        step_num = trans_num
                        consecutive_step_counter = step_num

                    extracted_steps.append({
                        "step_number": step_num,
                        "title": trans_label,
                        "content": content,
                        "phase": current_phase,
                        "source_passage_id": pid,
                        "raw_line": line_str,
                    })
                    transition_matched = True
                    break

            if transition_matched:
                continue

            # 4. Check lettered lists: "a) ...", "b) ..."
            m_let = LETTERED_LIST_PATTERN.match(line_str)
            if m_let:
                let_1, let_2, content = m_let.groups()
                let_char = (let_1 or let_2).lower()
                step_num = ord(let_char) - ord("a") + 1
                consecutive_step_counter = step_num
                extracted_steps.append({
                    "step_number": step_num,
                    "title": f"Mục {let_char.upper()}",
                    "content": content.strip(),
                    "phase": current_phase,
                    "source_passage_id": pid,
                    "raw_line": line_str,
                })
                continue

            # 5. Check if this line is a standalone phase header
            detected_phase = is_phase_header_line(line_str)
            if detected_phase:
                current_phase = detected_phase
                consecutive_step_counter = 0
                continue

            # 6. Check action-oriented bullets: lines starting with technical action verbs
            first_word = line_str.split()[0].lower() if line_str.split() else ""
            if any(first_word == v or first_word.startswith(f"{v}:") for v in ACTION_VERB_PREFIXES):
                consecutive_step_counter += 1
                extracted_steps.append({
                    "step_number": consecutive_step_counter,
                    "title": "",
                    "content": line_str,
                    "phase": current_phase,
                    "source_passage_id": pid,
                    "raw_line": line_str,
                })

        return extracted_steps

    def _deduplicate_and_reindex_steps(
        self, steps: List[Dict[str, Any]], process_type: str
    ) -> List[Dict[str, Any]]:
        """
        Deduplicates steps across overlapping chunks and reindexes step numbers monotonically.
        """
        if not steps:
            return []

        seen_signatures: Set[str] = set()
        unique_steps: List[Dict[str, Any]] = []

        for st in steps:
            content = st.get("content", "").strip()
            # Canonical signature: stripped alphanumeric without diacritics
            sig = strip_accents(re.sub(r"[^a-zA-Z0-9\(\)=+\-*/^]", "", content.lower()))
            if not sig or len(sig) < 8:
                sig = content.lower()

            phase = st.get("phase", "main")
            full_sig = f"{phase}::{sig}"

            if full_sig in seen_signatures:
                continue
            seen_signatures.add(full_sig)
            unique_steps.append(st)

        # For algorithms and general processes, keep all steps in a unified sequence unless explicitly split
        if process_type != "encryption_decryption":
            unique_steps.sort(key=lambda s: s.get("step_number", 0))
            for idx, st in enumerate(unique_steps):
                st["step_number"] = idx + 1
                st["phase"] = "main"
            return unique_steps

        # For encryption & decryption, group steps per phase and reindex each starting from 1
        phase_groups: Dict[str, List[Dict[str, Any]]] = {}
        for st in unique_steps:
            phase = st.get("phase", "main")
            phase_groups.setdefault(phase, []).append(st)

        reindexed_steps: List[Dict[str, Any]] = []
        phase_order = ["key_generation", "encryption", "decryption", "main"]
        for phase in phase_order:
            if phase in phase_groups:
                p_list = phase_groups[phase]
                p_list.sort(key=lambda s: s.get("step_number", 0))
                for idx, st in enumerate(p_list):
                    st["step_number"] = idx + 1
                    reindexed_steps.append(st)

        return reindexed_steps

    def _extract_grounded_notes_or_formulas(
        self, passages: List[Dict[str, Any]]
    ) -> List[str]:
        """Extracts complementary notes, formulas, or security properties if present."""
        notes: List[str] = []
        note_patterns = [
            re.compile(r"\b(?:lưu ý|nhận xét|chú ý|điều kiện|yêu cầu an ninh|độ phức tạp)\s*:\s*(.*)$", re.IGNORECASE),
            re.compile(r"\b(?:an ninh|bảo mật|tấn công|đặc điểm|ứng dụng)\s*:\s*(.*)$", re.IGNORECASE),
        ]

        for p in passages:
            lines = p["text"].splitlines()
            for line in lines:
                l = clean_text_line(line)
                if not l:
                    continue
                # Skip step lines
                if STEP_PREFIX_PATTERN.match(l) or NUMBERED_LIST_PATTERN.match(l):
                    continue
                for pat in note_patterns:
                    m = pat.search(l)
                    if m:
                        val = m.group(1).strip() if m.group(1).strip() else l
                        if val and val not in notes and len(val) > 8:
                            notes.append(val)
        return notes

    def _render_phase_title(self, phase: str, phase_index: int) -> str:
        """Renders clear, human-readable section headings for distinct sub-processes."""
        titles = {
            "key_generation": "Quy trình sinh khóa (Key Generation)",
            "encryption": "Quy trình mã hóa (Encryption Process)",
            "decryption": "Quy trình giải mã (Decryption Process)",
            "main": "Các bước thực hiện",
        }
        name = titles.get(phase, f"Quy trình {phase}")
        return f"### {phase_index}. {name}"

    def format_step_text(
        self,
        query: str,
        topic: str,
        process_type: str,
        steps: List[Dict[str, Any]],
        sub_processes: Dict[str, List[Dict[str, Any]]],
        notes: List[str],
    ) -> str:
        """
        Formats extracted steps into fluent, coherent, and polished markdown text.
        Conforms strictly to the user brief: 'trình bày mạch lạc theo thứ tự Bước 1, Bước 2...'.
        """
        if not steps:
            return ""

        lines: List[str] = []

        # 1. Opening Statement / Lead-in
        t_label = f" của {topic}" if topic else ""
        if process_type == "encryption_decryption":
            lines.append(f"Quy trình mã hóa và giải mã{t_label} được thực hiện qua các bước cụ thể như sau:\n")
        elif process_type == "algorithm_workflow":
            lines.append(f"Cách thức hoạt động và các bước thực hiện{t_label} diễn ra theo trình tự sau:\n")
        else:
            lines.append(f"Các bước thực hiện{t_label} được tiến hành tuần tự theo các bước sau:\n")

        # 2. Phase-wise formatting
        phase_order = ["key_generation", "encryption", "decryption", "main"]
        active_phases = [p for p in phase_order if p in sub_processes and sub_processes[p]]
        has_multiple_phases = len(active_phases) > 1 and process_type == "encryption_decryption"

        phase_counter = 1
        for phase in active_phases:
            p_steps = sub_processes[phase]
            if not p_steps:
                continue

            if has_multiple_phases:
                lines.append(self._render_phase_title(phase, phase_counter))
                phase_counter += 1

            for st in p_steps:
                s_num = st["step_number"]
                s_title = st.get("title", "").strip()
                s_content = st.get("content", "").strip()

                if s_title and s_title.endswith(":"):
                    s_title = s_title[:-1].strip()

                prefix = self.step_prefix
                if s_title and s_title != f"{prefix} {s_num}":
                    step_line = f"- **{prefix} {s_num} ({s_title})**: {s_content}"
                else:
                    step_line = f"- **{prefix} {s_num}**: {s_content}"

                if step_line[-1] not in ".!?:":
                    step_line += "."

                lines.append(step_line)

            lines.append("")  # Blank line between sections

        # 3. Grounded Notes / Additional Information
        if notes and self.include_summary:
            lines.append("### Lưu ý & Đặc điểm:")
            for note in notes[:3]:
                n_str = note.strip()
                if n_str[-1] not in ".!?:":
                    n_str += "."
                lines.append(f"- {n_str}")
            lines.append("")

        return "\n".join(lines).strip()

    def synthesize(
        self, query: str, passages: Union[List[Dict[str, Any]], List[Any]]
    ) -> Dict[str, Any]:
        """
        Synthesizes a complete step-by-step process answer from query and passages.
        """
        # Step 1: Detect intent and topic
        intent = detect_process_intent(query)
        norm_passages = self._normalize_passages(passages)
        process_type = intent.get("process_type", "general_process")

        if not norm_passages:
            return {
                "status": "no_content",
                "query": query,
                "topic": intent.get("topic", ""),
                "process_type": process_type,
                "steps": [],
                "sub_processes": {},
                "formatted_text": "",
                "answer": "",
                "step_count": 0,
                "citations": [],
                "is_process_query": intent.get("is_process_query", False),
                "metadata": {"error": "Empty or invalid passages provided"},
            }

        # Step 2: Extract steps from each passage
        raw_steps: List[Dict[str, Any]] = []
        citations_set: Set[str] = set()

        for p in norm_passages:
            p_steps = self._parse_passage_into_steps(p, process_type=process_type)
            raw_steps.extend(p_steps)
            citations_set.add(p["passage_id"])

        # Fallback heuristic: If no explicit steps detected, extract sentences as sequential steps
        if not raw_steps:
            seq_num = 1
            for p in norm_passages:
                sents = re.split(r"(?<=[.!?])\s+", p["text"])
                for s in sents:
                    clean_s = clean_text_line(s)
                    if clean_s and len(clean_s) > 15:
                        raw_steps.append({
                            "step_number": seq_num,
                            "title": "",
                            "content": clean_s,
                            "phase": "main",
                            "source_passage_id": p["passage_id"],
                            "raw_line": clean_s,
                        })
                        seq_num += 1

        # Step 3: Deduplicate and reindex steps
        cleaned_steps = self._deduplicate_and_reindex_steps(raw_steps, process_type=process_type)

        # Step 4: Group steps by phase / sub-process
        sub_processes: Dict[str, List[Dict[str, Any]]] = {}
        for st in cleaned_steps:
            ph = st.get("phase", "main")
            sub_processes.setdefault(ph, []).append(st)

        # Step 5: Extract complementary notes / formulas
        notes = self._extract_grounded_notes_or_formulas(norm_passages)

        # Step 6: Format text response
        formatted_answer = self.format_step_text(
            query=query,
            topic=intent.get("topic", ""),
            process_type=process_type,
            steps=cleaned_steps,
            sub_processes=sub_processes,
            notes=notes,
        )

        citations_list = sorted(list(citations_set))

        return {
            "status": "success",
            "query": query,
            "topic": intent.get("topic", ""),
            "process_type": process_type,
            "steps": cleaned_steps,
            "sub_processes": sub_processes,
            "formatted_text": formatted_answer,
            "answer": formatted_answer,
            "step_count": len(cleaned_steps),
            "citations": citations_list,
            "is_process_query": intent.get("is_process_query", False),
            "metadata": {
                "phases_detected": list(sub_processes.keys()),
                "has_notes": len(notes) > 0,
            },
        }


def synthesize_step_by_step_process(
    query: str, passages: Union[List[Dict[str, Any]], List[Any]]
) -> Dict[str, Any]:
    """
    Main entry point for Step-by-Step Process & Algorithm Synthesis.
    Complies with Sovereign Brain v5.5 specification.

    Args:
        query: User question, e.g. 'các bước thực hiện', 'quy trình mã hóa và giải mã',
               'cách hoạt động của thuật toán'.
        passages: List of passage dictionaries (must contain 'text', optionally 'section_title',
                  'passage_id', 'metadata').

    Returns:
        Structured dictionary with status, topic, process_type, ordered steps,
        sub-processes, and formatted answer text presenting Step 1, Step 2...
    """
    stitcher = StepByStepProcessStitcher()
    return stitcher.synthesize(query=query, passages=passages)
