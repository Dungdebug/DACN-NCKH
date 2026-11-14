"""
Chunker — splits documents into retrievable passages with unique IDs.

Design:
- Sentence-aware chunking to avoid splitting mid-sentence.
- Falls back to fixed-token chunking if no sentence boundaries found.
- Assigns passage IDs in format DOC{id}::P{idx}.
- Preserves character and token offsets for citation traceability.
- Adaptive Structural Boundary Overlap: zero-overlap across paragraph boundaries,
  strict zero-overlap on section / topic boundaries (Topic N, Chủ đề N, etc.),
  and absolute zero-overlap across document page breaks (=== Page N ===).
"""

import re
from typing import List, Optional, Tuple, Any, Callable
from src.hybrid_qa.document_store import Passage, DocumentRecord


TOPIC_HEADING_KEYWORDS = (
    r"topic|chủ đề|chu de|bài|bai|mục|muc|chương|chuong|phần|phan|section|chapter|lesson|unit"
)
ROMAN_NUMERALS = r"[IVXLCDM]+"


class DocumentChunker:
    """
    Chunks a document into passages suitable for retrieval.
    Supports sentence-aware and fixed-size chunking strategies.
    Strictly enforces structural isolation on heading / topic / page boundaries.
    """

    SENTENCE_BOUNDARY = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F])|'  # Period/excl/question + space + uppercase
        r'(?<=\n)\s*\n+'                             # Blank line
    )

    # Page Marker Regex: recognizes page dividers like === Page 1 ===
    PAGE_MARKER_RE = re.compile(r"^\s*===\s*Page\s*(\d+)\s*===\s*$", re.IGNORECASE)

    # Shared Heading Regex: recognizes Markdown headers, topic headers, page headers, and numbered sections
    HEADING_RE = re.compile(
        rf"^\s*(?:"
        rf"===\s*Page\s*\d+\s*===(?:\s+.*)?|"
        rf"#{{1,6}}\s+\S.*|"
        rf"(?:{TOPIC_HEADING_KEYWORDS})\s*\d+\s*[:.\-–].*|"
        rf"(?:\d+(?:\.\d+)*|{ROMAN_NUMERALS})\.\s+[A-ZÀ-Ỹ0-9].*"
        rf")$",
        re.MULTILINE | re.IGNORECASE,
    )

    # Heading Finder Regex: locates headings occurring at line start or inline after sentence boundaries
    HEADING_FINDER = re.compile(
        rf"(?:^|(?<=[\n.!?]))\s*(?P<heading>("
        rf"===\s*Page\s*\d+\s*===[^\n]*|"
        rf"#{{1,6}}\s+\S[^\n]*|"
        rf"(?:{TOPIC_HEADING_KEYWORDS})\s*\d+\s*[:.\-–][^\n]*|"
        rf"(?:\d+(?:\.\d+)*|{ROMAN_NUMERALS})\.\s+[A-ZÀ-Ỹ0-9][^\n]*"
        rf"))",
        re.IGNORECASE | re.MULTILINE,
    )

    ABBREVIATIONS = {
        "tp.", "ts.", "pgs.", "gs.", "ths.", "ks.", "cn.", "bs.", "đc.", "tr.",
        "dr.", "mr.", "mrs.", "ms.", "prof.", "fig.", "eq.", "sec.", "vol.", "no.", "vs.", "al.",
    }

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        min_chunk_size: int = 64,
        strategy: str = "sentence",  # "sentence" / "semantic" or "fixed"
        adaptive_overlap: bool = True,
        use_tokens: bool = False,
        tokenizer: Optional[Any] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.strategy = strategy
        self.adaptive_overlap = adaptive_overlap
        self.use_tokens = use_tokens
        self.tokenizer = tokenizer

    def _measure_len(self, text: str) -> int:
        """Measure length of text in tokens (if use_tokens=True) or characters."""
        if not text:
            return 0
        if not self.use_tokens:
            return len(text)
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        # Multilingual heuristic: ~1.35 tokens per word
        return max(1, int(round(len(text.split()) * 1.35)))

    @classmethod
    def is_heading(cls, text: str) -> bool:
        """Checks if a string represents or starts with a recognized section/topic heading."""
        if not text:
            return False
        clean = re.sub(r"^\s*===\s*Page\s*\d+\s*===\s*", "", text.strip()).strip()
        first_line = clean.split("\n")[0].strip()
        return bool(cls.HEADING_RE.match(first_line)) or bool(cls.PAGE_MARKER_RE.match(text.strip()))

    @classmethod
    def is_page_marker(cls, text: str) -> bool:
        """Checks if a string is a standalone page marker like === Page 1 ===."""
        return bool(cls.PAGE_MARKER_RE.match(text.strip()))

    def chunk_document(self, doc: DocumentRecord) -> List[Passage]:
        """
        Chunk a document into passages and assign unique IDs.
        Returns list of Passage objects with DOC{id}::P{idx} identifiers.
        """
        text = doc.raw_text
        if not text.strip():
            return []

        if self.strategy in ("sentence", "semantic"):
            raw_chunks = self._sentence_chunk(text)
        else:
            raw_chunks = self._fixed_chunk(text)

        passages = []
        for idx, (chunk_text, start_char, end_char) in enumerate(raw_chunks):
            passage_id = f"{doc.document_id}::P{idx:03d}"
            section_title = self._find_section_title(text, start_char)
            section_id = self._find_section_id(text, start_char)
            paragraph_id = self._find_paragraph_id(text, start_char)

            metadata = {
                "section_id": section_id,
                "paragraph_id": paragraph_id,
                "document_version": doc.version,
            }
            if doc.metadata:
                for k, v in doc.metadata.items():
                    if k not in metadata:
                        metadata[k] = v

            passages.append(Passage(
                passage_id=passage_id,
                document_id=doc.document_id,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                section_title=section_title,
                metadata=metadata,
            ))

        return passages

    def _sentence_chunk(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Sentence-aware chunking with Adaptive Structural Boundary Overlap:
        - Standalone page markers (=== Page N ===) are absolute boundaries: current sentences are
          flushed immediately with ZERO overlap, and page markers are stripped from passage text.
        - Section/topic headings (HEADING_RE) are flushed immediately with ZERO overlap.
        - Natural paragraph boundaries have ZERO overlap.
        - Overlap never crosses a heading or page break.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return self._fixed_chunk(text)

        chunks = []
        current_sentences = []
        current_len = 0
        current_start = 0

        for sent_text, sent_start, sent_end in sentences:
            # 1. Standalone page marker: flush immediately with ZERO overlap and discard marker
            if self.is_page_marker(sent_text):
                if current_sentences:
                    chunk_text = " ".join(s[0] for s in current_sentences).strip()
                    if chunk_text:
                        chunks.append((chunk_text, current_sentences[0][1], current_sentences[-1][2]))
                    current_sentences = []
                    current_len = 0
                current_start = sent_end
                continue

            # 2. Strip inline page marker prefix (e.g. "=== Page 31 === 10. HỆ MÃ HOÁ...")
            clean_sent = re.sub(r"^\s*===\s*Page\s*\d+\s*===\s*", "", sent_text).strip()
            if not clean_sent:
                continue

            sent_len = self._measure_len(clean_sent)
            sent_is_heading = self.is_heading(clean_sent)

            # 3. RULE: A heading must never be conjoined with previous section/topic sentences!
            if sent_is_heading and current_sentences:
                chunk_text = " ".join(s[0] for s in current_sentences).strip()
                if chunk_text:
                    chunks.append((chunk_text, current_sentences[0][1], current_sentences[-1][2]))
                # Strict ZERO overlap across heading boundaries
                current_sentences = []
                current_len = 0
                current_start = sent_start

            elif (
                current_len + sent_len > self.chunk_size
                and current_sentences
                and not (len(current_sentences) == 1 and self.is_heading(current_sentences[0][0]))
            ):
                chunk_text = " ".join(s[0] for s in current_sentences).strip()
                chunk_end = current_sentences[-1][2]
                if chunk_text:
                    chunks.append((chunk_text, current_sentences[0][1], chunk_end))

                # Check if boundary between chunk_end and sent_start is a natural paragraph/section break
                gap_text = text[chunk_end:sent_start] if sent_start >= chunk_end else ""
                is_natural_boundary = ("\n\n" in gap_text) or clean_sent.lstrip().startswith("#") or sent_is_heading

                if self.adaptive_overlap and is_natural_boundary:
                    # Zero-overlap on natural structural boundaries to save tokens
                    current_sentences = []
                    current_len = 0
                    current_start = sent_start
                else:
                    # Mid-paragraph split: keep last few sentences for continuity,
                    # but never allow overlap to step backward across a heading
                    overlap_len = 0
                    overlap_start = len(current_sentences)
                    for i in range(len(current_sentences) - 1, -1, -1):
                        if self.is_heading(current_sentences[i][0]):
                            overlap_start = i + 1
                            break
                        overlap_len += self._measure_len(current_sentences[i][0])
                        if overlap_len >= self.chunk_overlap:
                            overlap_start = i
                            break

                    current_sentences = current_sentences[overlap_start:]
                    current_len = sum(self._measure_len(s[0]) for s in current_sentences)
                    current_start = current_sentences[0][1] if current_sentences else sent_start

            current_sentences.append((clean_sent, sent_start, sent_end))
            current_len += sent_len

        # Emit remaining
        if current_sentences:
            chunk_text = " ".join(s[0] for s in current_sentences).strip()
            if self._measure_len(chunk_text) >= self.min_chunk_size:
                chunks.append((chunk_text, current_sentences[0][1], current_sentences[-1][2]))
            elif chunks:
                # Only merge trailing chunk if both chunks share the same section, neither is a heading,
                # and no page break separates them
                prev_text, prev_start, prev_end = chunks[-1]
                prev_sec = self._find_section_title(text, prev_start)
                curr_sec = self._find_section_title(text, current_sentences[0][1])
                gap_between = text[prev_end:current_sentences[0][1]]
                has_page_break = bool(re.search(r"===\s*Page\s*\d+\s*===", gap_between, re.IGNORECASE))

                if (
                    prev_sec
                    and curr_sec
                    and prev_sec == curr_sec
                    and not has_page_break
                    and not any(self.is_heading(s[0]) for s in current_sentences)
                ):
                    chunks[-1] = (prev_text + " " + chunk_text, prev_start, current_sentences[-1][2])
                else:
                    chunks.append((chunk_text, current_sentences[0][1], current_sentences[-1][2]))
            else:
                chunks.append((chunk_text, current_sentences[0][1], current_sentences[-1][2]))

        return chunks

    def _fixed_chunk(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Fixed-size chunking with overlap.
        Supports character-based or token/word-based slicing when use_tokens=True.
        """
        chunks = []
        if not self.use_tokens:
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end]
                if chunk_text.strip():
                    chunks.append((chunk_text.strip(), start, end))
                step = max(1, self.chunk_size - self.chunk_overlap)
                start += step
            return chunks

        # Token/Word-aware fixed slicing (VD: 300 tokens/chunk with overlap)
        words = list(re.finditer(r"\S+", text))
        if not words:
            return []

        # Map word bounds
        word_count = len(words)
        # Approximate tokens to words ratio (1 token ~ 0.75 words, so 300 tokens ~ 225 words)
        words_per_chunk = max(1, int(self.chunk_size / 1.35))
        words_overlap = max(0, int(self.chunk_overlap / 1.35))
        step = max(1, words_per_chunk - words_overlap)

        w_start = 0
        while w_start < word_count:
            w_end = min(w_start + words_per_chunk, word_count)
            c_start = words[w_start].start()
            c_end = words[w_end - 1].end()
            chunk_text = text[c_start:c_end].strip()
            if chunk_text:
                chunks.append((chunk_text, c_start, c_end))
            if w_end >= word_count:
                break
            w_start += step

        return chunks

    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into (sentence_text, start_char, end_char) tuples.
        Protects common abbreviations, numbered headings (1. 2. 10. I.), and decimals.
        Isolates section headings and page markers as distinct structural boundaries.
        """
        if not text.strip():
            return []

        boundaries: List[Tuple[int, int, str]] = []

        # 1. Standard sentence boundary matches (punctuation + space + uppercase, or \n\n)
        for match in self.SENTENCE_BOUNDARY.finditer(text):
            boundaries.append((match.start(), match.end(), "std"))

        # 2. Structural heading / page marker boundaries
        for match in self.HEADING_FINDER.finditer(text):
            h_start = match.start("heading")
            h_end = match.end("heading")
            if h_start > 0:
                boundaries.append((h_start, h_start, "head_start"))
            next_pos = h_end
            while next_pos < len(text) and text[next_pos] in " \t\r\n":
                next_pos += 1
            boundaries.append((h_end, next_pos, "head_end"))

        # Sort and merge overlapping / adjacent split boundaries
        boundaries.sort(key=lambda x: (x[0], x[1]))
        merged: List[List[Any]] = []
        for b_start, b_end, b_type in boundaries:
            if not merged:
                merged.append([b_start, b_end, b_type])
            else:
                prev = merged[-1]
                if b_start <= prev[1]:
                    prev[1] = max(prev[1], b_end)
                    if b_type != "std":
                        prev[2] = b_type
                else:
                    merged.append([b_start, b_end, b_type])

        sentences = []
        last_end = 0
        for b_start, b_end, b_type in merged:
            segment = text[last_end:b_start]
            cand = segment.strip()
            if not cand:
                last_end = max(last_end, b_end)
                continue

            last_word = cand.split()[-1] if cand.split() else ""
            # Protect numbered section prefixes (e.g. '1.', '3.', '10.', 'I.') from premature splitting
            if b_type == "std" and "\n" not in text[b_start:b_end]:
                if re.match(rf"^(?:\d+(?:\.\d+)*|{ROMAN_NUMERALS})\.$", last_word, re.IGNORECASE):
                    continue
                if last_word.lower() in self.ABBREVIATIONS:
                    continue

            lead = len(segment) - len(segment.lstrip())
            s_start = last_end + lead
            s_end = s_start + len(cand)
            sentences.append((cand, s_start, s_end))
            last_end = max(last_end, b_end)

        remaining = text[last_end:]
        if remaining.strip():
            cand = remaining.strip()
            lead = len(remaining) - len(remaining.lstrip())
            s_start = last_end + lead
            sentences.append((cand, s_start, s_start + len(cand)))

        return sentences

    @classmethod
    def _find_section_title(cls, full_text: str, char_pos: int) -> str:
        """Find the nearest preceding section heading for a character position."""
        best_title = ""
        best_pos = -1
        last_page = ""
        for match in cls.HEADING_FINDER.finditer(full_text):
            h_start = match.start("heading")
            raw_h = match.group("heading").strip().lstrip("#").strip()
            pm = cls.PAGE_MARKER_RE.match(raw_h)
            clean_h = re.sub(r"^===\s*Page\s*\d+\s*===\s*", "", raw_h).strip()

            if h_start <= (char_pos + 2):
                if clean_h and h_start > best_pos:
                    best_pos = h_start
                    best_title = clean_h
                elif pm and not best_title:
                    last_page = pm.group(0).strip("= ")

        return best_title or last_page

    @classmethod
    def _find_section_id(cls, full_text: str, char_pos: int) -> str:
        """Find the section ID (e.g. SEC001) for a character position."""
        headings = []
        for match in cls.HEADING_FINDER.finditer(full_text):
            raw_h = match.group("heading").strip()
            clean_h = re.sub(r"^===\s*Page\s*\d+\s*===\s*", "", raw_h).strip()
            if clean_h:
                headings.append((match.start("heading"), clean_h))
            elif cls.PAGE_MARKER_RE.match(raw_h):
                headings.append((match.start("heading"), raw_h))
        headings.sort(key=lambda x: x[0])

        sec_idx = 0
        for i, (pos, _) in enumerate(headings):
            if pos <= (char_pos + 2):
                sec_idx = i + 1
            else:
                break
        return f"SEC{sec_idx:03d}"

    @staticmethod
    def _find_paragraph_id(full_text: str, char_pos: int) -> str:
        """Find the paragraph ID (e.g. PARA001) for a character position."""
        prefix = full_text[:char_pos]
        paras = re.split(r'\n\s*\n+', prefix.strip())
        para_idx = len(paras) if paras and paras[0] else 1
        return f"PARA{para_idx:03d}"
