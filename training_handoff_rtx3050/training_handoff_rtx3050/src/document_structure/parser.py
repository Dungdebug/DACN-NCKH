"""
Document Structure Parser for Structure-Aligned Continuum Memory System (SA-CMS).
Extracts hierarchical structural spans: Document, Sections, Paragraphs, Chunks.
Includes robust fallback for documents without explicit headings and random boundary ablation.
"""

import re
import random
from typing import List, Dict, Any, Optional, Tuple

from src.document_structure.types import BoundaryType, StructuralSpan, DocumentStructure


class DocumentStructureParser:
    """
    Parses natural language documents into hierarchical structural spans:
    - Document level (Coarse timescale)
    - Section level (Intermediate timescale)
    - Paragraph level (Fine timescale)
    - Fallback chunk level (Token-based fallback)
    """

    HEADING_PATTERNS = [
        # Markdown headings: # Heading, ## Section
        re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE),
        # Numbered headings: 1. Introduction, 2.1 Related Work
        re.compile(r"^(\d+(\.\d+)*)\.?\s+([A-Z][^\n]+)$", re.MULTILINE),
        # All-caps standard standalone section titles: ABSTRACT, INTRODUCTION, METHOD
        re.compile(
            r"^(ABSTRACT|INTRODUCTION|BACKGROUND|RELATED WORK|METHOD|METHODOLOGY|"
            r"EXPERIMENTS|RESULTS|DISCUSSION|CONCLUSION|REFERENCES)\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
        # Explicit section delimiters: === Section Name ===
        re.compile(r"^={3,}\s*(.+?)\s*={3,}$", re.MULTILINE),
    ]

    PARAGRAPH_PATTERN = re.compile(r"\n\s*\n+")

    def __init__(
        self,
        fallback_paragraph_tokens: int = 32,
        fallback_section_tokens: int = 64,
        paragraphs_per_section_fallback: int = 3,
    ):
        self.fallback_paragraph_tokens = fallback_paragraph_tokens
        self.fallback_section_tokens = fallback_section_tokens
        self.paragraphs_per_section_fallback = paragraphs_per_section_fallback

    def _get_char_to_token_map(self, text: str, tokenizer=None) -> Tuple[List[int], List[int]]:
        """
        Builds mapping from character index to token index.
        Returns:
            token_ids: List of token IDs
            char_to_token: List of length len(text)+1 mapping char_idx -> token_idx
        """
        if tokenizer is not None:
            # Use tokenizer encoding with offsets if available
            enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            token_ids = enc.input_ids
            offset_mapping = enc.offset_mapping

            char_to_token = [0] * (len(text) + 1)
            curr_token = 0
            for t_idx, (start, end) in enumerate(offset_mapping):
                for c in range(start, min(end, len(text) + 1)):
                    char_to_token[c] = t_idx
                curr_token = t_idx + 1

            # Fill in remainder
            last_token = len(token_ids)
            for c in range(len(char_to_token) - 1, -1, -1):
                if char_to_token[c] == 0 and c > 0:
                    char_to_token[c] = char_to_token[min(c + 1, len(char_to_token) - 1)]
            char_to_token[-1] = last_token
            return token_ids, char_to_token
        else:
            # Word-based tokenization fallback
            words = text.split()
            token_ids = list(range(len(words)))
            char_to_token = [0] * (len(text) + 1)
            c_pos = 0
            for w_idx, word in enumerate(words):
                w_start = text.find(word, c_pos)
                if w_start != -1:
                    for i in range(c_pos, w_start + len(word)):
                        if i < len(char_to_token):
                            char_to_token[i] = w_idx
                    c_pos = w_start + len(word)
            for i in range(c_pos, len(char_to_token)):
                char_to_token[i] = len(words)
            return token_ids, char_to_token

    def parse_text(self, text: str, tokenizer=None) -> DocumentStructure:
        """
        Parses raw text into a hierarchical DocumentStructure.
        Detects real sections and paragraphs, with robust fallback when missing.
        """
        text = text.strip()
        if not text:
            return DocumentStructure(text="", total_tokens=0)

        token_ids, char_to_token = self._get_char_to_token_map(text, tokenizer)
        total_tokens = len(token_ids)

        # 1. PARAGRAPH DETECTION
        # Split on double newlines
        paragraph_spans = []
        raw_paragraphs = [p for p in self.PARAGRAPH_PATTERN.split(text) if p.strip()]

        if len(raw_paragraphs) > 1:
            search_pos = 0
            for p_idx, p_text in enumerate(raw_paragraphs):
                p_text_clean = p_text.strip()
                p_start_char = text.find(p_text_clean, search_pos)
                if p_start_char == -1:
                    p_start_char = search_pos
                p_end_char = p_start_char + len(p_text_clean)
                search_pos = p_end_char

                p_start_tok = char_to_token[p_start_char]
                p_end_tok = char_to_token[min(p_end_char, len(char_to_token) - 1)]
                # Ensure monotonic non-empty span
                if p_end_tok <= p_start_tok and p_start_tok < total_tokens:
                    p_end_tok = min(total_tokens, p_start_tok + 1)

                paragraph_spans.append(
                    StructuralSpan(
                        span_type=BoundaryType.PARAGRAPH,
                        start_token=p_start_tok,
                        end_token=p_end_tok,
                        start_char=p_start_char,
                        end_char=p_end_char,
                        text=p_text_clean,
                        metadata={"paragraph_idx": p_idx},
                    )
                )
        else:
            # Fallback for paragraphs: sentence-based or token chunk fallback
            sentences = re.split(r"(?<=[.!?])\s+", text)
            if len(sentences) > 1:
                search_pos = 0
                for s_idx, s_text in enumerate(sentences):
                    s_clean = s_text.strip()
                    if not s_clean:
                        continue
                    s_start_char = text.find(s_clean, search_pos)
                    if s_start_char == -1:
                        s_start_char = search_pos
                    s_end_char = s_start_char + len(s_clean)
                    search_pos = s_end_char

                    p_start_tok = char_to_token[s_start_char]
                    p_end_tok = char_to_token[min(s_end_char, len(char_to_token) - 1)]
                    if p_end_tok <= p_start_tok and p_start_tok < total_tokens:
                        p_end_tok = min(total_tokens, p_start_tok + 1)

                    paragraph_spans.append(
                        StructuralSpan(
                            span_type=BoundaryType.PARAGRAPH,
                            start_token=p_start_tok,
                            end_token=p_end_tok,
                            start_char=s_start_char,
                            end_char=s_end_char,
                            text=s_clean,
                            metadata={"paragraph_idx": s_idx, "fallback": "sentence"},
                        )
                    )
            else:
                # Continuous text chunk fallback
                for i in range(0, total_tokens, self.fallback_paragraph_tokens):
                    end_tok = min(total_tokens, i + self.fallback_paragraph_tokens)
                    paragraph_spans.append(
                        StructuralSpan(
                            span_type=BoundaryType.PARAGRAPH,
                            start_token=i,
                            end_token=end_tok,
                            text=f"Chunk [{i}:{end_tok}]",
                            metadata={"fallback": "token_chunk"},
                        )
                    )

        # 2. SECTION DETECTION
        # Find explicit section heading positions
        heading_matches = []
        for pat in self.HEADING_PATTERNS:
            for m in pat.finditer(text):
                heading_matches.append((m.start(), m.end(), m.group(0).strip()))

        # Deduplicate and sort by character start
        heading_matches = sorted(
            list({start: (start, end, title) for start, end, title in heading_matches}.values()),
            key=lambda x: x[0],
        )

        section_spans = []
        has_real_sections = len(heading_matches) >= 2

        if has_real_sections:
            for s_idx in range(len(heading_matches)):
                start_c = heading_matches[s_idx][0]
                end_c = heading_matches[s_idx + 1][0] if s_idx + 1 < len(heading_matches) else len(text)
                sec_text = text[start_c:end_c].strip()
                title = heading_matches[s_idx][2]

                s_start_tok = char_to_token[start_c]
                s_end_tok = char_to_token[min(end_c, len(char_to_token) - 1)]
                if s_end_tok <= s_start_tok and s_start_tok < total_tokens:
                    s_end_tok = min(total_tokens, s_start_tok + 1)

                section_spans.append(
                    StructuralSpan(
                        span_type=BoundaryType.SECTION,
                        start_token=s_start_tok,
                        end_token=s_end_tok,
                        start_char=start_c,
                        end_char=end_c,
                        title=title,
                        text=sec_text,
                        metadata={"section_idx": s_idx, "is_explicit": True},
                    )
                )
        else:
            # Fallback for sections:
            # If multiple paragraphs exist: cluster every N paragraphs into 1 section
            if len(paragraph_spans) >= self.paragraphs_per_section_fallback:
                n = self.paragraphs_per_section_fallback
                for s_idx, i in enumerate(range(0, len(paragraph_spans), n)):
                    cluster = paragraph_spans[i : i + n]
                    s_start_tok = cluster[0].start_token
                    s_end_tok = cluster[-1].end_token
                    section_spans.append(
                        StructuralSpan(
                            span_type=BoundaryType.SECTION,
                            start_token=s_start_tok,
                            end_token=s_end_tok,
                            title=f"Section Cluster {s_idx + 1}",
                            metadata={"section_idx": s_idx, "fallback": "paragraph_cluster"},
                        )
                    )
            else:
                # Token budget fallback for sections
                for s_idx, i in enumerate(range(0, total_tokens, self.fallback_section_tokens)):
                    end_tok = min(total_tokens, i + self.fallback_section_tokens)
                    section_spans.append(
                        StructuralSpan(
                            span_type=BoundaryType.SECTION,
                            start_token=i,
                            end_token=end_tok,
                            title=f"Section Chunk {s_idx + 1}",
                            metadata={"fallback": "token_chunk"},
                        )
                    )

        # 3. DOCUMENT LEVEL
        doc_span = StructuralSpan(
            span_type=BoundaryType.DOCUMENT,
            start_token=0,
            end_token=total_tokens,
            start_char=0,
            end_char=len(text),
            text=text,
            metadata={"total_paragraphs": len(paragraph_spans), "total_sections": len(section_spans)},
        )

        return DocumentStructure(
            text=text,
            total_tokens=total_tokens,
            paragraphs=paragraph_spans,
            sections=section_spans,
            document_span=doc_span,
            metadata={
                "has_real_sections": has_real_sections,
                "num_paragraphs": len(paragraph_spans),
                "num_sections": len(section_spans),
            },
        )

    @staticmethod
    def generate_random_boundaries(total_tokens: int, num_boundaries: int, seed: int = 42) -> List[int]:
        """
        Generates random token boundaries for Ablation A3.
        Matches the exact count of update events as real structural boundaries,
        distinguishing structure from arbitrary update frequency.
        """
        if total_tokens <= 1 or num_boundaries <= 0:
            return [total_tokens]

        rng = random.Random(seed)
        # Sample unique points strictly inside (1, total_tokens - 1)
        valid_points = list(range(2, total_tokens))
        k = min(len(valid_points), max(0, num_boundaries - 1))
        sampled = rng.sample(valid_points, k)
        sampled.append(total_tokens)
        sampled.sort()
        return sampled
