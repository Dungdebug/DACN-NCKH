"""
Sovereign Offline Subword BPE Tokenizer (Pillar 1 - Self-Built Local LLM).

Pure Python stdlib byte-level BPE tokenizer:
- Zero external APIs, zero third-party model downloads, zero HuggingFace hub calls.
- Base vocabulary covers 4 special tokens + 256 raw UTF-8 bytes (0% <unk> loss on
  Vietnamese diacritics, mathematical formulas, and source code).
- Supports fast frequency-weighted BPE merge training on local academic corpora,
  lossless round-trip encoding/decoding, JSON persistence, and PyTorch tensor output.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch


DEFAULT_ACADEMIC_SEED_CORPUS: Tuple[str, ...] = (
    "Hệ thống WaveCMS sử dụng cây bộ nhớ wavelet đa phân giải để lưu trữ tri thức tài liệu.",
    "Kiến trúc Transformer tự chủ chạy hoàn toàn cục bộ không phụ thuộc API bên ngoài.",
    "Lớp thích nghi hạng thấp Wavelet Low-Rank có dạng W = W_0 + B Theta A.",
    "Định lý Telescoping và bảo toàn năng lượng Parseval đảm bảo tính nhất quán đa phân giải.",
    "Quá trình gỡ bỏ tri thức chính xác Certified Exact Unlearning đạt độ phức tạp O(depth).",
    "Vòng lặp điểm bất động Banach hội tụ đơn điệu khi hệ số co rho nhỏ hơn 1.",
    "Kiểm soát rủi ro bảo giác Conformal Risk Control giới hạn tỷ lệ trả lời sai dưới epsilon.",
    "Mã hóa đối xứng sử dụng chung một khóa bí mật cho cả quá trình mã hóa và giải mã.",
    "Các thuật toán tiêu biểu gồm AES, DES, 3DES, ChaCha20 và các hệ mã cổ điển Caesar, Vigenere.",
    "Cấu trúc dữ liệu cây nhị phân tìm kiếm, đồ thị có hướng và thuật toán quy hoạch động.",
    "Độ phức tạp thời gian và không gian bộ nhớ trong phân tích thuật toán.",
    "Phương pháp đánh giá độ chính xác, độ phủ, điểm F1 và độ trễ suy luận trên CPU.",
)


class TokenBatch(dict):
    """Dictionary with attribute access compatible with standard tokenizer outputs."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to(self, device: Union[str, torch.device]) -> "TokenBatch":
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.to(device)
        return self


class SovereignTokenizer:
    """
    Self-built offline byte-level BPE tokenizer with Vietnamese/English academic support.
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    SPECIAL_TOKENS: Tuple[str, ...] = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
    NUM_SPECIAL = 4
    BYTE_OFFSET = 4
    BASE_VOCAB_SIZE = 260  # 4 special tokens + 256 byte tokens

    _PRETOKEN_RE = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)

    def __init__(
        self,
        merges: Optional[Sequence[Tuple[int, int, int]]] = None,
    ) -> None:
        self.pad_token = self.PAD_TOKEN
        self.bos_token = self.BOS_TOKEN
        self.eos_token = self.EOS_TOKEN
        self.unk_token = self.UNK_TOKEN

        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        # Map token_id -> raw bytes (for non-special tokens)
        self._id_to_bytes: Dict[int, bytes] = {
            self.BYTE_OFFSET + b: bytes([b]) for b in range(256)
        }
        self._special_id_to_str: Dict[int, str] = {
            idx: tok for idx, tok in enumerate(self.SPECIAL_TOKENS)
        }
        self._special_str_to_id: Dict[str, int] = {
            tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)
        }

        # Ordered list of merges: (left_id, right_id, merged_id)
        self.merges: List[Tuple[int, int, int]] = []
        self._merge_ranks: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._bpe_cache: Dict[bytes, Tuple[int, ...]] = {}

        if merges:
            for rank, (left_id, right_id, merged_id) in enumerate(merges):
                self._register_merge(int(left_id), int(right_id), int(merged_id), rank)

    def _register_merge(self, left_id: int, right_id: int, merged_id: int, rank: int) -> None:
        self.merges.append((left_id, right_id, merged_id))
        self._merge_ranks[(left_id, right_id)] = (rank, merged_id)
        self._id_to_bytes[merged_id] = self._id_to_bytes[left_id] + self._id_to_bytes[right_id]

    @property
    def vocab_size(self) -> int:
        return self.BASE_VOCAB_SIZE + len(self.merges)

    @classmethod
    def train_from_texts(
        cls,
        texts: Sequence[str],
        vocab_size: int = 1024,
        min_frequency: int = 2,
    ) -> "SovereignTokenizer":
        """
        Trains byte-level BPE merges purely in Python on local texts.
        """
        tokenizer = cls()
        if vocab_size <= cls.BASE_VOCAB_SIZE or not texts:
            return tokenizer

        word_freqs: Counter = Counter()
        for text in texts:
            if not text:
                continue
            for piece in cls._PRETOKEN_RE.findall(str(text)):
                b_seq = tuple(cls.BYTE_OFFSET + b for b in piece.encode("utf-8"))
                if b_seq:
                    word_freqs[b_seq] += 1

        max_merges = vocab_size - cls.BASE_VOCAB_SIZE
        next_id = cls.BASE_VOCAB_SIZE

        for rank in range(max_merges):
            pair_counts: Counter = Counter()
            for seq, freq in word_freqs.items():
                if len(seq) < 2:
                    continue
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += freq

            if not pair_counts:
                break

            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < min_frequency:
                break

            left_id, right_id = best_pair
            merged_id = next_id
            next_id += 1
            tokenizer._register_merge(left_id, right_id, merged_id, rank)

            # Update word_freqs in place with the new merge
            new_word_freqs: Counter = Counter()
            for seq, freq in word_freqs.items():
                if len(seq) < 2:
                    new_word_freqs[seq] += freq
                    continue
                merged_seq: List[int] = []
                i = 0
                n = len(seq)
                while i < n:
                    if i < n - 1 and seq[i] == left_id and seq[i + 1] == right_id:
                        merged_seq.append(merged_id)
                        i += 2
                    else:
                        merged_seq.append(seq[i])
                        i += 1
                new_word_freqs[tuple(merged_seq)] += freq
            word_freqs = new_word_freqs

        return tokenizer

    @classmethod
    def build_academic_tokenizer(
        cls,
        extra_texts: Optional[Sequence[str]] = None,
        vocab_size: int = 1024,
        min_frequency: int = 2,
    ) -> "SovereignTokenizer":
        """Builds a ready-to-use academic Vietnamese/English BPE tokenizer offline."""
        corpus: List[str] = list(DEFAULT_ACADEMIC_SEED_CORPUS)
        if extra_texts:
            corpus.extend(str(t) for t in extra_texts if t)
        return cls.train_from_texts(corpus, vocab_size=vocab_size, min_frequency=min_frequency)

    def _encode_bytes(self, raw_bytes: bytes) -> Tuple[int, ...]:
        cached = self._bpe_cache.get(raw_bytes)
        if cached is not None:
            return cached

        tokens = [self.BYTE_OFFSET + b for b in raw_bytes]
        if len(tokens) >= 2 and self._merge_ranks:
            while len(tokens) >= 2:
                best_rank = None
                best_pair = None
                best_merged = -1
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    info = self._merge_ranks.get(pair)
                    if info is not None:
                        rank, merged_id = info
                        if best_rank is None or rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                            best_merged = merged_id
                if best_pair is None:
                    break
                left_id, right_id = best_pair
                new_tokens: List[int] = []
                i = 0
                n = len(tokens)
                while i < n:
                    if i < n - 1 and tokens[i] == left_id and tokens[i + 1] == right_id:
                        new_tokens.append(best_merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

        result = tuple(tokens)
        if len(self._bpe_cache) < 16384:
            self._bpe_cache[raw_bytes] = result
        return result

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        """Encodes a UTF-8 string into token IDs without any network or external files."""
        ids: List[int] = []
        if add_special_tokens or add_bos:
            ids.append(self.bos_token_id)

        if text:
            for piece in self._PRETOKEN_RE.findall(str(text)):
                ids.extend(self._encode_bytes(piece.encode("utf-8")))

        if add_special_tokens or add_eos:
            ids.append(self.eos_token_id)

        if (truncation or max_length is not None) and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(
        self,
        token_ids: Union[Sequence[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """Losslessly decodes token IDs back into a UTF-8 string."""
        if isinstance(token_ids, torch.Tensor):
            id_list = token_ids.detach().cpu().tolist()
            if id_list and isinstance(id_list[0], list):
                id_list = id_list[0]
        else:
            id_list = [int(t) for t in token_ids]

        byte_chunks: List[bytes] = []
        for tid in id_list:
            tid_int = int(tid)
            if tid_int in self._special_id_to_str:
                if not skip_special_tokens:
                    byte_chunks.append(self._special_id_to_str[tid_int].encode("utf-8"))
                continue
            chunk = self._id_to_bytes.get(tid_int)
            if chunk is not None:
                byte_chunks.append(chunk)

        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def __call__(
        self,
        text: Union[str, Sequence[str]],
        return_tensors: Optional[str] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        add_special_tokens: bool = False,
    ) -> TokenBatch:
        """Lightweight batch/single encoding compatible with Transformer pipelines."""
        is_single = isinstance(text, str)
        texts = [text] if is_single else list(text)

        encoded_batch = [
            self.encode(
                t,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
            )
            for t in texts
        ]

        max_len = max((len(seq) for seq in encoded_batch), default=0)
        attention_masks: List[List[int]] = []
        padded_ids: List[List[int]] = []

        for seq in encoded_batch:
            pad_len = (max_len - len(seq)) if padding else 0
            padded_ids.append(seq + [self.pad_token_id] * pad_len)
            attention_masks.append([1] * len(seq) + [0] * pad_len)

        if return_tensors == "pt":
            if is_single and not padding:
                ids_tensor = torch.tensor([padded_ids[0]], dtype=torch.long)
                mask_tensor = torch.tensor([attention_masks[0]], dtype=torch.long)
            else:
                ids_tensor = torch.tensor(padded_ids, dtype=torch.long)
                mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
            return TokenBatch({"input_ids": ids_tensor, "attention_mask": mask_tensor})

        if is_single:
            return TokenBatch({"input_ids": padded_ids[0], "attention_mask": attention_masks[0]})
        return TokenBatch({"input_ids": padded_ids, "attention_mask": attention_masks})

    def save(self, path: Union[str, Path]) -> Path:
        """Saves tokenizer merges locally to JSON."""
        target = Path(path)
        if target.is_dir() or not target.suffix:
            target.mkdir(parents=True, exist_ok=True)
            target = target / "sovereign_tokenizer.json"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": 1,
            "vocab_size": self.vocab_size,
            "special_tokens": list(self.SPECIAL_TOKENS),
            "merges": [list(m) for m in self.merges],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SovereignTokenizer":
        """Loads a saved SovereignTokenizer strictly from local JSON."""
        target = Path(path)
        if target.is_dir():
            target = target / "sovereign_tokenizer.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        merges = [tuple(m) for m in payload.get("merges", [])]
        return cls(merges=merges)
