"""
Reflective Grounding Critic: Verifies answers against retrieved citations
to eliminate hallucinations, over-extrapolations, and unsupported numerical claims.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re


@dataclass
class ClaimVerificationResult:
    claim_text: str
    is_supported: bool
    supporting_citation_index: Optional[int] = None
    overlap_score: float = 0.0
    unsupported_entities: List[str] = field(default_factory=list)


@dataclass
class CriticReport:
    is_faithful: bool
    faithfulness_score: float
    claims: List[ClaimVerificationResult] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)


class GroundingCritic:
    """
    Reflective Critic that validates generated claims against source citations.
    Follows Ponytail principles: pure deterministic verification without invoking external LLMs.
    """

    NUMERIC_ENTITY_PATTERN = re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:%|phần trăm|triệu|tỷ|token[s]?|ms|s|gb|mb|năm)?(?:\b|(?<=%))",
        re.IGNORECASE,
    )

    METRIC_CONTRAST_PATTERN = re.compile(
        r"\b(\d+(?:[.,]\d+)?)\s*(%|phần trăm|mb|gb|ms|tokens?|s)(?:\b|(?<=%))",
        re.IGNORECASE,
    )

    UNIT_NORMALIZATION = {
        "%": "%",
        "phần trăm": "%",
        "mb": "mb",
        "gb": "gb",
        "ms": "ms",
        "s": "s",
        "token": "tokens",
        "tokens": "tokens",
    }

    def _is_derived_number_supported(self, num_token: str, citation_text: str) -> bool:
        """
        Validates if a numeric token in the answer is derived via simple arithmetic
        (difference |a - b| or sum a + b) from any pair of numbers present in the citation text.
        Tolerance: 1e-2.
        """
        clean_num = num_token.strip()
        m = re.search(r"(\d+(?:[.,]\d+)?)", clean_num)
        if not m:
            return False
        try:
            target_val = round(abs(float(m.group(1).replace(",", "."))), 2)
        except ValueError:
            return False

        raw_nums = re.findall(r"(?<![a-zA-Z\d.,])(\d+(?:[.,]\d+)?)(?![.,\d])", citation_text)
        nums = []
        for rn in raw_nums:
            try:
                nums.append(float(rn.replace(",", ".")))
            except ValueError:
                continue

        if len(nums) < 2:
            return False

        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                a, b = nums[i], nums[j]
                diff = round(abs(a - b), 2)
                if abs(target_val - diff) <= 0.010001:
                    return True
                sm = round(a + b, 2)
                if abs(target_val - sm) <= 0.010001:
                    return True

        return False

    def detect_metric_contrasts(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scans across multiple citations for numbers sharing the same unit
        (%, mb, gb, ms, s, tokens) across different sections/documents,
        returning structured contrast items:
        {"unit": unit, "values": [...], "delta": round(max_v - min_v, 2), "citations": [...]}
        """
        if not citations or len(citations) < 2:
            return []

        unit_entries: Dict[str, List[Dict[str, Any]]] = {}

        for idx, cit in enumerate(citations):
            text = cit.get("text", "") or cit.get("content", "")
            if not text:
                continue

            cit_id = cit.get("citation_index")
            if cit_id is None:
                cit_id = cit.get("id") or (idx + 1)

            doc_title = cit.get("document_title") or cit.get("title") or f"doc_{idx}"
            sec_title = cit.get("section_title") or cit.get("section") or f"sec_{idx}"
            source_key = (doc_title, sec_title, cit_id)

            matches = self.METRIC_CONTRAST_PATTERN.findall(text)
            seen_for_cit = set()
            for num_str, unit_raw in matches:
                norm_unit = self.UNIT_NORMALIZATION.get(unit_raw.lower(), unit_raw.lower())
                try:
                    val = round(float(num_str.replace(",", ".")), 2)
                except ValueError:
                    continue

                dedup_key = (norm_unit, val)
                if dedup_key in seen_for_cit:
                    continue
                seen_for_cit.add(dedup_key)

                if norm_unit not in unit_entries:
                    unit_entries[norm_unit] = []
                unit_entries[norm_unit].append({
                    "value": val,
                    "citation_id": cit_id,
                    "source_key": source_key,
                })

        contrasts: List[Dict[str, Any]] = []

        for unit, entries in unit_entries.items():
            cit_ids = set(e["citation_id"] for e in entries)
            source_keys = set(e["source_key"] for e in entries)

            # Must span across different citations / sections / documents
            if len(cit_ids) < 2 and len(source_keys) < 2:
                continue

            values = [e["value"] for e in entries]
            max_v = max(values)
            min_v = min(values)
            delta = round(max_v - min_v, 2)

            ordered_cits = []
            for e in entries:
                cid = e["citation_id"]
                if cid not in ordered_cits:
                    ordered_cits.append(cid)

            contrasts.append({
                "unit": unit,
                "values": values,
                "delta": delta,
                "citations": ordered_cits,
            })

        return contrasts

    def verify_answer(self, answer: str, citations: List[Dict[str, Any]]) -> CriticReport:
        if not citations:
            return CriticReport(is_faithful=False, faithfulness_score=0.0, unsupported_claims=[answer])

        # Aggregate citation text for fast verification
        all_citation_text = " ".join(c.get("text", "") for c in citations).lower()
        citation_tokens = set(re.findall(r"\w+", all_citation_text))

        # Filter out meta-headers and sources from claims
        clean_text = answer
        for header in [
            "### Nguồn tham khảo",
            "💡 **Câu hỏi tự học tiếp theo:**",
            "💡 **Tổng hợp & Đúc kết chuyên sâu:**",
            "🧠 **Tiến trình nơ-ron",
        ]:
            if header in clean_text:
                clean_text = clean_text.split(header)[0]

        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", clean_text)
            if len(s.strip()) >= 15
            and not s.strip().startswith(("#", "-", "*", "🧠", "💡", "📌", "👉", "🔗"))
            and not s.strip().endswith(":")
        ]

        if not sentences:
            return CriticReport(is_faithful=True, faithfulness_score=1.0)

        claims_res: List[ClaimVerificationResult] = []
        unsupported: List[str] = []

        for sentence in sentences:
            s_low = sentence.lower()
            s_tokens = set(re.findall(r"\w+", s_low))
            if not s_tokens:
                continue

            # 1. Check numbers/metrics (literal presence or derived arithmetic)
            numbers_in_sentence = self.NUMERIC_ENTITY_PATTERN.findall(sentence)
            unsupported_nums = []
            for n in numbers_in_sentence:
                n_clean = n.strip()
                if n_clean.lower() in all_citation_text:
                    continue
                if self._is_derived_number_supported(n_clean, all_citation_text):
                    continue
                unsupported_nums.append(n_clean)

            # 2. Token overlap score
            overlap_count = len(s_tokens.intersection(citation_tokens))
            overlap_score = overlap_count / max(1, len(s_tokens))

            # Supported if reasonable token overlap AND no hallucinated numbers
            is_supported = (overlap_score >= 0.30) and (len(unsupported_nums) == 0)

            if not is_supported:
                unsupported.append(sentence)

            claims_res.append(
                ClaimVerificationResult(
                    claim_text=sentence,
                    is_supported=is_supported,
                    overlap_score=overlap_score,
                    unsupported_entities=unsupported_nums,
                )
            )

        supported_count = sum(1 for c in claims_res if c.is_supported)
        score = supported_count / max(1, len(claims_res))
        is_faithful = (score >= 0.70) and (len(unsupported) == 0)

        return CriticReport(
            is_faithful=is_faithful,
            faithfulness_score=score,
            claims=claims_res,
            unsupported_claims=unsupported,
        )

    def correct_answer(self, answer: str, report: CriticReport) -> str:
        """
        Reflectively corrects an answer by appending an explicit boundary advisory
        or pruning unsubstantiated claims.
        """
        if report.is_faithful or not report.unsupported_claims:
            return answer

        # If answer contains hallucinated metrics, append an advisory warning
        advisories = []
        for claim in report.claims:
            if not claim.is_supported and claim.unsupported_entities:
                advisories.append(
                    f"*(Lưu ý đối soát: Chi tiết '{', '.join(claim.unsupported_entities)}' không có trong tài liệu đối chiếu)*"
                )

        if advisories:
            return f"{answer}\n\n" + "\n".join(advisories)

        return f"{answer}\n\n*(Lưu ý đối soát: Một số nhận định trên không có trong tài liệu trực tiếp)*"

