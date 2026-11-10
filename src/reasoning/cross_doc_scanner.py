"""
Cross-Document Contradiction Scanner for SA-CMS.
Detects factual discrepancies, numerical conflicts, and polarity disagreements across multiple documents.
100% Strict Offline, Zero External APIs, Deterministic Rule-Based Auditor.
"""

import re
import time
import unicodedata
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for canonical metric alignment."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


class ContradictionType(str, Enum):
    NUMERICAL_DISCREPANCY = "numerical_discrepancy"  # Disagreeing numbers for identical metrics
    POLARITY_CONFLICT = "polarity_conflict"          # Affirmative vs negative assertions
    CLAIM_INCONSISTENCY = "claim_inconsistency"      # Divergent qualitative conclusions


@dataclass
class ContradictionAlert:
    """A detected contradiction between two distinct document sources."""
    topic: str
    contradiction_type: ContradictionType
    severity: str  # "HIGH", "MEDIUM"
    claim_a: Dict[str, Any]
    claim_b: Dict[str, Any]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "contradiction_type": self.contradiction_type.value,
            "severity": self.severity,
            "claim_a": self.claim_a,
            "claim_b": self.claim_b,
            "explanation": self.explanation,
        }


@dataclass
class ContradictionScanReport:
    """Consolidated report across all scanned documents."""
    total_documents: int
    contradictions_found: int
    alerts: List[ContradictionAlert] = field(default_factory=list)
    scan_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "contradictions_found": self.contradictions_found,
            "alerts": [a.to_dict() for a in self.alerts],
            "scan_duration_ms": round(self.scan_duration_ms, 2),
        }


class CrossDocumentContradictionScanner:
    """
    Audits document collections for conflicting factual assertions.
    """

    def __init__(self, tolerance_ratio: float = 0.05):
        self.tolerance_ratio = tolerance_ratio  # 5% acceptable margin for numerical noise

    def _normalize_metric_subject(self, subj: str) -> str:
        s = strip_accents(subj.lower().strip())
        s = re.sub(r"[^\w\s]", " ", s)
        words = [w for w in s.split() if w not in ("cua", "cac", "nhung", "he", "thong", "mo", "hinh")]
        return "_".join(words)

    def extract_numerical_claims(
        self,
        text: str,
        document_id: str,
        section_title: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Extracts quantitative claims: (Subject, Value, Unit, RawSentence).
        """
        claims = []
        # Pattern: (Subject) ... (đạt|giảm|tăng|là) (\d+(?:[.,]\d+)?)\s*(%|ms|s|token|lần|tỷ|triệu)?
        pattern = re.compile(
            r"([\w\s\-\_]{3,35})\s+(?:đạt mức|đạt tỷ lệ|đạt|giảm|tăng|duy trì|là)\s+(\d+(?:[.,]\d+)?)\s*(%|ms|s|token|lần|tỷ|triệu)?",
            re.IGNORECASE,
        )

        # Protect decimals before splitting sentences
        protected = re.sub(r"(?<=\d)\.(?=\d)", "\u0001", text)
        protected = re.sub(r"(?<=\d),(?=\d)", "\u0002", protected)
        sentences = re.split(r"(?<=[.!?])\s+|\n+", protected)

        for sent in sentences:
            clean_sent = sent.replace("\u0001", ".").replace("\u0002", ",").strip()
            if len(clean_sent) < 15:
                continue

            for match in pattern.finditer(clean_sent):
                raw_subj = match.group(1).strip(" ,.;:-")
                val_str = match.group(2).replace(",", ".")
                unit = match.group(3) or ""

                try:
                    val = float(val_str)
                except ValueError:
                    continue

                subj_key = self._normalize_metric_subject(raw_subj)
                if len(subj_key) >= 4:
                    claims.append({
                        "subject": raw_subj,
                        "subject_key": subj_key,
                        "value": val,
                        "unit": unit,
                        "sentence": clean_sent,
                        "document_id": document_id,
                        "section_title": section_title,
                    })

        return claims

    def extract_polarity_claims(
        self,
        text: str,
        document_id: str,
        section_title: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Extracts affirmative vs negative polarity claims on key topics.
        """
        claims = []
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

        pos_cues = ["hoàn toàn khả thi", "khả thi", "thành công", "hỗ trợ đầy đủ", "tối ưu hiệu quả", "đáp ứng tốt"]
        neg_cues = ["chưa khả thi", "không khả thi", "thất bại", "không hỗ trợ", "chưa thể tối ưu", "không đáp ứng"]

        for sent in sentences:
            clean = sent.strip()
            if len(clean) < 15:
                continue
            norm = strip_accents(clean.lower())

            # Check negative cues first (longer/more specific)
            found_neg = [c for c in neg_cues if strip_accents(c) in norm]
            if found_neg:
                # Extract surrounding topic (words before the cue)
                cue = found_neg[0]
                idx = norm.find(strip_accents(cue))
                topic = clean[:idx].strip(" ,.;:-")
                claims.append({
                    "topic": topic[-40:] if len(topic) > 40 else topic,
                    "polarity": "NEGATIVE",
                    "cue": cue,
                    "sentence": clean,
                    "document_id": document_id,
                    "section_title": section_title,
                })
                continue

            found_pos = [c for c in pos_cues if strip_accents(c) in norm]
            if found_pos:
                cue = found_pos[0]
                idx = norm.find(strip_accents(cue))
                topic = clean[:idx].strip(" ,.;:-")
                claims.append({
                    "topic": topic[-40:] if len(topic) > 40 else topic,
                    "polarity": "POSITIVE",
                    "cue": cue,
                    "sentence": clean,
                    "document_id": document_id,
                    "section_title": section_title,
                })

        return claims

    def scan_documents(
        self,
        docs: List[Dict[str, Any]],
    ) -> ContradictionScanReport:
        """
        Scans a list of documents: [{ 'document_id': str, 'text': str, 'title': Optional[str] }].
        """
        t_start = time.perf_counter()
        alerts: List[ContradictionAlert] = []

        if len(docs) < 2:
            return ContradictionScanReport(
                total_documents=len(docs),
                contradictions_found=0,
                alerts=[],
                scan_duration_ms=0.0,
            )

        all_num_claims: List[Dict[str, Any]] = []
        all_pol_claims: List[Dict[str, Any]] = []

        for d in docs:
            d_id = d.get("document_id", "doc")
            raw_text = d.get("text", "")
            title = d.get("title", "")
            all_num_claims.extend(self.extract_numerical_claims(raw_text, document_id=d_id, section_title=title))
            all_pol_claims.extend(self.extract_polarity_claims(raw_text, document_id=d_id, section_title=title))

        # 1. Compare numerical claims across DIFFERENT documents
        for i in range(len(all_num_claims)):
            for j in range(i + 1, len(all_num_claims)):
                c1 = all_num_claims[i]
                c2 = all_num_claims[j]

                if c1["document_id"] == c2["document_id"]:
                    continue

                # Check subject match
                k1, k2 = c1["subject_key"], c2["subject_key"]
                if k1 == k2 or (len(k1) > 6 and k1 in k2) or (len(k2) > 6 and k2 in k1):
                    val1, val2 = c1["value"], c2["value"]
                    # Calculate relative difference
                    max_val = max(abs(val1), abs(val2), 1e-5)
                    rel_diff = abs(val1 - val2) / max_val

                    if rel_diff > self.tolerance_ratio:
                        explanation = (
                            f"Phát hiện mâu thuẫn số liệu về '{c1['subject']}': "
                            f"Tài liệu [{c1['document_id']}] báo cáo {val1}{c1['unit']}, "
                            f"nhưng tài liệu [{c2['document_id']}] báo cáo {val2}{c2['unit']} "
                            f"(độ lệch: {round(rel_diff * 100, 1)}%)."
                        )
                        alerts.append(ContradictionAlert(
                            topic=c1["subject"],
                            contradiction_type=ContradictionType.NUMERICAL_DISCREPANCY,
                            severity="HIGH" if rel_diff > 0.20 else "MEDIUM",
                            claim_a=c1,
                            claim_b=c2,
                            explanation=explanation,
                        ))

        # 2. Compare polarity claims across DIFFERENT documents
        for i in range(len(all_pol_claims)):
            for j in range(i + 1, len(all_pol_claims)):
                p1 = all_pol_claims[i]
                p2 = all_pol_claims[j]

                if p1["document_id"] == p2["document_id"]:
                    continue

                if p1["polarity"] != p2["polarity"]:
                    t1_norm = strip_accents(p1["topic"].lower())
                    t2_norm = strip_accents(p2["topic"].lower())

                    # Check topic overlap
                    words1 = set(t1_norm.split())
                    words2 = set(t2_norm.split())
                    common = words1 & words2
                    if len(common) >= 2:
                        explanation = (
                            f"Phát hiện xung đột kết luận về '{p1['topic']}': "
                            f"Tài liệu [{p1['document_id']}] khẳng định {p1['polarity']} ('{p1['cue']}'), "
                            f"trong khi [{p2['document_id']}] kết luận {p2['polarity']} ('{p2['cue']}')."
                        )
                        alerts.append(ContradictionAlert(
                            topic=p1["topic"],
                            contradiction_type=ContradictionType.POLARITY_CONFLICT,
                            severity="HIGH",
                            claim_a=p1,
                            claim_b=p2,
                            explanation=explanation,
                        ))

        duration = (time.perf_counter() - t_start) * 1000.0
        return ContradictionScanReport(
            total_documents=len(docs),
            contradictions_found=len(alerts),
            alerts=alerts,
            scan_duration_ms=duration,
        )

