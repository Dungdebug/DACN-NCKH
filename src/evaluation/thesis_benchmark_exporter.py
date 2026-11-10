"""
Thesis Benchmark Exporter (Ticket 7.6).
Compiles empirical benchmarking metrics comparing proposed SA-CMS against baselines
(ICL B1, RAG B2, Cartridges B3, Hope-Attention B5, SA-CMS Hybrid P2).
Generates publication-ready LaTeX tables for thesis chapters 4 & 5.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any


@dataclass
class BenchmarkScorecard:
    """Evaluation scorecard for a single method configuration."""
    method: str
    token_savings_pct: float
    grounding_acc_pct: float
    latency_ms: float
    vram_mb: int
    refusal_precision_pct: float = 95.0


class ThesisBenchmarkExporter:
    """
    Exports empirical benchmarks to publication-ready LaTeX tables.
    """

    def export_rq_latex_table(self, scorecards: List[BenchmarkScorecard], label: str = "tab:rq_comparison") -> str:
        """
        Exports the master comparison table (RQ1 - RQ5) in LaTeX format.
        """
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{So sánh hiệu năng tổng thể giữa các phương pháp tiếp cận (RQ1--RQ5)}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lcccc}",
            "\\hline",
            "\\textbf{Phương pháp} & \\textbf{Tiết kiệm Token (\\%)} & \\textbf{Độ chính xác (\\%)} & \\textbf{Độ trễ (ms)} & \\textbf{VRAM (MB)} \\\\",
            "\\hline",
        ]

        for sc in scorecards:
            lines.append(
                f"{sc.method} & {sc.token_savings_pct:.1f}\\% & {sc.grounding_acc_pct:.1f}\\% & {sc.latency_ms:.1f} & {sc.vram_mb} \\\\"
            )

        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ])

        return "\n".join(lines)

    def export_ablation_latex_table(self, ablation_records: List[Dict[str, Any]], label: str = "tab:ablation_study") -> str:
        """
        Exports the Section 7.4 Ablation Study table in LaTeX format.
        """
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Kết quả thực nghiệm bóc tách thành phần kiến trúc SA-CMS (Mục 7.4 Đề cương)}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lcc}",
            "\\hline",
            "\\textbf{Cấu hình thử nghiệm} & \\textbf{F1-Score (\\%)} & \\textbf{Token Tiết kiệm} \\\\",
            "\\hline",
        ]

        for rec in ablation_records:
            config = rec.get("config", "")
            f1 = rec.get("f1_score", 0.0)
            tokens = rec.get("tokens_saved", "")
            lines.append(f"{config} & {f1:.1f}\\% & {tokens} \\\\")

        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ])

        return "\n".join(lines)

