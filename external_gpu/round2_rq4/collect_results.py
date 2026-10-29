#!/usr/bin/env python3
"""
Statistical Aggregator and Report Generator for RQ4.
Synthesizes sequential retention metrics across milestones D0, D0+5, D0+10, D0+20.
Generates rq4_sequential_verdict.json.
"""

import sys
import json
from pathlib import Path

def collect_results():
    print("=" * 70)
    print("RQ4 SEQUENTIAL RETENTION AND FORGETTING SYNTHESIS")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent
    res_file = base_dir / "sequential_results.json"
    
    if not res_file.exists():
        print(f"[FAIL] Missing {res_file.name}. Run sequential experiment first.")
        return False

    with open(res_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    evals = data.get("milestone_evaluations", {})
    summary_table = []

    for m in ["D0", "D0+5", "D0+10", "D0+20"]:
        if m in evals:
            rec = evals[m]
            summary_table.append({
                "milestone": m,
                "docs_ingested": rec["documents_ingested_so_far"],
                "acc_before": rec["accuracy_before"],
                "acc_after": rec["accuracy_after"],
                "forgetting_fk": rec["forgetting_fk"],
                "retention_pct": rec["retention_rate_pct"],
                "new_doc_acc": rec["new_document_accuracy"],
                "snapshot_size_mb": rec["snapshot_size_mb"]
            })

    # Summary analysis
    final_retention = summary_table[-1]["retention_pct"] if summary_table else 0.0
    final_forgetting = summary_table[-1]["forgetting_fk"] if summary_table else 1.0

    verdict = {
        "metadata": {
            "title": "RQ4 Sequential Document Ingestion Synthesis",
            "date": "2026-10-06",
            "milestones_evaluated": len(summary_table),
            "stream_length": 21
        },
        "scoreboard": summary_table,
        "retention_curve_summary": {
            "initial_accuracy_d0": summary_table[0]["acc_before"] if summary_table else 0.0,
            "final_accuracy_d0_after_20": summary_table[-1]["acc_after"] if summary_table else 0.0,
            "cumulative_forgetting_f20": final_forgetting,
            "final_retention_rate_pct": final_retention
        },
        "scientific_verdict": {
            "status": "VALID_OBSERVED_STREAM",
            "interpretation": f"Sau khi nạp liên tục 20 tài liệu mới, bộ nhớ SA-CMS giữ lại {final_retention}% độ chính xác trên tài liệu ban đầu D0 với độ quên lũy tiến F_20 = {final_forgetting:+.4f}."
        }
    }

    out_file = base_dir / "rq4_sequential_verdict.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"| Milestone | Docs Ingested | Acc Before | Acc After | Forgetting (F_k) | Retention % | New Doc Acc |")
    print(f"|:---------:|:-------------:|:----------:|:---------:|:----------------:|:-----------:|:-----------:|")
    for r in summary_table:
        print(f"| {r['milestone']:9s} | {r['docs_ingested']:13d} | {r['acc_before']:10.4f} | {r['acc_after']:9.4f} | {r['forgetting_fk']:+16.4f} | {r['retention_pct']:10.1f}% | {r['new_doc_acc']:11.4f} |")
    print("=" * 70)
    print(f"Final Retention Rate on D0 after 20 docs: {final_retention}% (F_20 = {final_forgetting:+.4f})")
    print(f"Report saved to: {out_file}")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = collect_results()
    sys.exit(0 if success else 1)
