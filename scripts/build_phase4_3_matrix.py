"""
Build Phase 4.3 Benchmark Matrix:
results/phase4_3/benchmark_matrix.csv
"""

import csv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT_DIR / "results" / "phase4_3" / "benchmark_matrix.csv"

rows = [
    {
        "Method": "B1",
        "RQ1": "COMPLETED",
        "RQ2": "NOT_RUN",
        "RQ3": "COMPLETED",
        "RQ4": "NOT_RUN",
        "RQ5": "COMPLETED",
        "QASPER": "COMPLETED",
        "LongHealth": "COMPLETED",
        "MK-NIAH": "COMPLETED",
        "Vietnamese": "COMPLETED",
        "Status": "COMPLETED",
        "Checkpoint": "None (frozen backbone)",
        "External GPU Required": "False"
    },
    {
        "Method": "B2",
        "RQ1": "NOT_RUN",
        "RQ2": "NOT_RUN",
        "RQ3": "COMPLETED",
        "RQ4": "NOT_RUN",
        "RQ5": "COMPLETED",
        "QASPER": "NOT_RUN",
        "LongHealth": "NOT_RUN",
        "MK-NIAH": "NOT_RUN",
        "Vietnamese": "COMPLETED",
        "Status": "COMPLETED",
        "Checkpoint": "None (frozen backbone)",
        "External GPU Required": "False"
    },
    {
        "Method": "B3",
        "RQ1": "EXCLUDED",
        "RQ2": "EXCLUDED",
        "RQ3": "EXCLUDED",
        "RQ4": "EXCLUDED",
        "RQ5": "EXCLUDED",
        "QASPER": "EXCLUDED",
        "LongHealth": "EXCLUDED",
        "MK-NIAH": "EXCLUDED",
        "Vietnamese": "EXCLUDED",
        "Status": "EXCLUDED",
        "Checkpoint": "None",
        "External GPU Required": "False"
    },
    {
        "Method": "B4",
        "RQ1": "COMPLETED",
        "RQ2": "NOT_RUN",
        "RQ3": "NOT_RUN",
        "RQ4": "NEED_EXTERNAL_GPU",
        "RQ5": "COMPLETED",
        "QASPER": "COMPLETED",
        "LongHealth": "COMPLETED",
        "MK-NIAH": "COMPLETED",
        "Vietnamese": "COMPLETED",
        "Status": "PARTIAL",
        "Checkpoint": "checkpoints/phase4_1/cms_1lvl_seed_{seed}.pt",
        "External GPU Required": "True"
    },
    {
        "Method": "B5",
        "RQ1": "COMPLETED",
        "RQ2": "COMPLETED",
        "RQ3": "COMPLETED",
        "RQ4": "NEED_EXTERNAL_GPU",
        "RQ5": "COMPLETED",
        "QASPER": "COMPLETED",
        "LongHealth": "COMPLETED",
        "MK-NIAH": "COMPLETED",
        "Vietnamese": "COMPLETED",
        "Status": "PARTIAL",
        "Checkpoint": "checkpoints/phase4_1/cms_3lvl_seed_{seed}.pt",
        "External GPU Required": "True"
    },
    {
        "Method": "P1",
        "RQ1": "COMPLETED",
        "RQ2": "COMPLETED",
        "RQ3": "COMPLETED",
        "RQ4": "NEED_EXTERNAL_GPU",
        "RQ5": "COMPLETED",
        "QASPER": "COMPLETED",
        "LongHealth": "COMPLETED",
        "MK-NIAH": "COMPLETED",
        "Vietnamese": "COMPLETED",
        "Status": "PARTIAL",
        "Checkpoint": "checkpoints/phase4_1/cms_3lvl_seed_{seed}.pt",
        "External GPU Required": "True"
    },
    {
        "Method": "P2",
        "RQ1": "NOT_RUN",
        "RQ2": "NOT_RUN",
        "RQ3": "COMPLETED",
        "RQ4": "NOT_RUN",
        "RQ5": "COMPLETED",
        "QASPER": "NOT_RUN",
        "LongHealth": "NOT_RUN",
        "MK-NIAH": "NOT_RUN",
        "Vietnamese": "COMPLETED",
        "Status": "COMPLETED",
        "Checkpoint": "checkpoints/phase4_1/cms_3lvl_seed_{seed}.pt",
        "External GPU Required": "False"
    },
    {
        "Method": "A1",
        "RQ1": "NOT_RUN",
        "RQ2": "NEED_EXTERNAL_GPU",
        "RQ3": "NOT_RUN",
        "RQ4": "NOT_RUN",
        "RQ5": "NOT_RUN",
        "QASPER": "NEED_EXTERNAL_GPU",
        "LongHealth": "NEED_EXTERNAL_GPU",
        "MK-NIAH": "NOT_RUN",
        "Vietnamese": "NOT_RUN",
        "Status": "NEED_EXTERNAL_GPU",
        "Checkpoint": "external_gpu/phase4_2/A1/cms_3lvl_random_seed_{seed}.pt",
        "External GPU Required": "True"
    },
    {
        "Method": "A2",
        "RQ1": "NOT_RUN",
        "RQ2": "COMPLETED",
        "RQ3": "NOT_RUN",
        "RQ4": "NOT_RUN",
        "RQ5": "NOT_RUN",
        "QASPER": "COMPLETED",
        "LongHealth": "COMPLETED",
        "MK-NIAH": "NOT_RUN",
        "Vietnamese": "NOT_RUN",
        "Status": "COMPLETED",
        "Checkpoint": "checkpoints/phase4_1/cms_2lvl_seed_{seed}.pt",
        "External GPU Required": "False"
    },
    {
        "Method": "A3",
        "RQ1": "NOT_RUN",
        "RQ2": "NEED_EXTERNAL_GPU",
        "RQ3": "NOT_RUN",
        "RQ4": "NOT_RUN",
        "RQ5": "NOT_RUN",
        "QASPER": "NEED_EXTERNAL_GPU",
        "LongHealth": "NEED_EXTERNAL_GPU",
        "MK-NIAH": "NOT_RUN",
        "Vietnamese": "NOT_RUN",
        "Status": "NEED_EXTERNAL_GPU",
        "Checkpoint": "external_gpu/phase4_2/A3/cms_3lvl_additive_seed_{seed}.pt",
        "External GPU Required": "True"
    }
]

fieldnames = [
    "Method", "RQ1", "RQ2", "RQ3", "RQ4", "RQ5",
    "QASPER", "LongHealth", "MK-NIAH", "Vietnamese",
    "Status", "Checkpoint", "External GPU Required"
]

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Benchmark matrix written to {OUT_FILE}")
