"""
Generate Phase 4.3 Neutral Scientific Figures:
Saved in results/phase4_3/figures/
- fig_rq1_qasper.png
- fig_rq1_longhealth.png
- fig_rq1_mkniah.png
- fig_rq2_p1_vs_b5.png
- fig_rq3_f1.png
- fig_rq3_faithfulness.png
- fig_rq3_refusal.png
- fig_rq5_latency.png
- fig_rq5_vram.png
- fig_checkpoint_size.png
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import csv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT_DIR / "results" / "phase4_3" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Palette
COLORS = {
    "B1": "#4A90E2",
    "B2": "#50E3C2",
    "B4": "#F5A623",
    "B5": "#9013FE",
    "P1": "#D0021B",
    "P2": "#417505",
    "A2": "#7ED321",
    "diff": "#4A4A4A"
}

def plot_rq1():
    # Load 01_rq1_summary.csv
    rq1_csv = ROOT_DIR / "results" / "phase4_3" / "tables" / "01_rq1_summary.csv"
    data = {}
    with open(rq1_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            b = r["dataset"]
            if b not in data:
                data[b] = {}
            data[b][r["method"]] = {
                "mean": float(r["mean"]),
                "ci_low": float(r["ci_95_bootstrap_lower"]),
                "ci_high": float(r["ci_95_bootstrap_upper"])
            }
            
    methods = ["B1", "B4", "B5", "P1"]
    
    # 1. QASPER F1
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    means = [data["QASPER"][m]["mean"] for m in methods]
    lowers = [data["QASPER"][m]["ci_low"] for m in methods]
    uppers = [data["QASPER"][m]["ci_high"] for m in methods]
    err_low = [m - l for m, l in zip(means, lowers)]
    err_high = [u - m for m, u in zip(means, uppers)]
    bars = ax.bar(methods, means, yerr=[err_low, err_high], capsize=5, color=[COLORS[m] for m in methods], edgecolor="black", alpha=0.85)
    ax.set_ylabel("Token F1 Score")
    ax.set_title("QASPER Token F1 Score by Method (B1, B4, B5, P1)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.45)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_rq1_qasper.png")
    plt.close()

    # 2. LongHealth Accuracy
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    means = [data["LongHealth"][m]["mean"] for m in methods]
    lowers = [data["LongHealth"][m]["ci_low"] for m in methods]
    uppers = [data["LongHealth"][m]["ci_high"] for m in methods]
    err_low = [m - l for m, l in zip(means, lowers)]
    err_high = [u - m for m, u in zip(means, uppers)]
    bars = ax.bar(methods, means, yerr=[err_low, err_high], capsize=5, color=[COLORS[m] for m in methods], edgecolor="black", alpha=0.85)
    ax.set_ylabel("Multiple-Choice Accuracy")
    ax.set_title("LongHealth Accuracy by Method (B1, B4, B5, P1)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.45)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_rq1_longhealth.png")
    plt.close()

    # 3. MK-NIAH Accuracy
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    means = [data["MK-NIAH"][m]["mean"] for m in methods]
    lowers = [data["MK-NIAH"][m]["ci_low"] for m in methods]
    uppers = [data["MK-NIAH"][m]["ci_high"] for m in methods]
    err_low = [m - l for m, l in zip(means, lowers)]
    err_high = [u - m for m, u in zip(means, uppers)]
    bars = ax.bar(methods, means, yerr=[err_low, err_high], capsize=5, color=[COLORS[m] for m in methods], edgecolor="black", alpha=0.85)
    ax.set_ylabel("Retrieval Accuracy")
    ax.set_title("MK-NIAH Retrieval Accuracy by Method", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.6)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_rq1_mkniah.png")
    plt.close()

def plot_rq2():
    # 4. P1 vs B5 Mean Diff across Pooled, QASPER, LongHealth
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    categories = ["Pooled (N=90)", "QASPER (N=30)", "LongHealth (N=60)"]
    diffs = [0.0093, -0.0055, 0.0167]
    ci_low = [-0.0051, -0.0186, 0.0]
    ci_high = [0.0340, 0.0050, 0.0500]
    
    err_low = [d - l for d, l in zip(diffs, ci_low)]
    err_high = [u - d for d, u in zip(diffs, ci_high)]
    
    colors = ["#2B6CB0", "#C53030", "#2F855A"]
    bars = ax.bar(categories, diffs, yerr=[err_low, err_high], capsize=6, color=colors, edgecolor="black", alpha=0.85, width=0.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Mean Difference (P1 - B5)")
    ax.set_title("RQ2: P1 vs B5 Paired Differences Across Subsets (with 95% Bootstrap CI)", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.03, 0.07)
    
    for bar in bars:
        yval = bar.get_height()
        y_pos = yval + 0.006 if yval >= 0 else yval - 0.01
        ax.text(bar.get_x() + bar.get_width()/2.0, y_pos, f"{yval:+.4f}", ha="center", va="bottom" if yval>=0 else "top", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_rq2_p1_vs_b5.png")
    plt.close()

def plot_rq3():
    methods = ["B1", "B2", "B5", "P1", "P2"]
    f1_vals = [0.0048, 0.1596, 0.0502, 0.0753, 0.1596]
    faith_vals = [0.0, 95.7, 0.0, 0.0, 98.92]
    corr_refusal = [0.0, 76.0, 0.0, 0.0, 76.0]
    false_refusal = [0.0, 0.0, 0.0, 0.0, 0.0]

    # 5. RQ3 F1
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
    bars = ax.bar(methods, f1_vals, color=[COLORS[m] for m in methods], edgecolor="black", alpha=0.85)
    ax.set_ylabel("Token F1 Score")
    ax.set_title("Context-Evicted QA Token F1 Score Across Methods", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.25)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f"{yval:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_rq3_f1.png")
    plt.close()

    # 6. RQ3 Faithfulness
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
    bars = ax.bar(methods, faith_vals, color=[COLORS[m] for m in methods], edgecolor="black", alpha=0.85)
    ax.set_ylabel("Faithfulness (%) [Citation-Supported Rate]")
    ax.set_title("Citation-Supported Faithfulness Rate on Generated Answers", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_rq3_faithfulness.png")
    plt.close()

    # 7. RQ3 Refusal Rates
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    x = np.arange(len(methods))
    width = 0.35
    rects1 = ax.bar(x - width/2, corr_refusal, width, label="Correct Refusal (%)", color="#3182CE", edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x + width/2, false_refusal, width, label="False Refusal (%)", color="#E53E3E", edgecolor="black", alpha=0.85)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Correct Refusal and False Refusal Rates Across Methods", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc="upper left")
    ax.set_ylim(0, 100)
    for rect in rects1:
        h = rect.get_height()
        if h > 0:
            ax.text(rect.get_x() + rect.get_width()/2.0, h + 2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_rq3_refusal.png")
    plt.close()

def plot_rq5():
    # 8. RQ5 Latency Breakdown
    methods = ["B1", "B2", "B4", "B5", "P1", "P2"]
    gen_time = [5168.39, 4117.37, 978.59, 1009.40, 1036.54, 4247.96]
    ret_time = [0.0, 2.13, 0.0, 0.0, 0.0, 1.35]
    
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    x = np.arange(len(methods))
    ax.bar(methods, gen_time, label="Generation Latency (ms)", color="#4A5568", edgecolor="black", alpha=0.85)
    ax.bar(methods, ret_time, bottom=gen_time, label="Retrieval Latency (ms)", color="#ED8936", edgecolor="black", alpha=0.85)
    ax.set_ylabel("Total Latency (ms)")
    ax.set_title("End-to-End Query Latency Breakdown by Component", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 6200)
    for i, m in enumerate(methods):
        tot = gen_time[i] + ret_time[i]
        ax.text(i, tot + 120, f"{tot:.1f}ms", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_rq5_latency.png")
    plt.close()

    # 9. Peak VRAM
    vram_vals = [364.69, 353.31, 310.67, 317.73, 331.24, 369.48]
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
    bars = ax.bar(methods, vram_vals, color=[COLORS.get(m, "#4A5568") for m in methods], edgecolor="black", alpha=0.85)
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title("Peak VRAM Consumption During Inference", fontsize=11, fontweight="bold")
    ax.set_ylim(250, 420)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 3, f"{yval:.1f}MB", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_rq5_vram.png")
    plt.close()

    # 10. Checkpoint Size
    all_methods = ["B1", "B2", "B4", "A2", "B5", "P1", "P2"]
    sizes = [0.0, 0.0, 6.77, 13.54, 20.28, 20.28, 20.28]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    bars = ax.bar(all_methods, sizes, color=["#A0AEC0", "#A0AEC0", "#ED8936", "#48BB78", "#9F7AEA", "#E53E3E", "#38A169"], edgecolor="black", alpha=0.85)
    ax.set_ylabel("Checkpoint Size on Disk (MB)")
    ax.set_title("Adapter Storage Footprint by Model Architecture", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 25)
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{yval:.2f}MB", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_checkpoint_size.png")
    plt.close()

if __name__ == "__main__":
    print("Generating Phase 4.3 Neutral Scientific Figures...")
    plot_rq1()
    plot_rq2()
    plot_rq3()
    plot_rq5()
    print("All 10 figures successfully generated in results/phase4_3/figures/!")
