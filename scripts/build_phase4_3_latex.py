"""
Generate Paper-Ready LaTeX Tables for Phase 4.3:
results/phase4_3/latex_tables/
- rq1_main.tex
- rq2_comparison.tex
- rq3_context_evicted.tex
- rq5_cost.tex
- benchmark_status.tex
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
LATEX_DIR = ROOT_DIR / "results" / "phase4_3" / "latex_tables"
LATEX_DIR.mkdir(parents=True, exist_ok=True)

def generate_rq1_latex():
    content = r"""\begin{table*}[t]
\centering
\small
\caption{RQ1 Main Benchmark Evaluation across QASPER, LongHealth, and MK-NIAH (3 Seeds: 42, 43, 44).}
\label{tab:rq1_main}
\begin{tabular}{llcccccc}
\toprule
\textbf{Method} & \textbf{Evaluation Condition} & \multicolumn{2}{c}{\textbf{QASPER ($N=30$)}} & \multicolumn{2}{c}{\textbf{LongHealth ($N=60$)}} & \multicolumn{2}{c}{\textbf{MK-NIAH ($N=300$)}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}
 & & Token $F_1$ [95\% CI] & Target Prob & Accuracy [95\% CI] & Target Prob & Accuracy [95\% CI] & Target Prob \\
\midrule
B1 (ICL Context) & In-Context (Truncated 512) & 0.2590 [0.1930, 0.3435] & 0.1705 & 0.2000 [0.1000, 0.3167] & 0.0077 & 0.4000 [0.3433, 0.4500] & 0.5068 \\
B4 (1-Level CMS) & Context-Evicted (Memory) & 0.1025 [0.0700, 0.1368] & 0.1431 & 0.2333 [0.1333, 0.3500] & 0.1349 & 0.0000 [0.0000, 0.0000] & 0.0479 \\
B5 (3-Level CMS) & Context-Evicted (Memory) & 0.0957 [0.0662, 0.1292] & 0.1434 & 0.2000 [0.1167, 0.3000] & 0.0938 & 0.0000 [0.0000, 0.0000] & 0.0349 \\
P1 (SA-CMS)      & Context-Evicted (Memory) & 0.0896 [0.0548, 0.1287] & 0.1253 & 0.1667 [0.0667, 0.2671] & 0.1308 & 0.0000 [0.0000, 0.0000] & 0.1318 \\
\bottomrule
\end{tabular}
\end{table*}
"""
    with open(LATEX_DIR / "rq1_main.tex", "w", encoding="utf-8") as f:
        f.write(content)
    print("Saved: rq1_main.tex")

def generate_rq2_latex():
    content = r"""\begin{table*}[t]
\centering
\small
\caption{RQ2 Statistical Comparison: Structure-Aligned CMS (P1) vs. Fixed-Token CMS (B5) under Strictly Matched Update Budget.}
\label{tab:rq2_comparison}
\begin{tabular}{lccccccccc}
\toprule
\textbf{Evaluation Subset} & \textbf{$N$} & \textbf{P1 Mean} & \textbf{B5 Mean} & \textbf{Mean Diff} & \textbf{95\% Bootstrap CI} & \textbf{$t$-stat} & \textbf{$p$-value ($t$)} & \textbf{Wilcoxon $p$} & \textbf{Cohen's $d$} \\
\midrule
Pooled Observations        & 90 & 0.1570 & 0.1477 & +0.0093 & [-0.0051, +0.0340] & 0.8203  & 0.4143 & 0.8589 & 0.0865 \\
QASPER (Token $F_1$)       & 30 & 0.1042 & 0.1097 & -0.0055 & [-0.0186, +0.0050] & -0.9199 & 0.3652 & 0.4446 & -0.1680 \\
LongHealth (Accuracy)      & 60 & 0.1833 & 0.1667 & +0.0167 & [0.0000, +0.0500]  & 1.0000  & 0.3214 & 0.3173 & 0.1291 \\
\midrule
A2: SA-CMS 2-Level         & 90 & 0.2101 & N/A    & N/A     & N/A                & N/A     & N/A    & N/A    & N/A \\
A1: Random Boundary CMS    & -- & \multicolumn{8}{c}{\textit{Awaiting External GPU Execution (\texttt{NEED\_EXTERNAL\_GPU})}} \\
A3: SA-CMS Additive        & -- & \multicolumn{8}{c}{\textit{Awaiting External GPU Execution (\texttt{NEED\_EXTERNAL\_GPU})}} \\
\bottomrule
\end{tabular}
\end{table*}
"""
    with open(LATEX_DIR / "rq2_comparison.tex", "w", encoding="utf-8") as f:
        f.write(content)
    print("Saved: rq2_comparison.tex")

def generate_rq3_latex():
    content = r"""\begin{table*}[t]
\centering
\small
\caption{RQ3 Context-Evicted QA Evaluation and Robustness on Vietnamese Benchmark (100 Items, 3 Seeds).}
\label{tab:rq3_context_evicted}
\begin{tabular}{lccccccccc}
\toprule
\textbf{Method} & \textbf{Retriever} & \textbf{Token $F_1$ [95\% CI]} & \textbf{EM} & \textbf{Faithfulness [\%]} & \textbf{Corr. Refusal} & \textbf{False Refusal} & \textbf{False Ans.} & \textbf{Latency (ms)} & \textbf{VRAM (MB)} \\
\midrule
B1 (ICL Context) & None  & 0.0048 [0.0016, 0.0088] & 0.0 & 0.00\% [0.00, 0.00]   & 0.0\%  & 0.0\% & 100.0\% & 12,760.58 & 364.69 \\
B2 (BM25 RAG)    & BM25  & 0.1596 [0.1164, 0.2057] & 0.0 & 95.70\% [92.47, 98.39] & 76.0\% & 0.0\% & 24.0\%  & 7,873.78  & 353.31 \\
B5 (Fixed CMS)   & None  & 0.0502 [0.0346, 0.0679] & 0.0 & 0.00\% [0.00, 0.00]   & 0.0\%  & 0.0\% & 100.0\% & 4,634.92  & 317.73 \\
P1 (SA-CMS)      & None  & 0.0753 [0.0563, 0.0958] & 0.0 & 0.00\% [0.00, 0.00]   & 0.0\%  & 0.0\% & 100.0\% & 4,626.27  & 331.24 \\
P2 (Hybrid)      & BM25  & 0.1596 [0.1164, 0.2057] & 0.0 & 98.92\% [97.31, 100.0] & 76.0\% & 0.0\% & 24.0\%  & 7,768.73  & 369.48 \\
\bottomrule
\end{tabular}
\end{table*}
"""
    with open(LATEX_DIR / "rq3_context_evicted.tex", "w", encoding="utf-8") as f:
        f.write(content)
    print("Saved: rq3_context_evicted.tex")

def generate_rq5_latex():
    content = r"""\begin{table*}[t]
\centering
\small
\caption{RQ5 Computational and Memory Cost Profiles under Standardized Evaluation Conditions.}
\label{tab:rq5_cost}
\begin{tabular}{lcccccc}
\toprule
\textbf{Method} & \textbf{Checkpoint (MB)} & \textbf{Trainable Params} & \textbf{Retrieval (ms)} & \textbf{Generation (ms)} & \textbf{Total Query (ms)} & \textbf{Peak VRAM (MB)} \\
\midrule
B1 (ICL Context) & 0.00  & 0         & 0.00 & 5,168.39 & 5,168.39 & 364.69 \\
B2 (BM25 RAG)    & 0.00  & 0         & 2.13 & 4,117.37 & 4,119.49 & 353.31 \\
B4 (1-Level CMS) & 6.77  & 1,771,584 & 0.00 & 978.59   & 978.59   & 310.67 \\
B5 (3-Level CMS) & 20.28 & 5,314,752 & 0.00 & 1,009.40 & 1,009.40 & 317.73 \\
P1 (SA-CMS)      & 20.28 & 5,314,752 & 0.00 & 1,036.54 & 1,036.54 & 331.24 \\
P2 (Hybrid)      & 20.28 & 5,314,752 & 1.35 & 4,247.96 & 4,249.31 & 369.48 \\
\bottomrule
\end{tabular}
\end{table*}
"""
    with open(LATEX_DIR / "rq5_cost.tex", "w", encoding="utf-8") as f:
        f.write(content)
    print("Saved: rq5_cost.tex")

def generate_benchmark_status_latex():
    content = r"""\begin{table*}[t]
\centering
\small
\caption{Benchmark Execution Matrix across Methods, Research Questions, and Datasets.}
\label{tab:benchmark_status}
\begin{tabular}{llccccccc}
\toprule
\textbf{Method} & \textbf{Category} & \textbf{RQ1} & \textbf{RQ2} & \textbf{RQ3} & \textbf{RQ4} & \textbf{RQ5} & \textbf{Overall Status} & \textbf{External GPU} \\
\midrule
B1 & Baseline (ICL Context) & COMPLETED & NOT\_RUN & COMPLETED & NOT\_RUN & COMPLETED & COMPLETED & False \\
B2 & Baseline (BM25 RAG)    & NOT\_RUN  & NOT\_RUN & COMPLETED & NOT\_RUN & COMPLETED & COMPLETED & False \\
B3 & Baseline (Cartridges)  & EXCLUDED  & EXCLUDED & EXCLUDED  & EXCLUDED & EXCLUDED  & EXCLUDED  & False \\
B4 & Baseline (1-Level)     & COMPLETED & NOT\_RUN & NOT\_RUN  & NEED\_GPU& COMPLETED & PARTIAL   & True \\
B5 & Baseline (Fixed CMS)   & COMPLETED & COMPLETED& COMPLETED & NEED\_GPU& COMPLETED & PARTIAL   & True \\
P1 & Proposed (SA-CMS)      & COMPLETED & COMPLETED& COMPLETED & NEED\_GPU& COMPLETED & PARTIAL   & True \\
P2 & Proposed (Hybrid)      & NOT\_RUN  & NOT\_RUN & COMPLETED & NOT\_RUN & COMPLETED & COMPLETED & False \\
\midrule
A1 & Ablation (Random Bound)& NOT\_RUN  & NEED\_GPU& NOT\_RUN  & NOT\_RUN & NOT\_RUN  & NEED\_EXTERNAL\_GPU & True \\
A2 & Ablation (2-Level)     & NOT\_RUN  & COMPLETED& NOT\_RUN  & NOT\_RUN & NOT\_RUN  & COMPLETED & False \\
A3 & Ablation (Additive)    & NOT\_RUN  & NEED\_GPU& NOT\_RUN  & NOT\_RUN & NOT\_RUN  & NEED\_EXTERNAL\_GPU & True \\
\bottomrule
\end{tabular}
\end{table*}
"""
    with open(LATEX_DIR / "benchmark_status.tex", "w", encoding="utf-8") as f:
        f.write(content)
    print("Saved: benchmark_status.tex")

if __name__ == "__main__":
    print("Generating LaTeX Tables...")
    generate_rq1_latex()
    generate_rq2_latex()
    generate_rq3_latex()
    generate_rq5_latex()
    generate_benchmark_status_latex()
    print("All LaTeX tables generated successfully!")
