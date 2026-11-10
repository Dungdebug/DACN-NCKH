# SA-CMS Phase 4.5 Final Training & Evaluation Output Package

This package contains the official experimental results, model checkpoints, raw result data, statistics, and validation records for the SA-CMS project (Phase 4.5 + RQ2).

## Directory Structure
- `A1_random_boundary/`: Results and evaluation artifacts for Experiment A1 (Random Boundary CMS).
- `A3_additive_ungated/`: Results and evaluation artifacts for Experiment A3 (Additive / Ungated SA-CMS).
- `RQ4_sequential/`: Results and 36 snapshot checkpoints for Experiment RQ4 (Sequential Continual Ingestion).
- `RQ2_structure_vs_token/`: Results and paired statistical audit for Experiment RQ2 (Structure-Aligned vs Fixed-Token).
- `checkpoints/`: Complete set of validated PyTorch `.pt` checkpoints.
- `raw_results/`: Raw item-level and benchmark JSON output files.
- `statistics/`: Statistical analysis outputs (paired t-test, Wilcoxon, Bootstrap CI, Cohen's d).
- `logs/`: Training and evaluation execution logs.
- `validation/`: Automated validation reports.
- `manifests/`: Checksum and dataset provenance manifests.
- `checksums.sha256`: SHA-256 checksums of all critical package files.
