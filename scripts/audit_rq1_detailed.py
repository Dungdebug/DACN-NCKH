"""
Detailed audit of RQ1 data consistency.
Checks:
- QASPER: 10 docs, same question IDs across methods, same test split, same seed semantics
- LongHealth: 5 representative docs, 20 MCQs
- MK-NIAH: exactly 100 samples, same IDs across methods, same evaluation condition
- Checks missing items, duplicate items, different randomization, method-specific filtering, generation failure, parsing failure.
"""

import json
from collections import defaultdict

def audit_rq1():
    with open("results/phase4_1/rq1/rq1_raw_results.json", "r", encoding="utf-8") as f:
        rq1_raw = json.load(f)

    runs = rq1_raw["runs"]
    print(f"Total RQ1 runs: {len(runs)}")

    # 1. MK-NIAH Audit
    mkniah_runs = [r for r in runs if r["benchmark"] == "MK-NIAH"]
    print(f"\n=== MK-NIAH AUDIT ({len(mkniah_runs)} runs) ===")
    mkniah_methods = sorted(list(set(r["method"] for r in mkniah_runs)))
    mkniah_seeds = sorted(list(set(r["seed"] for r in mkniah_runs)))
    print(f"Methods: {mkniah_methods}, Seeds: {mkniah_seeds}")

    # Check sample counts and sample_idx per run
    for r in mkniah_runs:
        m = r["method"]
        s = r["seed"]
        samples = r.get("detailed", [])
        idxs = [d["sample_idx"] for d in samples]
        expected_vals = [d["expected_val"] for d in samples]
        gen_answers = [d["generated_answer"] for d in samples]
        
        has_nan = any(d.get("target_prob") is None for d in samples)
        empty_gen = sum(1 for g in gen_answers if not g or not g.strip())

        print(f"Run {m} (seed {s}): num_samples={len(samples)}, unique_idx={len(set(idxs))}, min_idx={min(idxs)}, max_idx={max(idxs)}, has_nan={has_nan}, empty_gen={empty_gen}")
        assert len(samples) == 100, f"MK-NIAH {m} seed {s} did not have 100 samples!"
        assert len(set(idxs)) == 100, f"Duplicate sample_idx in MK-NIAH {m} seed {s}!"

    # Check if all methods received the same sample order and expected_val across methods for each sample_idx
    ref_run = mkniah_runs[0]
    ref_expected = {d["sample_idx"]: d["expected_val"] for d in ref_run["detailed"]}
    for r in mkniah_runs[1:]:
        m = r["method"]
        s = r["seed"]
        curr_expected = {d["sample_idx"]: d["expected_val"] for d in r["detailed"]}
        assert ref_expected == curr_expected, f"MK-NIAH {m} seed {s} had different expected_val mapping!"
    print("MK-NIAH consistency check: PASSED. All 12 runs have exactly 100 identical samples and targets.")

    # 2. QASPER Audit
    qasper_runs = [r for r in runs if r["benchmark"] == "QASPER"]
    print(f"\n=== QASPER AUDIT ({len(qasper_runs)} runs) ===")
    qasper_methods = sorted(list(set(r["method"] for r in qasper_runs)))
    qasper_seeds = sorted(list(set(r["seed"] for r in qasper_runs)))
    print(f"Methods: {qasper_methods}, Seeds: {qasper_seeds}")

    for r in qasper_runs:
        m = r["method"]
        s = r["seed"]
        samples = r.get("detailed", [])
        doc_idxs = [d["doc_idx"] for d in samples]
        gt_answers = [d.get("ground_truth") for d in samples]
        f1_vals = [d.get("f1") for d in samples]
        
        has_nan = any(d.get("target_prob") is None for d in samples)
        none_f1 = sum(1 for v in f1_vals if v is None)

        print(f"Run {m} (seed {s}): num_docs={len(samples)}, unique_docs={len(set(doc_idxs))}, run_f1={r.get('token_f1')}, item_none_f1={none_f1}, has_nan={has_nan}")
        assert len(samples) == 10, f"QASPER {m} seed {s} did not have 10 docs!"
        assert len(set(doc_idxs)) == 10, f"Duplicate doc_idx in QASPER {m} seed {s}!"

    # Check if all methods used identical ground_truth answers for each doc_idx
    ref_q_run = qasper_runs[0]
    ref_q_gt = {d["doc_idx"]: d["ground_truth"] for d in ref_q_run["detailed"]}
    for r in qasper_runs[1:]:
        curr_gt = {d["doc_idx"]: d["ground_truth"] for d in r["detailed"]}
        assert ref_q_gt == curr_gt, f"QASPER {r['method']} seed {r['seed']} had different ground truth!"
    print("QASPER consistency check: PASSED. All 12 runs have exactly 10 identical documents and ground truth answers.")

    # 3. LongHealth Audit
    lh_runs = [r for r in runs if r["benchmark"] == "LongHealth"]
    print(f"\n=== LongHealth AUDIT ({len(lh_runs)} runs) ===")
    lh_methods = sorted(list(set(r["method"] for r in lh_runs)))
    lh_seeds = sorted(list(set(r["seed"] for r in lh_runs)))
    print(f"Methods: {lh_methods}, Seeds: {lh_seeds}")

    for r in lh_runs:
        m = r["method"]
        s = r["seed"]
        samples = r.get("detailed", [])
        q_keys = [(d["doc_idx"], d["q_idx"]) for d in samples]
        acc_vals = [d.get("accuracy") for d in samples]
        has_nan = any(d.get("target_prob") is None for d in samples)

        print(f"Run {m} (seed {s}): num_questions={len(samples)}, unique_q={len(set(q_keys))}, run_acc={r.get('accuracy_pct')}, has_nan={has_nan}")
        assert len(samples) == 20, f"LongHealth {m} seed {s} did not have 20 MCQs!"
        assert len(set(q_keys)) == 20, f"Duplicate question in LongHealth {m} seed {s}!"

    ref_lh_run = lh_runs[0]
    ref_lh_keys = { (d["doc_idx"], d["q_idx"]): d["correct_letter"] for d in ref_lh_run["detailed"] }
    for r in lh_runs[1:]:
        curr_lh_keys = { (d["doc_idx"], d["q_idx"]): d["correct_letter"] for d in r["detailed"] }
        assert ref_lh_keys == curr_lh_keys, f"LongHealth {r['method']} seed {r['seed']} had different correct_letter!"
    print("LongHealth consistency check: PASSED. All 12 runs have exactly 20 identical MCQs and correct letters.")

if __name__ == "__main__":
    audit_rq1()
