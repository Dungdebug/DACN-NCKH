"""
Detailed audit of RQ2 Mean Diff vs RQ1 QASPER F1.
Answers all 10 questions specified in Phase 4.2 Task 1:
1. Mean Diff duoc tinh tu metric nao?
2. Dataset nao?
3. Co phai QASPER item-level khong?
4. Co tinh tren 90 cau hoi hay subset khac?
5. Co gop 3 seeds truoc hay sau paired test?
6. Pairing chinh xac theo question_id chua?
7. Co duplication question_id khong?
8. B5 va P1 co cung document_id khong?
9. Co filter sample nao truoc statistical test khong?
10. Mean Diff co thuc su la mean(P1-B5) tren paired item-level metric khong?
"""

import json
import numpy as np
import scipy.stats

def audit_rq2():
    with open("results/phase4_1/rq2/rq2_raw_results.json", "r", encoding="utf-8") as f:
        rq2_raw = json.load(f)

    with open("results/phase4_1/rq1/rq1_raw_results.json", "r", encoding="utf-8") as f:
        rq1_raw = json.load(f)

    # 1. Inspect RQ1 QASPER runs
    print("=== 1. RQ1 QASPER RUNS ===")
    for r in rq1_raw["runs"]:
        if r["benchmark"] == "QASPER":
            print(f"RQ1 {r['method']} seed {r['seed']}: token_f1={r.get('token_f1')}, target_prob={r.get('avg_target_prob')}")
            for d in r.get("detailed", [])[:2]:
                print(f"    sample: doc={d.get('doc_idx')}, f1={d.get('token_f1')}")

    # 2. Inspect RQ2 structure
    print("\n=== 2. RQ2 RAW STRUCTURE ===")
    runs = rq2_raw["runs"]
    print(f"Total entries in rq2_raw['runs']: {len(runs)}")
    
    b5_runs = [r for r in runs if r["method"] == "B5"]
    p1_runs = [r for r in runs if r["method"] == "P1"]
    a1_runs = [r for r in runs if r["method"] == "A1"]
    a2_runs = [r for r in runs if r["method"] == "A2"]
    print(f"B5 count: {len(b5_runs)}, P1 count: {len(p1_runs)}, A1 count: {len(a1_runs)}, A2 count: {len(a2_runs)}")

    # Check datasets and IDs
    b5_qasper = [r for r in b5_runs if r["dataset"] == "QASPER"]
    p1_qasper = [r for r in p1_runs if r["dataset"] == "QASPER"]
    b5_lh = [r for r in b5_runs if r["dataset"] == "LongHealth"]
    p1_lh = [r for r in p1_runs if r["dataset"] == "LongHealth"]
    print(f"B5 QASPER: {len(b5_qasper)}, P1 QASPER: {len(p1_qasper)}")
    print(f"B5 LongHealth: {len(b5_lh)}, P1 LongHealth: {len(p1_lh)}")

    # Check unique IDs and pairing
    b5_keys = [(r["dataset"], r["item_id"], r["seed"]) for r in b5_runs]
    p1_keys = [(r["dataset"], r["item_id"], r["seed"]) for r in p1_runs]
    print(f"Unique B5 keys: {len(set(b5_keys))} out of {len(b5_keys)}")
    print(f"Unique P1 keys: {len(set(p1_keys))} out of {len(p1_keys)}")
    assert len(set(b5_keys)) == len(b5_keys), "Duplicate B5 keys detected!"
    assert len(set(p1_keys)) == len(p1_keys), "Duplicate P1 keys detected!"
    assert set(b5_keys) == set(p1_keys), "Mismatch between B5 and P1 keys!"

    # Verify pairing order
    diffs_all = []
    p1_vals_all = []
    b5_vals_all = []
    
    diffs_qasper = []
    p1_vals_qasper = []
    b5_vals_qasper = []

    diffs_lh = []
    p1_vals_lh = []
    b5_vals_lh = []

    for k in sorted(list(set(b5_keys))):
        r_b5 = [r for r in b5_runs if (r["dataset"], r["item_id"], r["seed"]) == k][0]
        r_p1 = [r for r in p1_runs if (r["dataset"], r["item_id"], r["seed"]) == k][0]
        
        dset = k[0]
        val_metric = "token_f1" if dset == "QASPER" else "accuracy"
        v_b5 = r_b5[val_metric]
        v_p1 = r_p1[val_metric]
        d = v_p1 - v_b5

        diffs_all.append(d)
        p1_vals_all.append(v_p1)
        b5_vals_all.append(v_b5)

        if dset == "QASPER":
            diffs_qasper.append(d)
            p1_vals_qasper.append(v_p1)
            b5_vals_qasper.append(v_b5)
        else:
            diffs_lh.append(d)
            p1_vals_lh.append(v_p1)
            b5_vals_lh.append(v_b5)

    print("\n=== 3. COMPUTED STATISTICS ===")
    print(f"All 90 items (30 QASPER + 60 LongHealth):")
    print(f"  P1 mean: {np.mean(p1_vals_all):.6f}")
    print(f"  B5 mean: {np.mean(b5_vals_all):.6f}")
    print(f"  Mean diff: {np.mean(diffs_all):.6f}")
    print(f"  Std diff: {np.std(diffs_all, ddof=1):.6f}")
    t_stat, p_val = scipy.stats.ttest_rel(p1_vals_all, b5_vals_all)
    w_stat, w_pval = scipy.stats.wilcoxon(p1_vals_all, b5_vals_all, zero_method="wilcox")
    cohens_d = np.mean(diffs_all) / np.std(diffs_all, ddof=1)
    print(f"  Paired t-test: t={t_stat:.4f}, p={p_val:.6f}")
    print(f"  Wilcoxon: W={w_stat:.4f}, p={w_pval:.6f}")
    print(f"  Cohen's d: {cohens_d:.4f}")

    print(f"\nQASPER only (30 items):")
    print(f"  P1 mean F1: {np.mean(p1_vals_qasper):.6f}")
    print(f"  B5 mean F1: {np.mean(b5_vals_qasper):.6f}")
    print(f"  Mean diff: {np.mean(diffs_qasper):.6f}")
    print(f"  Std diff: {np.std(diffs_qasper, ddof=1):.6f}")
    t_q, p_q = scipy.stats.ttest_rel(p1_vals_qasper, b5_vals_qasper)
    w_q, wp_q = scipy.stats.wilcoxon(p1_vals_qasper, b5_vals_qasper, zero_method="wilcox")
    print(f"  Paired t-test: t={t_q:.4f}, p={p_q:.6f}")
    print(f"  Wilcoxon: W={w_q:.4f}, p={wp_q:.6f}")

    print(f"\nLongHealth only (60 items):")
    print(f"  P1 mean Acc: {np.mean(p1_vals_lh):.6f}")
    print(f"  B5 mean Acc: {np.mean(b5_vals_lh):.6f}")
    print(f"  Mean diff: {np.mean(diffs_lh):.6f}")
    print(f"  Std diff: {np.std(diffs_lh, ddof=1):.6f}")
    t_lh, p_lh = scipy.stats.ttest_rel(p1_vals_lh, b5_vals_lh)
    w_lh, wp_lh = scipy.stats.wilcoxon(p1_vals_lh, b5_vals_lh, zero_method="wilcox")
    print(f"  Paired t-test: t={t_lh:.4f}, p={p_lh:.6f}")
    print(f"  Wilcoxon: W={w_lh:.4f}, p={wp_lh:.6f}")

    # 4. Check why RQ1 QASPER F1 is ~0.28 vs RQ2 QASPER F1 ~0.10
    print("\n=== 4. RQ1 vs RQ2 QASPER PROMPT AND EVALUATION COMPARISON ===")
    # Look at how RQ1 QASPER detailed items were generated
    for r in rq1_raw["runs"]:
        if r["benchmark"] == "QASPER" and r["method"] in ("B5", "P1"):
            print(f"RQ1 {r['method']} seed {r['seed']} detailed length: {len(r.get('detailed', []))}")
            for d in r.get("detailed", [])[:2]:
                print(f"  doc={d.get('doc_idx')}, gen_ans={repr(d.get('gen_answer'))}, gt={repr(d.get('ground_truth'))}, f1={d.get('token_f1')}")

    for r in rq2_raw["runs"]:
        if r["dataset"] == "QASPER" and r["method"] in ("B5", "P1") and r["doc_idx"] < 2 and r["seed"] == 42:
            print(f"RQ2 {r['method']} seed {r['seed']} doc={r['doc_idx']}: gen_ans={repr(r.get('gen_answer'))}, gt={repr(r.get('ground_truth'))}, f1={r.get('token_f1')}")

if __name__ == "__main__":
    audit_rq2()
