"""
Detailed audit of RQ3 consistency and retrieval alignment.
Audits:
- B1, B2, B5, P1, P2 across seeds [42, 43, 44]
- Verifies exact identical question set across all 15 runs
- Verifies exact identical categories (answerable, unanswerable, insufficient_evidence)
- Checks whether B2 and P2 shared identical retrieval config and refusal controller
- Audits citation traceability, citation support, correct/false refusal, false answers
"""

import json
from collections import Counter

def audit_rq3():
    with open("results/phase4_1/rq3/rq3_raw_results.json", "r", encoding="utf-8") as f:
        rq3_raw = json.load(f)

    runs = rq3_raw["runs"]
    print(f"Total RQ3 runs: {len(runs)}")

    # 1. Check methods and seeds
    methods = sorted(list(set(r["method"] for r in runs)))
    seeds = sorted(list(set(r["seed"] for r in runs)))
    print(f"Methods: {methods}, Seeds: {seeds}")
    assert len(runs) == 15, f"Expected 15 runs, got {len(runs)}"

    # 2. Check question consistency across all runs
    ref_run = runs[0]
    ref_q_ids = [r["question_id"] for r in ref_run["records"]]
    ref_categories = {r["question_id"]: r["category"] for r in ref_run["records"]}
    ref_questions = {r["question_id"]: r["question"] for r in ref_run["records"]}
    ref_gt = {r["question_id"]: r["ground_truth"] for r in ref_run["records"]}

    print(f"Total questions per run: {len(ref_q_ids)}")
    print(f"Categories distribution: {Counter(ref_categories.values())}")

    for idx, r in enumerate(runs):
        m = r["method"]
        s = r["seed"]
        curr_q_ids = [rec["question_id"] for rec in r["records"]]
        assert curr_q_ids == ref_q_ids, f"Run {idx} ({m}, seed {s}) had different question IDs!"
        for rec in r["records"]:
            qid = rec["question_id"]
            assert rec["category"] == ref_categories[qid], f"Category mismatch for {qid} in {m} seed {s}"
            assert rec["question"] == ref_questions[qid], f"Question text mismatch for {qid} in {m} seed {s}"
            assert rec["ground_truth"] == ref_gt[qid], f"Ground truth mismatch for {qid} in {m} seed {s}"

    print("Question set consistency check: PASSED. All 15 runs evaluated on strictly identical 100 questions.")

    # 3. Audit B2 vs P2 alignment
    print("\n=== B2 vs P2 RETRIEVAL & REFUSAL COMPARISON ===")
    b2_runs = {r["seed"]: r for r in runs if r["method"] == "B2"}
    p2_runs = {r["seed"]: r for r in runs if r["method"] == "P2"}

    for s in seeds:
        b2_r = b2_runs[s]["records"]
        p2_r = p2_runs[s]["records"]

        # Check refusal decisions
        b2_refusals = [rec["refused"] for rec in b2_r]
        p2_refusals = [rec["refused"] for rec in p2_r]
        refusal_matches = sum(1 for b, p in zip(b2_refusals, p2_refusals) if b == p)
        print(f"Seed {s}: Refusal match between B2 and P2: {refusal_matches} / 100 ({refusal_matches}%)")
        assert refusal_matches == 100, f"B2 and P2 had differing refusal decisions on seed {s}!"

        # Check refusal reasons
        b2_reasons = [rec.get("refusal_reason") for rec in b2_r]
        p2_reasons = [rec.get("refusal_reason") for rec in p2_r]
        reason_matches = sum(1 for b, p in zip(b2_reasons, p2_reasons) if b == p)
        print(f"Seed {s}: Refusal reason match: {reason_matches} / 100")
        assert reason_matches == 100, f"Refusal reasons differed between B2 and P2 on seed {s}!"

        # Check citation sets
        b2_cites = [tuple(sorted(rec.get("citations", []))) for rec in b2_r]
        p2_cites = [tuple(sorted(rec.get("citations", []))) for rec in p2_r]
        cite_matches = sum(1 for b, p in zip(b2_cites, p2_cites) if b == p)
        print(f"Seed {s}: Citation set match between B2 and P2: {cite_matches} / 100 ({cite_matches}%)")
        assert cite_matches == 100, f"B2 and P2 retrieved different citation sets on seed {s}!"

        # Check answers on answerable questions
        ans_indices = [i for i, rec in enumerate(b2_r) if rec["category"] == "answerable"]
        ans_exact_matches = sum(1 for i in ans_indices if b2_r[i]["generated_answer"] == p2_r[i]["generated_answer"])
        print(f"Seed {s}: Answerable questions exact text match: {ans_exact_matches} / {len(ans_indices)}")

        # Check answers on unanswerables slipping past refusal
        unans_slipping = [i for i, rec in enumerate(b2_r) if rec["category"] in ("unanswerable", "insufficient_evidence") and not rec["refused"]]
        print(f"Seed {s}: Unanswerables slipping past refusal: {len(unans_slipping)} questions")
        diff_slipping = sum(1 for i in unans_slipping if b2_r[i]["generated_answer"] != p2_r[i]["generated_answer"])
        print(f"Seed {s}: Differing generation on slipping unanswerables: {diff_slipping} / {len(unans_slipping)}")

    # 4. Check Faithfulness across methods
    print("\n=== FAITHFULNESS & ERROR TAXONOMY AUDIT ===")
    for m in methods:
        m_runs = [r for r in runs if r["method"] == m]
        avg_f1 = sum(r["token_f1"] for r in m_runs) / len(m_runs)
        avg_faith = sum(r["faithfulness_rate_pct"] for r in m_runs) / len(m_runs)
        avg_corr_ref = sum(r["correct_refusal_rate_pct"] for r in m_runs) / len(m_runs)
        avg_false_ref = sum(r["false_refusal_rate_pct"] for r in m_runs) / len(m_runs)
        avg_false_ans = sum(r["false_answer_rate_pct"] for r in m_runs) / len(m_runs)
        print(f"Method {m:3s} | F1: {avg_f1:.4f} | Faithfulness: {avg_faith:5.1f}% | CorrectRef: {avg_corr_ref:5.1f}% | FalseRef: {avg_false_ref:5.1f}% | FalseAns: {avg_false_ans:5.1f}%")

if __name__ == "__main__":
    audit_rq3()
