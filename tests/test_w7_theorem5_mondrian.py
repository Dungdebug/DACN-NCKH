"""
Tests for Theorem 5: MondrianConformalRiskController (Ticket T5b).

100% Strict Offline, Zero External APIs, Pure Python & NumPy.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from src.wave_cms.causal_attribution import (
    MondrianConformalRiskController,
    RefusalDecision,
)
from src.wave_cms.nonconformity import MONDRIAN_GROUPS, SIGNAL_NAMES


def _make_synthetic_signal_row(
    rng: np.random.RandomState,
    group: str,
    y: int,
    row_id: int,
    pmi_missing: bool = False,
) -> Dict[str, Any]:
    """Generates synthetic 8-signal row correlated with y and group-specific shifts."""
    group_shift = {
        "VI-encyclopedia": -0.05,
        "EN-encyclopedia": 0.05,
        "EN-reading": 0.0,
        "EN-longpaper": -0.08,
    }[group]
    base = (0.75 if y == 1 else 0.25) + group_shift
    signals = {
        "margin": float(np.clip(base + rng.normal(0.0, 0.15), 0.0, 1.0)),
        "entropy": float(np.clip((1.0 - base) + rng.normal(0.0, 0.15), 0.0, 1.0)),
        "lolo_drop": float(rng.normal(1.2 if y == 1 else -0.4, 0.5)),
        "salient_cov": float(np.clip(base + rng.normal(0.0, 0.12), 0.0, 1.0)),
        "negation_mismatch": float(0.8 if (y == 0 and rng.rand() < 0.4) else 0.0),
        "answer_type_ok": float(1.0 if (y == 1 or rng.rand() < 0.5) else 0.2),
        "pmi": 0.0 if pmi_missing else float(rng.normal(1.5 if y == 1 else -0.5, 0.6)),
        "bm25_max": float(np.clip(base + rng.normal(0.0, 0.15), 0.0, 1.0)),
        "pmi_missing": 1.0 if pmi_missing else 0.0,
    }
    return {"id": row_id, "group": group, "y": y, "signals": signals}


class TestMondrianConformalRiskController:
    def test_theorem5_montecarlo_2000_trials_per_group_risk_control(self):
        """
        Theorem 5 Monte-Carlo verification (2,000 trials across 4 Mondrian groups):
        E[ell_test(lambda_hat_g) | group=g] <= epsilon_g + 0.01.
        """
        rng = np.random.RandomState(2026)
        controller = MondrianConformalRiskController()
        eps_by_group = {
            "VI-encyclopedia": 0.08,
            "EN-encyclopedia": 0.10,
            "EN-reading": 0.12,
            "EN-longpaper": 0.15,
        }
        n_trials = 2000
        n_cal_g = 60
        n_test_g = 40

        group_losses: Dict[str, List[float]] = {g: [] for g in MONDRIAN_GROUPS}

        for _ in range(n_trials):
            cal_rows: List[Dict[str, Any]] = []
            test_rows_by_group: Dict[str, List[Dict[str, Any]]] = {}

            for g_idx, g in enumerate(MONDRIAN_GROUPS):
                # Exchangeable pool of n_cal_g + n_test_g samples from group-specific law
                n_total = n_cal_g + n_test_g
                scores = rng.beta(2.0 + 0.3 * g_idx, 1.8, size=n_total)
                probs = np.clip(0.15 + 0.80 * (scores ** 1.4), 0.02, 0.98)
                ys = (rng.rand(n_total) < probs).astype(np.int32)

                perm = rng.permutation(n_total)
                cal_idx = perm[:n_cal_g]
                test_idx = perm[n_cal_g:]

                for idx in cal_idx:
                    cal_rows.append({"group": g, "score": float(scores[idx]), "y": int(ys[idx])})
                test_rows_by_group[g] = [
                    {"score": float(scores[idx]), "y": int(ys[idx])} for idx in test_idx
                ]

            stats = controller.calibrate_groups(cal_rows, eps_by_group=eps_by_group)
            for g in MONDRIAN_GROUPS:
                lam_g = stats[g]["lambda_g"]
                t_rows = test_rows_by_group[g]
                losses = [
                    1.0 if (r["score"] >= lam_g and r["y"] == 0) else 0.0
                    for r in t_rows
                ]
                group_losses[g].append(float(np.mean(losses)))

        for g in MONDRIAN_GROUPS:
            mean_risk = float(np.mean(group_losses[g]))
            assert mean_risk <= eps_by_group[g] + 0.01, (
                f"Group {g} exceeded CRC bound: mean_risk={mean_risk:.4f} > "
                f"eps={eps_by_group[g]:.4f} + 0.01"
            )

    def test_threshold_monotonicity_in_epsilon(self):
        """lambda_hat_g is non-increasing as epsilon_g increases."""
        rng = np.random.RandomState(77)
        controller = MondrianConformalRiskController()
        cal_rows = []
        for g in MONDRIAN_GROUPS:
            for _ in range(80):
                s = float(rng.uniform(0.0, 1.0))
                y = int(rng.rand() < (0.1 + 0.85 * s))
                cal_rows.append({"group": g, "score": s, "y": y})

        eps_grid = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
        prev_lambdas = {g: float("inf") for g in MONDRIAN_GROUPS}
        for eps in eps_grid:
            stats = controller.calibrate_groups(cal_rows, eps_by_group=eps)
            for g in MONDRIAN_GROUPS:
                lam_g = stats[g]["lambda_g"]
                assert lam_g <= prev_lambdas[g] + 1e-12
                prev_lambdas[g] = lam_g

    def test_no_manual_clamping_and_infeasible_group_fallback(self):
        """Thresholds are not clamped to [0.28, 0.42], and tiny groups flag infeasible_group_size."""
        controller = MondrianConformalRiskController()
        cal_rows = []
        # Group A: all correct -> lambda_g should drop to 0.0 (< 0.28)
        for i in range(40):
            cal_rows.append({"group": "VI-encyclopedia", "score": 0.1 + 0.02 * i, "y": 1})
        # Group B: many high-score wrong -> lambda_g should be > 0.42
        for i in range(40):
            s = 0.02 * (i + 1)
            y = 1 if s > 0.85 else 0
            cal_rows.append({"group": "EN-reading", "score": s, "y": y})
        # Group C: infeasible n_g = 3 < 1/0.10 - 1 = 9
        for i in range(3):
            cal_rows.append({"group": "EN-longpaper", "score": 0.9, "y": 1})

        stats = controller.calibrate_groups(cal_rows, eps_by_group=0.10)
        assert stats["VI-encyclopedia"]["lambda_g"] < 0.28
        assert stats["EN-reading"]["lambda_g"] > 0.42
        assert stats["EN-longpaper"]["infeasible_group_size"] is True
        assert stats["EN-longpaper"]["lambda_g"] > 1.0

    def test_end_to_end_no_leakage_and_evaluate_signals(self):
        """
        Verifies Fit -> Cal -> Test disjointness (Fit ∩ Cal ∩ Test = ∅),
        pmi_missing imputation from Fit mean, select_eps_on_fit, and evaluate_signals.
        """
        rng = np.random.RandomState(99)
        all_rows: List[Dict[str, Any]] = []
        row_id = 0
        for g in MONDRIAN_GROUPS:
            for _ in range(90):
                y = int(rng.rand() < 0.65)
                pmi_missing = bool(rng.rand() < 0.2)
                all_rows.append(_make_synthetic_signal_row(rng, g, y, row_id, pmi_missing=pmi_missing))
                row_id += 1

        perm = rng.permutation(len(all_rows))
        fit_rows = [all_rows[i] for i in perm[:160]]
        cal_rows = [all_rows[i] for i in perm[160:260]]
        test_rows = [all_rows[i] for i in perm[260:]]

        fit_ids = {r["id"] for r in fit_rows}
        cal_ids = {r["id"] for r in cal_rows}
        test_ids = {r["id"] for r in test_rows}
        assert fit_ids.isdisjoint(cal_ids)
        assert fit_ids.isdisjoint(test_ids)
        assert cal_ids.isdisjoint(test_ids)

        ctrl = MondrianConformalRiskController()
        fit_info = ctrl.fit_weights(fit_rows)
        assert fit_info["status"] == "SUCCESS"
        assert ctrl.weights_ is not None and ctrl.weights_.shape == (len(SIGNAL_NAMES),)
        assert ctrl.feature_means_ is not None and ctrl.feature_stds_ is not None
        assert 0.5 <= ctrl.fit_auroc_ <= 1.0

        eps_map = ctrl.select_eps_on_fit(fit_rows)
        assert set(eps_map.keys()) == set(MONDRIAN_GROUPS)

        cal_stats = ctrl.calibrate_groups(cal_rows, eps_by_group=eps_map)
        assert ctrl.is_calibrated is True
        assert set(cal_stats.keys()) == set(MONDRIAN_GROUPS)

        for r in test_rows:
            g = r["group"]
            decision = ctrl.evaluate_signals(r["signals"], group=g)
            assert isinstance(decision, RefusalDecision)
            assert decision.group == g
            assert decision.lambda_g == pytest.approx(cal_stats[g]["lambda_g"])
            assert decision.epsilon_g == pytest.approx(eps_map[g])
            assert 0.0 <= decision.score <= 1.0
            assert decision.refused is (decision.score < decision.lambda_g)
            assert len(decision.top_risk_factors) == 3
            d_dict = decision.to_dict()
            assert d_dict["group"] == g
            assert "top_risk_factors" in d_dict

    def test_single_class_fit_fallback_numpy_gd(self):
        """Single-class Fit split falls back cleanly to pure-NumPy gradient descent."""
        rng = np.random.RandomState(11)
        fit_rows = [
            _make_synthetic_signal_row(rng, "EN-encyclopedia", y=1, row_id=i)
            for i in range(25)
        ]
        ctrl = MondrianConformalRiskController()
        res = ctrl.fit_weights(fit_rows)
        assert res["status"] == "SUCCESS"
        assert ctrl.weights_ is not None
        s = ctrl.score_signals(fit_rows[0]["signals"])
        assert 0.0 <= s <= 1.0

    def test_r3_fallback_to_global_for_small_group(self):
        """With fallback_to_global=True, infeasible groups (n_g < 15) use global_lambda_."""
        ctrl = MondrianConformalRiskController()
        cal_rows = []
        for i in range(40):
            cal_rows.append({"group": "EN-reading", "score": 0.1 + 0.02 * i, "y": 1 if i >= 5 else 0})
        for i in range(4):
            cal_rows.append({"group": "EN-longpaper", "score": 0.9, "y": 1})

        stats = ctrl.calibrate_groups(cal_rows, eps_by_group=0.10, fallback_to_global=True)
        assert stats["EN-longpaper"]["infeasible_group_size"] is True
        assert stats["EN-longpaper"]["fallback_to_global"] is True
        assert stats["EN-longpaper"]["lambda_g"] == pytest.approx(ctrl.global_lambda_)

    def test_r3_forward_selection_with_kill_criterion(self):
        """forward_select_r3=True keeps helpful R3 signals and kills noisy ones."""
        rng = np.random.RandomState(123)
        fit_rows: List[Dict[str, Any]] = []
        for i in range(120):
            y = int(i % 2 == 0)
            # Base 8 signals are noisy (weak/random) so base CV AUROC ~ 0.5
            signals = {name: float(rng.normal(0.0, 1.0)) for name in SIGNAL_NAMES}
            signals["pmi_missing"] = 0.0
            signals["cross_retriever_agree"] = float(rng.normal(0.0, 1.0))
            signals["sent_margin"] = float(1.0 if y == 1 else 0.0) + float(rng.normal(0.0, 0.1))
            signals["idf_query_cov"] = float(rng.normal(0.0, 1.0))
            signals["analyzer_conf"] = float(rng.normal(0.0, 1.0))
            fit_rows.append({"id": i, "group": "EN-reading", "y": y, "signals": signals})

        ctrl = MondrianConformalRiskController(seed=123)
        res = ctrl.fit_weights(fit_rows, forward_select_r3=True, min_auroc_gain=0.005)
        assert res["status"] == "SUCCESS"
        assert "cv_auroc_base" in res and "cv_auroc_selected" in res
        assert "sent_margin" in res["kept_r3_signals"]
        assert len(res["killed_r3_signals"]) >= 1
        assert res["cv_auroc_selected"] >= res["cv_auroc_base"] + 0.005
        assert ctrl.weights_.shape == (len(ctrl.selected_signal_names_),)
        assert ctrl.active_signal_names_ == ctrl.selected_signal_names_

    def test_r3_audit_resplit_risk_500_trials(self):
        """500-trial Monte-Carlo resplit audit runs fast and reports per-group bounds."""
        import time

        rng = np.random.RandomState(7)
        pooled = []
        for g in ("EN-reading", "VI-encyclopedia"):
            for _ in range(80):
                s = float(rng.uniform(0.0, 1.0))
                y = int(rng.rand() < (0.15 + 0.8 * s))
                pooled.append({"group": g, "score": s, "y": y})

        ctrl = MondrianConformalRiskController()
        t0 = time.perf_counter()
        report = ctrl.audit_resplit_risk(
            pooled, eps=0.10, cal_fraction=0.25, n_trials=500, seed=42, fallback_to_global=True
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0
        for g in ("EN-reading", "VI-encyclopedia"):
            assert g in report
            assert set(report[g].keys()) >= {"mean_risk", "std_risk", "se_risk", "ci95_upper", "exceeds_bound"}
            assert report[g]["ci95_upper"] == pytest.approx(report[g]["mean_risk"] + 1.96 * report[g]["se_risk"])
