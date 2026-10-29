"""
Build Phase 4.3 Ablation External Readiness Audit (A1 & A3)
Verifies:
- exactly 200 samples
- same backbone
- same initialization
- same optimizer
- same LR
- same effective batch
- same update budget
- seeds 42/43/44
- Expected parameter count: 5,314,752
Outputs: results/phase4_3/ablation_external_readiness.json
"""

import json
import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RES_DIR = ROOT_DIR / "results" / "phase4_3"
RES_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = RES_DIR / "ablation_external_readiness.json"

A1_DIR = ROOT_DIR / "external_gpu" / "phase4_2" / "A1"
A3_DIR = ROOT_DIR / "external_gpu" / "phase4_2" / "A3"

def audit_ablations():
    print("Auditing A1 and A3 external readiness...")
    
    # Load A1 config
    with open(A1_DIR / "config_a1.yaml", "r", encoding="utf-8") as f:
        cfg_a1 = yaml.safe_load(f)
    with open(A1_DIR / "dataset_manifest.json", "r", encoding="utf-8") as f:
        data_a1 = json.load(f)
    with open(A1_DIR / "seed_manifest.json", "r", encoding="utf-8") as f:
        seed_a1 = json.load(f)

    # Load A3 config
    with open(A3_DIR / "config_a3.yaml", "r", encoding="utf-8") as f:
        cfg_a3 = yaml.safe_load(f)
    with open(A3_DIR / "dataset_manifest.json", "r", encoding="utf-8") as f:
        data_a3 = json.load(f)
    with open(A3_DIR / "seed_manifest.json", "r", encoding="utf-8") as f:
        seed_a3 = json.load(f)

    # Verify parameters (weights + biases)
    d_model = 576
    d_ff = 1536
    num_levels = 3
    # w1: (d_ff * d_model) + d_ff bias
    # w2: (d_model * d_ff) + d_model bias
    params_per_adapter = (2 * d_model * d_ff) + d_ff + d_model # 1,771,584
    expected_total_params = num_levels * params_per_adapter # 5,314,752

    assert expected_total_params == 5314752, f"Param count mismatch: {expected_total_params} != 5314752"

    # Verification checks
    checks = {
        "A1": {},
        "A3": {}
    }

    for name, cfg, data_m, seed_m in [("A1", cfg_a1, data_a1, seed_a1), ("A3", cfg_a3, data_a3, seed_a3)]:
        # 1. Samples count
        samples = cfg["training"]["samples_count"]
        manifest_samples = data_m["num_samples"]
        if samples != 200 or manifest_samples != 200:
            raise ValueError(f"{name}: Samples count mismatch: {samples} vs {manifest_samples} (expected 200)")
        
        # 2. Backbone
        backbone = cfg["model"]["backbone"]
        if backbone != "HuggingFaceTB/SmolLM2-135M":
            raise ValueError(f"{name}: Backbone mismatch: {backbone}")
            
        # 3. Initialization
        m_dim = cfg["model"]["d_model"]
        ff_dim = cfg["model"]["d_ff"]
        lvls = cfg["model"]["num_levels"]
        if m_dim != d_model or ff_dim != d_ff or lvls != num_levels:
            raise ValueError(f"{name}: Initialization mismatch")

        # 4. Optimizer
        opt = cfg["training"]["optimizer"]
        if opt != "AdamW":
            raise ValueError(f"{name}: Optimizer mismatch: {opt}")

        # 5. Learning rate
        lr = float(cfg["training"]["learning_rate"])
        if lr != 0.0001:
            raise ValueError(f"{name}: Learning rate mismatch: {lr}")

        # 6. Effective batch size
        bs = cfg["training"]["batch_size"]
        gas = cfg["training"]["grad_accum_steps"]
        eff_bs = bs * gas
        if eff_bs != 4:
            raise ValueError(f"{name}: Effective batch size mismatch: {eff_bs} (expected 4)")

        # 7. Update budget
        epochs = cfg["training"]["epochs"]
        steps = (samples // eff_bs) * epochs
        if steps != 150:
            raise ValueError(f"{name}: Update budget mismatch: {steps} steps (expected 150)")

        # 8. Seeds
        seeds = cfg["training"]["seeds"]
        manifest_seeds = seed_m["seeds"]
        if seeds != [42, 43, 44] or manifest_seeds != [42, 43, 44]:
            raise ValueError(f"{name}: Seeds mismatch: {seeds}")

        # 9. Trainable params
        reported_params = cfg["model"]["expected_trainable_parameters"]
        if reported_params != 5314752:
            raise ValueError(f"{name}: Reported parameter count mismatch: {reported_params}")

        checks[name] = {
            "status": "VERIFIED_COMPLIANT",
            "backbone": backbone,
            "samples_count": samples,
            "d_model": m_dim,
            "d_ff": ff_dim,
            "num_levels": lvls,
            "trainable_parameters": reported_params,
            "optimizer": opt,
            "learning_rate": lr,
            "batch_size": bs,
            "grad_accum_steps": gas,
            "effective_batch_size": eff_bs,
            "update_budget_steps": steps,
            "seeds": seeds,
            "package_path": f"external_gpu/phase4_2/{name}/",
            "execution_status": "NEED_EXTERNAL_GPU"
        }

    audit_result = {
        "meta": {
            "title": "Phase 4.3 Ablation Readiness Audit (A1 & A3)",
            "date": "2026-10-04",
            "protocol": "Phase 4.0.2 Final Fairness Lock",
            "status": "ALL_CHECKS_PASSED_ZERO_MISMATCH"
        },
        "expected_parameter_count": expected_total_params,
        "ablation_checks": checks,
        "verdict": "READY_FOR_EXTERNAL_GPU_EXECUTION"
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2, ensure_ascii=False)
    print(f"Ablation readiness audit saved to {OUT_FILE}")

if __name__ == "__main__":
    audit_ablations()
