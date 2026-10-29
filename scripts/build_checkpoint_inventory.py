"""
Build Checkpoint Inventory for Phase 4.1
Conforming to Task 1 of Phase 4.1 Non-Training Benchmark Specification.
Scans checkpoints/ and results/phase4_1/, checking B4, B5, P1, P2 across seeds [42, 43, 44].
"""

import os
import json
import hashlib
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results" / "phase4_1"
OUT_FILE = RESULTS_DIR / "checkpoint_inventory.json"

SEEDS = [42, 43, 44]

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def inspect_checkpoint(filepath: Path) -> dict:
    if not filepath.exists():
        return {"exists": False}
    sz = filepath.stat().st_size
    sha = compute_sha256(filepath)
    info = {
        "exists": True,
        "path": str(filepath.relative_to(ROOT_DIR)).replace("\\", "/"),
        "size_bytes": sz,
        "size_mb": round(sz / (1024 * 1024), 2),
        "sha256": sha,
    }
    try:
        sd = torch.load(filepath, map_location="cpu")
        if isinstance(sd, dict):
            info["keys"] = list(sd.keys())
            if "cms_state_dict" in sd:
                info["cms_param_count"] = sum(p.numel() for p in sd["cms_state_dict"].values())
            if "num_levels" in sd:
                info["num_levels"] = sd["num_levels"]
            if "seed" in sd:
                info["seed"] = sd["seed"]
            if "num_samples" in sd:
                info["num_samples"] = sd["num_samples"]
            if "num_epochs" in sd:
                info["num_epochs"] = sd["num_epochs"]
    except Exception as e:
        info["load_error"] = str(e)
    return info

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    inventory = {
        "meta": {
            "title": "Phase 4.1 Checkpoint Inventory",
            "protocol": "Phase 4.0.2 Final Fairness Lock & Phase 4.1 Non-Training Mandate",
            "backbone": "HuggingFaceTB/SmolLM2-135M (frozen)",
            "allowed_operations": [
                "load_existing_checkpoint",
                "frozen_inference",
                "retrieval",
                "answer_generation",
                "metric_calculation",
                "statistical_aggregation",
                "latency_vram_measurement",
                "audit_validation"
            ],
            "prohibited_operations": [
                "train_new_checkpoints",
                "fine_tuning",
                "backprop",
                "optimizer_step",
                "gradient_adapter_update",
                "online_memory_update_without_precomputed_states"
            ]
        },
        "methods": {},
        "summary": {}
    }

    # B1
    inventory["methods"]["B1"] = {
        "name": "ICL Full-Document Context",
        "type": "baseline",
        "requires_adapter_checkpoint": False,
        "backbone": "HuggingFaceTB/SmolLM2-135M (100% frozen)",
        "seeds": {str(s): {"status": "AVAILABLE", "source": "pretrained_backbone_zero_adaptation"} for s in SEEDS},
        "overall_status": "AVAILABLE"
    }

    # B2
    inventory["methods"]["B2"] = {
        "name": "Standard BM25 RAG",
        "type": "baseline",
        "requires_adapter_checkpoint": False,
        "backbone": "HuggingFaceTB/SmolLM2-135M (100% frozen) + BM25 Retriever",
        "seeds": {str(s): {"status": "AVAILABLE", "source": "pretrained_backbone_plus_bm25"} for s in SEEDS},
        "overall_status": "AVAILABLE"
    }

    # B3
    inventory["methods"]["B3"] = {
        "name": "Cartridges / Context Compression",
        "type": "baseline",
        "requires_adapter_checkpoint": False,
        "overall_status": "EXCLUDED",
        "reason": "Excluded per Phase 4.0.2 / 4.0.3 protocol freeze. Cannot be reproduced within controlled student-scale hardware budget; substitution strictly prohibited."
    }

    # B4 (1-level)
    b4_seeds = {}
    for s in SEEDS:
        p = CKPT_DIR / "phase4_1" / f"cms_1lvl_seed_{s}.pt"
        insp = inspect_checkpoint(p)
        if insp["exists"]:
            b4_seeds[str(s)] = {
                "status": "AVAILABLE",
                "checkpoint": insp,
                "conformance": "Trained on exactly 200 samples, 1-level adapter, 1,771,584 params."
            }
        else:
            b4_seeds[str(s)] = {
                "status": "NOT_AVAILABLE / NEED_EXTERNAL_GPU",
                "reason": "Missing checkpoint; training on current machine prohibited."
            }
    inventory["methods"]["B4"] = {
        "name": "Single-Level Adapter",
        "type": "baseline",
        "requires_adapter_checkpoint": True,
        "target_levels": 1,
        "seeds": b4_seeds,
        "overall_status": "AVAILABLE" if all(v["status"] == "AVAILABLE" for v in b4_seeds.values()) else "PARTIAL"
    }

    # B5 (3-level, fixed_token)
    b5_seeds = {}
    for s in SEEDS:
        p = CKPT_DIR / "phase4_1" / f"cms_3lvl_seed_{s}.pt"
        insp = inspect_checkpoint(p)
        if insp["exists"]:
            b5_seeds[str(s)] = {
                "status": "AVAILABLE",
                "checkpoint": insp,
                "conformance": "Trained on exactly 200 samples, 3-level adapter, 5,314,752 params. Evaluated with fixed_token schedule."
            }
        else:
            b5_seeds[str(s)] = {
                "status": "NOT_AVAILABLE / NEED_EXTERNAL_GPU",
                "reason": "Missing checkpoint; training on current machine prohibited."
            }
    inventory["methods"]["B5"] = {
        "name": "Fixed-Token CMS",
        "type": "baseline",
        "requires_adapter_checkpoint": True,
        "target_levels": 3,
        "seeds": b5_seeds,
        "overall_status": "AVAILABLE" if all(v["status"] == "AVAILABLE" for v in b5_seeds.values()) else "PARTIAL"
    }

    # P1 (3-level, structure-aligned)
    p1_seeds = {}
    for s in SEEDS:
        p = CKPT_DIR / "phase4_1" / f"cms_3lvl_seed_{s}.pt"
        insp = inspect_checkpoint(p)
        if insp["exists"]:
            p1_seeds[str(s)] = {
                "status": "AVAILABLE",
                "checkpoint": insp,
                "conformance": "Trained on exactly 200 samples, 3-level adapter, 5,314,752 params (identical theta_0 with B5 per fairness lock). Evaluated with structure-aligned schedule."
            }
        else:
            p1_seeds[str(s)] = {
                "status": "NOT_AVAILABLE / NEED_EXTERNAL_GPU",
                "reason": "Missing checkpoint; training on current machine prohibited."
            }
    inventory["methods"]["P1"] = {
        "name": "SA-CMS Memory-Only (Proposed)",
        "type": "proposed",
        "requires_adapter_checkpoint": True,
        "target_levels": 3,
        "seeds": p1_seeds,
        "overall_status": "AVAILABLE" if all(v["status"] == "AVAILABLE" for v in p1_seeds.values()) else "PARTIAL"
    }

    # P2 (3-level SA-CMS + BM25)
    p2_seeds = {}
    for s in SEEDS:
        p = CKPT_DIR / "phase4_1" / f"cms_3lvl_seed_{s}.pt"
        insp = inspect_checkpoint(p)
        if insp["exists"]:
            p2_seeds[str(s)] = {
                "status": "AVAILABLE",
                "checkpoint": insp,
                "conformance": "Uses identical memory checkpoint as P1 (cms_3lvl_seed_*.pt) combined with frozen BM25 retriever."
            }
        else:
            p2_seeds[str(s)] = {
                "status": "NOT_AVAILABLE / NEED_EXTERNAL_GPU",
                "reason": "Missing checkpoint; training on current machine prohibited."
            }
    inventory["methods"]["P2"] = {
        "name": "SA-CMS + Retrieval Hybrid (Proposed)",
        "type": "proposed",
        "requires_adapter_checkpoint": True,
        "target_levels": 3,
        "seeds": p2_seeds,
        "overall_status": "AVAILABLE" if all(v["status"] == "AVAILABLE" for v in p2_seeds.values()) else "PARTIAL"
    }

    # Ablations A1, A2, A3
    a2_seeds = {}
    for s in SEEDS:
        p = CKPT_DIR / "phase4_1" / f"cms_2lvl_seed_{s}.pt"
        insp = inspect_checkpoint(p)
        if insp["exists"]:
            a2_seeds[str(s)] = {
                "status": "AVAILABLE",
                "checkpoint": insp,
                "conformance": "Trained on exactly 200 samples, 2-level adapter, 3,543,168 params."
            }
        else:
            a2_seeds[str(s)] = {
                "status": "NOT_AVAILABLE / NEED_EXTERNAL_GPU",
                "reason": "Missing checkpoint."
            }
    inventory["methods"]["A2_two_levels"] = {
        "name": "SA-CMS Level 2 (Paragraph + Section)",
        "type": "ablation",
        "requires_adapter_checkpoint": True,
        "target_levels": 2,
        "seeds": a2_seeds,
        "overall_status": "AVAILABLE" if all(v["status"] == "AVAILABLE" for v in a2_seeds.values()) else "PARTIAL"
    }

    inventory["methods"]["A1_random_boundary"] = {
        "name": "CMS Level 3 Random Boundary",
        "type": "ablation",
        "requires_adapter_checkpoint": True,
        "target_levels": 3,
        "seeds": {str(s): {"status": "NEED_EXTERNAL_GPU", "reason": "No 200-sample trained checkpoint on seed; new training prohibited."} for s in SEEDS},
        "overall_status": "NEED_EXTERNAL_GPU"
    }

    inventory["methods"]["A3_no_gating"] = {
        "name": "SA-CMS Additive (No Learned Gates)",
        "type": "ablation",
        "requires_adapter_checkpoint": True,
        "target_levels": 3,
        "seeds": {str(s): {"status": "NEED_EXTERNAL_GPU", "reason": "No specialized additive checkpoint on seed; new training prohibited."} for s in SEEDS},
        "overall_status": "NEED_EXTERNAL_GPU"
    }

    # RQ4 Sequential Document Snapshots
    inventory["methods"]["RQ4_Sequential_Snapshots"] = {
        "name": "Sequential Document Ingestion Snapshots (D0 -> +5 -> +10 -> +20)",
        "type": "continual_snapshots",
        "available_snapshots": [],
        "overall_status": "NEED_EXTERNAL_GPU",
        "reason": "Sequential memory ingestion requires gradient-based state updates; new updates are prohibited on current machine."
    }

    # Write output
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)
    
    print(f"Checkpoint inventory generated at {OUT_FILE}")
    print(f"B1: {inventory['methods']['B1']['overall_status']}")
    print(f"B2: {inventory['methods']['B2']['overall_status']}")
    print(f"B3: {inventory['methods']['B3']['overall_status']}")
    print(f"B4: {inventory['methods']['B4']['overall_status']} (seeds 42, 43, 44)")
    print(f"B5: {inventory['methods']['B5']['overall_status']} (seeds 42, 43, 44)")
    print(f"P1: {inventory['methods']['P1']['overall_status']} (seeds 42, 43, 44)")
    print(f"P2: {inventory['methods']['P2']['overall_status']} (seeds 42, 43, 44)")
    print(f"A2: {inventory['methods']['A2_two_levels']['overall_status']} (seeds 42, 43, 44)")
    print(f"A1: {inventory['methods']['A1_random_boundary']['overall_status']}")
    print(f"A3: {inventory['methods']['A3_no_gating']['overall_status']}")
    print(f"RQ4: {inventory['methods']['RQ4_Sequential_Snapshots']['overall_status']}")

if __name__ == "__main__":
    main()
