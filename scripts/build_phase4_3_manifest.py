"""
Build Master Experiment Manifest for Phase 4.3
Defines all experimental runs, methods, datasets, seeds, checkpoints, update budgets,
retrieval configs, statuses, and source artifact references.
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "results" / "phase4_3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "master_experiment_manifest.json"

def build_manifest():
    manifest = {
        "meta": {
            "title": "Phase 4.3 Master Experiment Manifest",
            "protocol_version": "4.0.2-final-fairness-lock & 4.0.3-scope-lock",
            "freeze_date": "2026-10-04",
            "backbone": "HuggingFaceTB/SmolLM2-135M",
            "total_backbone_parameters": 134515008,
            "backbone_frozen": True,
            "seeds": [42, 43, 44],
            "hardware_target": "NVIDIA GeForce GTX 1650 Ti (4GB) / Handoff: RTX 3050 (6GB/8GB)",
            "context_window_limit": 512
        },
        "methods_catalog": {
            "B1": {"name": "ICL Full-Document Context", "category": "baseline", "levels": 0, "trainable_params": 0, "status": "AVAILABLE"},
            "B2": {"name": "Standard BM25 RAG", "category": "baseline", "levels": 0, "trainable_params": 0, "status": "AVAILABLE"},
            "B3": {"name": "Cartridges / Context Compression", "category": "baseline", "levels": 0, "trainable_params": 0, "status": "EXCLUDED"},
            "B4": {"name": "Single-Level Adapter", "category": "baseline", "levels": 1, "trainable_params": 1771584, "status": "AVAILABLE"},
            "B5": {"name": "Fixed-Token CMS", "category": "baseline", "levels": 3, "trainable_params": 5314752, "status": "AVAILABLE"},
            "P1": {"name": "SA-CMS Memory-Only (Proposed)", "category": "proposed", "levels": 3, "trainable_params": 5314752, "status": "AVAILABLE"},
            "P2": {"name": "SA-CMS + Retrieval Hybrid (Proposed)", "category": "proposed", "levels": 3, "trainable_params": 5314752, "status": "AVAILABLE"},
            "A1": {"name": "CMS Level-3 Random Boundary", "category": "ablation", "levels": 3, "trainable_params": 5314752, "status": "NEED_EXTERNAL_GPU"},
            "A2": {"name": "SA-CMS Level 2 (Paragraph + Section)", "category": "ablation", "levels": 2, "trainable_params": 3543168, "status": "AVAILABLE"},
            "A3": {"name": "SA-CMS Additive (Ungated)", "category": "ablation", "levels": 3, "trainable_params": 5314752, "status": "NEED_EXTERNAL_GPU"}
        },
        "datasets_catalog": {
            "QASPER": {"documents_count": 10, "items_count": 10, "type": "document_qa", "split": "test"},
            "LongHealth": {"documents_count": 5, "items_count": 20, "type": "multiple_choice_qa", "split": "test"},
            "MK-NIAH": {"documents_count": 1, "items_count": 100, "type": "multi_key_needle_retrieval", "split": "test"},
            "Vietnamese": {"documents_count": 20, "items_count": 350, "type": "layered_document_qa", "split": "test"},
            "Incremental_QASPER_RQ4": {"documents_count": 21, "items_count": 10, "type": "sequential_forgetting_stream", "split": "test"}
        },
        "experiments": []
    }

    # Retrieval configs
    retrieval_frozen_config = {
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "top_k": 5,
        "score_threshold": 3.0,
        "chunk_size": 256,
        "chunk_overlap": 32,
        "evidence_selector_threshold": 3.0,
        "refusal_controller": {
            "min_evidence_score": 3.0,
            "min_evidence_count": 1,
            "min_query_coverage": 0.35
        }
    }

    # 1. RQ1 Experiments (B1, B4, B5, P1 on MK-NIAH, QASPER, LongHealth)
    for m in ["B1", "B4", "B5", "P1"]:
        for dset, d_info in [("MK-NIAH", manifest["datasets_catalog"]["MK-NIAH"]),
                             ("QASPER", manifest["datasets_catalog"]["QASPER"]),
                             ("LongHealth", manifest["datasets_catalog"]["LongHealth"])]:
            ckpt = "None" if m == "B1" else f"checkpoints/phase4_1/cms_{1 if m=='B4' else 3}lvl_seed_{{seed}}.pt"
            manifest["experiments"].append({
                "experiment_id": f"RQ1_{m}_{dset}",
                "rq": "RQ1",
                "method": m,
                "dataset": dset,
                "split": d_info["split"],
                "number_of_docs": d_info["documents_count"],
                "number_of_questions_or_items": d_info["items_count"],
                "seeds": [42, 43, 44],
                "checkpoint": ckpt,
                "training_samples": 0 if m == "B1" else 200,
                "update_budget": "0" if m == "B1" else "Matched online gradient steps",
                "retrieval_config": None,
                "context_size": 512,
                "quantization": "float16 (cuda) / float32 (cpu)",
                "status": "COMPLETED",
                "result_file": "results/phase4_1/rq1/rq1_raw_results.json",
                "audit_file": "results/phase4_2/rq1_data_audit.json",
                "external_gpu_requirement": False
            })

    # 2. RQ2 Experiments (B5, P1, A2, A1, A3 on QASPER and LongHealth)
    for m in ["B5", "P1", "A2", "A1", "A3"]:
        for dset, d_info in [("QASPER", manifest["datasets_catalog"]["QASPER"]),
                             ("LongHealth", manifest["datasets_catalog"]["LongHealth"])]:
            is_ext = m in ("A1", "A3")
            if m == "A1":
                ckpt = "external_gpu/phase4_2/A1/cms_3lvl_random_seed_{seed}.pt"
                audit_file = "results/phase4_3/ablation_external_readiness.json"
            elif m == "A3":
                ckpt = "external_gpu/phase4_2/A3/cms_3lvl_additive_seed_{seed}.pt"
                audit_file = "results/phase4_3/ablation_external_readiness.json"
            else:
                ckpt = f"checkpoints/phase4_1/cms_{2 if m=='A2' else 3}lvl_seed_{{seed}}.pt"
                audit_file = "results/phase4_2/rq2_statistical_audit.json"

            manifest["experiments"].append({
                "experiment_id": f"RQ2_{m}_{dset}",
                "rq": "RQ2",
                "method": m,
                "dataset": dset,
                "split": d_info["split"],
                "number_of_docs": d_info["documents_count"],
                "number_of_questions_or_items": d_info["items_count"],
                "seeds": [42, 43, 44],
                "checkpoint": ckpt,
                "training_samples": 200,
                "update_budget": "Strictly identical update events (delta = 0)",
                "retrieval_config": None,
                "context_size": 512,
                "quantization": "float16 (cuda)",
                "status": "NEED_EXTERNAL_GPU" if is_ext else "COMPLETED",
                "result_file": "results/phase4_1/rq2/rq2_raw_results.json" if not is_ext else None,
                "audit_file": audit_file,
                "external_gpu_requirement": is_ext
            })

    # 3. RQ3 Experiments (B1, B2, B5, P1, P2 on Vietnamese Context-Evicted 100)
    for m in ["B1", "B2", "B5", "P1", "P2"]:
        ckpt = "None" if m in ("B1", "B2") else "checkpoints/phase4_1/cms_3lvl_seed_{seed}.pt"
        manifest["experiments"].append({
            "experiment_id": f"RQ3_{m}_ContextEvicted100",
            "rq": "RQ3",
            "method": m,
            "dataset": "Vietnamese_Context_Evicted_100",
            "split": "test",
            "number_of_docs": 20,
            "number_of_questions_or_items": 100,
            "seeds": [42, 43, 44],
            "checkpoint": ckpt,
            "training_samples": 0 if m in ("B1", "B2") else 200,
            "update_budget": "Evicted during QA; 0 online QA steps",
            "retrieval_config": retrieval_frozen_config if m in ("B2", "P2") else None,
            "context_size": 512,
            "quantization": "float16 (cuda)",
            "status": "COMPLETED",
            "result_file": "results/phase4_1/rq3/rq3_raw_results.json",
            "audit_file": "results/phase4_2/rq3_consistency_audit.json",
            "external_gpu_requirement": False
        })

    # 4. RQ4 Experiments (B4, B5, P1 on Incremental Corpus D0 -> +5 -> +10 -> +20)
    for m in ["B4", "B5", "P1"]:
        manifest["experiments"].append({
            "experiment_id": f"RQ4_{m}_ContinualIngestion",
            "rq": "RQ4",
            "method": m,
            "dataset": "Incremental_QASPER_RQ4",
            "split": "test",
            "number_of_docs": 21,
            "number_of_questions_or_items": 10,
            "seeds": [42, 43, 44],
            "checkpoint": f"checkpoints/phase4_2/RQ4/rq4_{m}_seed_{{seed}}_{{interval}}.pt",
            "training_samples": 200,
            "update_budget": "Sequential document ingestion gradient updates",
            "retrieval_config": None,
            "context_size": 512,
            "quantization": "float16 (cuda)",
            "status": "NEED_EXTERNAL_GPU",
            "result_file": None,
            "audit_file": "results/phase4_3/rq4_external_readiness.json",
            "external_gpu_requirement": True
        })

    # 5. RQ5 Experiments (Cost Profiling: B1, B2, B4, B5, P1, P2)
    for m in ["B1", "B2", "B4", "B5", "P1", "P2"]:
        manifest["experiments"].append({
            "experiment_id": f"RQ5_{m}_CostProfile",
            "rq": "RQ5",
            "method": m,
            "dataset": "Standardized_Cost_Profile",
            "split": "test",
            "number_of_docs": 20,
            "number_of_questions_or_items": 1,
            "seeds": [42],
            "checkpoint": "None" if m in ("B1", "B2") else f"checkpoints/phase4_1/cms_{1 if m=='B4' else 3}lvl_seed_42.pt",
            "training_samples": 0 if m in ("B1", "B2") else 200,
            "update_budget": "Frozen inference latency & VRAM profiling",
            "retrieval_config": retrieval_frozen_config if m in ("B2", "P2") else None,
            "context_size": 512,
            "quantization": "float16 (cuda)",
            "status": "COMPLETED",
            "result_file": "results/phase4_1/non_training/rq5/rq5_cost_profile.csv",
            "audit_file": None,
            "external_gpu_requirement": False
        })

    # 6. Vietnamese Final Benchmark (All 350 questions: 300 answerable, 50 unanswerable)
    for m in ["B1", "B2", "B4", "B5", "P1", "P2"]:
        manifest["experiments"].append({
            "experiment_id": f"Vietnamese_Full_{m}",
            "rq": "Vietnamese_Benchmark",
            "method": m,
            "dataset": "Vietnamese",
            "split": "test",
            "number_of_docs": 20,
            "number_of_questions_or_items": 350,
            "seeds": [42, 43, 44],
            "checkpoint": "None" if m in ("B1", "B2") else f"checkpoints/phase4_1/cms_{1 if m=='B4' else 3}lvl_seed_{{seed}}.pt",
            "training_samples": 0 if m in ("B1", "B2") else 200,
            "update_budget": "0 online QA updates",
            "retrieval_config": retrieval_frozen_config if m in ("B2", "P2") else None,
            "context_size": 512,
            "quantization": "float16 (cuda)",
            "status": "COMPLETED",
            "result_file": "results/phase4_1/vietnamese/vietnamese_raw_results.json",
            "audit_file": "results/phase4_1_vietnamese_p2_b2_audit.json",
            "external_gpu_requirement": False
        })

    # 7. Excluded B3 Record
    manifest["experiments"].append({
        "experiment_id": "B3_Cartridges_Compression",
        "rq": "ALL",
        "method": "B3",
        "dataset": "ALL",
        "split": "test",
        "number_of_docs": 0,
        "number_of_questions_or_items": 0,
        "seeds": [],
        "checkpoint": None,
        "training_samples": 0,
        "update_budget": None,
        "retrieval_config": None,
        "context_size": 0,
        "quantization": None,
        "status": "EXCLUDED",
        "result_file": None,
        "audit_file": "configs/phase4_experiment.yaml",
        "external_gpu_requirement": False,
        "exclusion_reason": "Not reproducible within controlled student-scale hardware budget (4GB VRAM / single GPU constraint; requires heavy offline distillation and large teacher compute)."
    })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Master experiment manifest generated at {OUT_FILE}")
    print(f"Total experiment entries: {len(manifest['experiments'])}")

if __name__ == "__main__":
    build_manifest()
