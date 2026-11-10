"""
Smoke Test for Hope-Attention Baseline (Phase 1).
code train :python external_gpu/phase4_4/run_all.py

Verifies the 6 mandatory gates:
1. Model initialization
2. Memory initialization
3. Forward pass execution
4. Memory update execution (Equation 71)
5. Checkpoint save/load verification
6. Inference execution
"""

import os
import sys
import tempfile
import torch
import yaml

from src.hope_attention.model import HopeAttentionLM
from src.inference.generator import TextGenerator
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logger import setup_logger


def run_smoke_test(config_path: str = "configs/baseline_config.yaml") -> bool:
    logger = setup_logger(name="smoke_test", log_dir="logs")
    logger.info("=== STARTING PHASE 1 SMOKE TEST ===")

    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["training"]["device"] == "cuda" else "cpu")
    logger.info(f"Target execution device: {device}")

    # 1. Model Initialization
    logger.info("[Gate 1/6] Verifying model initialization...")
    model_cfg = cfg["model"]
    cms_cfg = cfg["cms"]
    model = HopeAttentionLM(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        num_levels=cms_cfg["num_levels"],
        lowest_chunk_size=cms_cfg["lowest_chunk_size"],
        base_lr=cms_cfg["base_lr"],
        dropout=model_cfg["dropout"],
        chain_type=cms_cfg["chain_type"],
        max_seq_len=model_cfg["max_seq_len"],
        tie_weights=model_cfg["tie_weights"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f" Model initialized successfully. Total parameters: {total_params:,}")

    # 2. Memory Initialization
    logger.info("[Gate 2/6] Verifying memory initialization...")
    first_block_cms = model.blocks[0].cms
    logger.info(f" CMS Levels: {first_block_cms.num_levels}")
    logger.info(f" Chunk sizes: {first_block_cms.schedule.chunk_sizes}")
    logger.info(f" Frequencies: {first_block_cms.schedule.frequencies}")
    assert len(first_block_cms.schedule.chunk_sizes) == cms_cfg["num_levels"]
    assert len(first_block_cms._initial_state_dict) > 0
    logger.info(" Memory initialized and base theta_0 cached.")

    # 3. Forward Pass
    logger.info("[Gate 3/6] Verifying forward pass...")
    batch_size = 2
    seq_len = 32
    dummy_input = torch.randint(0, model_cfg["vocab_size"], (batch_size, seq_len), device=device)
    dummy_target = torch.randint(0, model_cfg["vocab_size"], (batch_size, seq_len), device=device)

    logits, loss, _ = model(dummy_input, targets=dummy_target)
    assert logits.shape == (batch_size, seq_len, model_cfg["vocab_size"])
    assert loss is not None and not torch.isnan(loss)
    logger.info(f" Forward pass successful. Output shape: {logits.shape}, Loss: {loss.item():.4f}")

    # 4. Memory Update (Equation 71)
    logger.info("[Gate 4/6] Verifying online memory update (Equation 71)...")
    # Take initial weights of the fastest level in Layer 0
    fastest_level_idx = cms_cfg["num_levels"] - 1
    w_before = model.blocks[0].cms.chain.blocks[fastest_level_idx].w1.weight.clone()

    # Step by 64 tokens (sufficient to trigger all levels)
    layer_updates = model.update_cms_online(loss, num_tokens=64)
    w_after = model.blocks[0].cms.chain.blocks[fastest_level_idx].w1.weight

    diff = torch.norm(w_after - w_before).item()
    assert diff > 1e-5, f"Weights did not change after update! Diff: {diff}"
    logger.info(f" Equation 71 update executed. Parameter norm change: {diff:.6f}")
    logger.info(f" Layer update events: {layer_updates}")

    # Verify reset_memory
    model.reset_memory()
    w_reset = model.blocks[0].cms.chain.blocks[fastest_level_idx].w1.weight
    reset_diff = torch.norm(w_reset - w_before).item()
    assert reset_diff == 0.0, f"Memory reset failed to restore theta_0! Diff: {reset_diff}"
    logger.info(" Memory reset restored initial theta_0 exactly.")

    # 5. Checkpoint Save/Load
    logger.info("[Gate 5/6] Verifying checkpoint save and load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "smoke_checkpoint.pt")
        save_checkpoint(
            model=model,
            save_path=ckpt_path,
            step=1,
            config=cfg,
            metrics={"smoke_loss": loss.item()},
        )
        assert os.path.exists(ckpt_path)
        logger.info(f" Checkpoint saved to: {ckpt_path}")

        # Fresh model to load into
        fresh_model = HopeAttentionLM(
            vocab_size=model_cfg["vocab_size"],
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            d_ff=model_cfg["d_ff"],
            num_levels=cms_cfg["num_levels"],
            lowest_chunk_size=cms_cfg["lowest_chunk_size"],
            base_lr=cms_cfg["base_lr"],
            dropout=model_cfg["dropout"],
            chain_type=cms_cfg["chain_type"],
            max_seq_len=model_cfg["max_seq_len"],
            tie_weights=model_cfg["tie_weights"],
        ).to(device)

        loaded_data = load_checkpoint(ckpt_path, fresh_model, device=device)
        assert loaded_data["step"] == 1
        assert "smoke_loss" in loaded_data["metrics"]

        # Check weight equality
        w_orig = model.embedding.tok_emb.weight
        w_loaded = fresh_model.embedding.tok_emb.weight
        ckpt_diff = torch.norm(w_orig - w_loaded).item()
        assert ckpt_diff == 0.0, f"Loaded weights differ from saved weights! Diff: {ckpt_diff}"
        logger.info(" Checkpoint loaded and validated with exact parity.")

    # 6. Inference
    logger.info("[Gate 6/6] Verifying inference execution...")
    generator = TextGenerator(model, device=device)
    prompt = torch.tensor([[10, 20, 30, 40]], dtype=torch.long, device=device)
    generated = generator.generate(prompt, max_new_tokens=16, temperature=0.0)
    assert generated.shape == (1, 4 + 16)
    logger.info(f" Inference successful. Generated sequence: {generated.cpu().tolist()[0]}")

    logger.info("=== ALL 6 SMOKE TEST GATES PASSED! ===")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    if not success:
        sys.exit(1)
