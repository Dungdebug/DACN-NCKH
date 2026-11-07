"""
Checkpoint saving and loading utilities for models and CMS memory states.
"""

from typing import Dict, Any, Optional
import os
import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    step: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Saves complete model checkpoint including weights, CMS memory states, and config metadata.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config or {},
        "metrics": metrics or {},
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    if hasattr(model, "get_all_memory_states"):
        state["cms_memory_states"] = model.get_all_memory_states()

    torch.save(state, save_path)
    return save_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Loads checkpoint into model and optionally optimizer.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if hasattr(model, "set_all_memory_states") and "cms_memory_states" in checkpoint:
        model.set_all_memory_states(checkpoint["cms_memory_states"])

    return checkpoint
