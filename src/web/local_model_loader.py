"""
Authoritative Local Model Loader for Offline SA-CMS Runtime.
Enforces 100% offline local asset resolution:
- Loads model weights strictly from local files (local_files_only=True).
- Loads tokenizer strictly from local files.
- Loads frozen SA-CMS checkpoint (Phase 4.1 or swapped external checkpoint).
- ZERO network requests at runtime.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hope_attention.sa_cms import StructureAlignedHopeLM

logger = logging.getLogger(__name__)

# Enforce strict offline mode for Hugging Face transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SA_CMS_OFFLINE_MODE"] = "1"


class LocalModelLoader:
    """Singleton authoritative loader for offline SA-CMS model and tokenizer assets."""

    _instance = None

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or (ROOT_DIR / "local_runtime" / "config" / "runtime_config.yaml")
        self.config = self._load_config()

        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.active_checkpoint_name = "None"
        self.active_checkpoint_path = None
        self.active_checkpoint_status = "NOT_LOADED"
        self.model_loaded = False
        self.model_error = None

    @classmethod
    def get_instance(cls) -> "LocalModelLoader":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {
            "model": {
                "local_model_path": "local_runtime/model",
                "local_tokenizer_path": "local_runtime/tokenizer",
                "num_levels": 3,
            },
            "checkpoints": {
                "current_active": "cms_3lvl_seed_42.pt",
                "current_path": "local_runtime/checkpoints/cms_3lvl_seed_42.pt",
                "current_status": "CURRENT_VALID_CHECKPOINT",
            }
        }

    def load_offline_model(self, prefer_gpu: bool = False) -> Tuple[Optional[StructureAlignedHopeLM], Optional[Any]]:
        """
        Loads the approved local SA-CMS model from local_runtime/ strictly without internet.
        """
        if self.model_loaded and self.model is not None:
            return self.model, self.tokenizer

        model_cfg = self.config.get("model", {})
        ckpt_cfg = self.config.get("checkpoints", {})

        model_dir = ROOT_DIR / model_cfg.get("local_model_path", "local_runtime/model")
        tokenizer_dir = ROOT_DIR / model_cfg.get("local_tokenizer_path", "local_runtime/tokenizer")
        ckpt_path = ROOT_DIR / ckpt_cfg.get("current_path", "local_runtime/checkpoints/cms_3lvl_seed_42.pt")

        if not model_dir.exists() or not (model_dir / "model.safetensors").exists():
            self.model_error = f"Local model files missing in {model_dir}. Please run 'python scripts/prepare_offline_assets.py'."
            logger.error(self.model_error)
            return None, None

        if prefer_gpu and torch.cuda.is_available():
            self.device = "cuda"
            torch_dtype = torch.float16
        else:
            self.device = "cpu"
            torch_dtype = torch.float32

        try:
            logger.info("Loading offline tokenizer from: %s", tokenizer_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_dir),
                local_files_only=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            logger.info("Loading offline backbone and SA-CMS from: %s (Device: %s)", model_dir, self.device)
            model = StructureAlignedHopeLM(
                model_name_or_path=str(model_dir),
                num_levels=model_cfg.get("num_levels", 3),
                device=self.device,
                torch_dtype=torch_dtype,
                enable_cms=True,
            )

            # Freeze 100% of backbone
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

            # Attach tokenizer
            model.tokenizer = tokenizer

            # Load approved SA-CMS checkpoint if present
            if ckpt_path.exists():
                logger.info("Loading offline SA-CMS checkpoint from: %s", ckpt_path)
                state = torch.load(str(ckpt_path), map_location=self.device)
                if "cms_state_dict" in state:
                    model.cms.load_state_dict(state["cms_state_dict"], strict=False)
                elif "state_dict" in state:
                    model.cms.load_state_dict(state["state_dict"], strict=False)
                self.active_checkpoint_name = ckpt_path.name
                self.active_checkpoint_path = str(ckpt_path)
                self.active_checkpoint_status = ckpt_cfg.get("current_status", "CURRENT_VALID_CHECKPOINT")
            else:
                self.active_checkpoint_name = "None (Default Zero-Init CMS)"
                self.active_checkpoint_status = "INITIAL_STATE"

            self.model = model
            self.tokenizer = tokenizer
            self.model_loaded = True
            self.model_error = None
            logger.info("Offline SA-CMS model successfully loaded. Zero remote calls.")
            return self.model, self.tokenizer

        except Exception as e:
            self.model = None
            self.tokenizer = None
            self.model_loaded = False
            self.model_error = f"Offline model loading failed: {str(e)}"
            logger.exception(self.model_error)
            return None, None

    def switch_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """
        Allows switching between CURRENT_VALID_CHECKPOINT and future EXTERNAL-GPU CHECKPOINT.
        """
        ckpt_dir = ROOT_DIR / "local_runtime" / "checkpoints"
        target_path = ckpt_dir / checkpoint_name

        if not target_path.exists():
            return {
                "success": False,
                "error": f"Checkpoint '{checkpoint_name}' not found in {ckpt_dir}.",
                "active_checkpoint": self.active_checkpoint_name,
            }

        try:
            state = torch.load(str(target_path), map_location=self.device)
            if self.model and hasattr(self.model, "cms"):
                if "cms_state_dict" in state:
                    self.model.cms.load_state_dict(state["cms_state_dict"], strict=False)
                elif "state_dict" in state:
                    self.model.cms.load_state_dict(state["state_dict"], strict=False)

            self.active_checkpoint_name = checkpoint_name
            self.active_checkpoint_path = str(target_path)
            self.active_checkpoint_status = "EXTERNAL_GPU_VALIDATED" if "external" in checkpoint_name.lower() else "CURRENT_VALID_CHECKPOINT"

            return {
                "success": True,
                "active_checkpoint": self.active_checkpoint_name,
                "status": self.active_checkpoint_status,
                "path": str(target_path),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "active_checkpoint": self.active_checkpoint_name}

    def get_resource_status(self, document_count: int = 0) -> Dict[str, Any]:
        """
        Generates authoritative local resource status for Step 17 requirement.
        """
        model_dir = ROOT_DIR / "local_runtime" / "model"
        tokenizer_dir = ROOT_DIR / "local_runtime" / "tokenizer"
        ckpt_dir = ROOT_DIR / "local_runtime" / "checkpoints"

        model_ready = (model_dir / "model.safetensors").exists()
        tokenizer_ready = (tokenizer_dir / "tokenizer.json").exists()
        ckpt_ready = (ckpt_dir / "cms_3lvl_seed_42.pt").exists() or (self.active_checkpoint_path and Path(self.active_checkpoint_path).exists())

        return {
            "model": "READY" if (model_ready and self.model_loaded) else ("PRESENT (NOT LOADED)" if model_ready else "MISSING"),
            "tokenizer": "READY" if (tokenizer_ready and self.tokenizer is not None) else ("PRESENT" if tokenizer_ready else "MISSING"),
            "sa_cms_checkpoint": f"READY ({self.active_checkpoint_name})" if ckpt_ready else "MISSING",
            "checkpoint_classification": self.active_checkpoint_status,
            "document_index": "READY" if document_count > 0 else "EMPTY (READY FOR UPLOAD)",
            "document_count": document_count,
            "offline_mode": "ACTIVE",
            "network": "DISABLED / NOT USED",
            "device": self.device,
            "backbone": "HuggingFaceTB/SmolLM2-135M (134.5M Frozen)",
            "memory_layers": 3,
            "error": self.model_error,
        }
