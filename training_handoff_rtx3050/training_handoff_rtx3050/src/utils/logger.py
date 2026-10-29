"""
Structured logging utility for experiments and verification runs.
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, Any


def setup_logger(name: str = "hope_nckh", log_dir: str = "logs", filename: str = None) -> logging.Logger:
    """
    Sets up a logger with both console and file handlers.
    """
    os.makedirs(log_dir, exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.log"

    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (UTF-8 safe)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Logger initialized. File sink: {log_path}")
    return logger


def save_metrics_json(metrics: Dict[str, Any], filepath: str) -> None:
    """Saves metrics dictionary to formatted JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
