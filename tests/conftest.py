"""
Global Pytest Configuration for SA-CMS & NL DocChat Test Suite.
Ensures tests execute against the authoritative inner workspace root regardless
of whether pytest is invoked from the outer IDE root or inner project directory.
"""

import os
import sys
from pathlib import Path

INNER_ROOT = Path(__file__).resolve().parent.parent
OUTER_CANDIDATE = INNER_ROOT.parent

# Remove outer root from sys.path if present so inner `src` package always wins
sys.path = [str(INNER_ROOT)] + [
    p for p in sys.path
    if Path(p or ".").resolve() not in (INNER_ROOT, OUTER_CANDIDATE)
]

if Path.cwd().resolve() != INNER_ROOT and (INNER_ROOT / "local_runtime").exists():
    os.chdir(str(INNER_ROOT))

