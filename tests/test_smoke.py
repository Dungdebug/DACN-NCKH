"""
Pytest integration for smoke test.
"""

try:
    from smoke_test import run_smoke_test
except ImportError:
    from scripts.experiments.smoke_test import run_smoke_test


def test_full_smoke_pipeline():
    assert run_smoke_test() is True
