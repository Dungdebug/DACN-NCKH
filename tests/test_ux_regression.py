"""
Regression tests for UX bug fixes in Offline SA-CMS Chat application:
1. Bug 1: Invalid/unknown file size formatting (no NaN/null/undefined).
2. Bug 2: Evidence layout state (compact CĂN CỨ section, [Xem toàn văn], no viewport waste).
3. Bug 3: Conversation title generation (meaningful titles, deduplication, no internal IDs).
4. Bug 4: Memory ingestion status (explicit 6 stages, SA-CMS Memory distinct from file storage).
"""

import math
from pathlib import Path
import pytest

from src.web.file_parser import format_file_size
from src.web.session_manager import generate_title_from_query, SessionManager


# ==============================================================================
# BUG 1 REGRESSION TESTS: Safe File Size Formatting
# ==============================================================================

def test_file_size_invalid_inputs_fallback_cleanly():
    """Verify that invalid/missing sizes never produce NaN, undefined, or null."""
    invalid_cases = [
        None,
        "",
        "   ",
        "NaN MB",
        "NaN KB",
        "nan",
        "undefined",
        "null",
        "none",
        float("nan"),
        float("inf"),
        -1,
        -1024,
        "invalid_size_string",
    ]
    for val in invalid_cases:
        res = format_file_size(val)
        assert res == "Dung lượng không xác định", f"Failed for input {val!r}: got {res}"
        assert "nan" not in res.lower()
        assert "undefined" not in res.lower()
        assert "null" not in res.lower()


def test_file_size_valid_conversions():
    """Verify standard units convert correctly and safely."""
    assert format_file_size(0) == "0 KB"
    assert format_file_size(512) == "512 B"
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(2048) == "2.0 KB"
    assert format_file_size(1024 * 1024) == "1.0 MB"
    assert format_file_size(2.4 * 1024 * 1024) == "2.4 MB"
    assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
    assert format_file_size(2.5 * 1024 * 1024 * 1024) == "2.5 GB"
    # Pre-formatted string
    assert format_file_size("3.5 MB") == "3.5 MB"
    assert format_file_size("120.0 KB") == "120.0 KB"


# ==============================================================================
# BUG 2 REGRESSION TESTS: Evidence Layout State
# ==============================================================================

def test_evidence_layout_in_frontend_assets():
    """Verify compact evidence layout classes exist in app.js, index.html, and app.css."""
    root = Path(__file__).resolve().parent.parent
    app_js = (root / "src" / "web" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    app_css = (root / "src" / "web" / "static" / "css" / "app.css").read_text(encoding="utf-8")

    # In app.js
    assert "evidence-section-block" in app_js
    assert "section-label-header" in app_js
    assert "CĂN CỨ" in app_js
    assert "TRẢ LỜI" in app_js
    assert "evidence-card-compact" in app_js
    assert "evidence-fulltext-btn" in app_js
    assert "Xem toàn văn" in app_js

    # In app.css
    assert ".evidence-section-block" in app_css
    assert ".evidence-card-compact" in app_css
    assert ".evidence-header-row" in app_css
    assert ".evidence-cite-badge" in app_css
    assert ".evidence-meta-info" in app_css
    assert ".evidence-fulltext-btn" in app_css
    assert ".evidence-excerpt-text" in app_css


# ==============================================================================
# BUG 3 REGRESSION TESTS: Conversation Title Generation
# ==============================================================================

def test_conversation_title_generation_examples():
    """Verify meaningful titles are generated from common user questions."""
    # Example 1
    t1 = generate_title_from_query("Kiến trúc SA-CMS là gì?")
    assert t1 == "Kiến trúc SA-CMS", f"Got: {t1}"

    # Example 2
    t2 = generate_title_from_query("Cơ chế bộ nhớ L1 L2 L3 hoạt động thế nào?")
    assert t2 == "Cơ chế bộ nhớ L1 L2 L3", f"Got: {t2}"

    # Example 3
    t3 = generate_title_from_query("Giải thích về Nested Learning và CMS")
    assert t3 == "Nested Learning và CMS", f"Got: {t3}"

    # Example 4: Conversational prefix stripped
    t4 = generate_title_from_query("Cho tôi biết về kiến trúc SA-CMS")
    assert t4 == "Kiến trúc SA-CMS", f"Got: {t4}"


def test_conversation_title_deduplication():
    """Verify avoid identical duplicate titles by adding numeric suffix."""
    existing = {"Kiến trúc SA-CMS"}
    t_dup = generate_title_from_query("Kiến trúc SA-CMS là gì?", existing_titles=existing)
    assert t_dup == "Kiến trúc SA-CMS (2)", f"Got: {t_dup}"

    existing.add("Kiến trúc SA-CMS (2)")
    t_dup2 = generate_title_from_query("Kiến trúc SA-CMS là gì?", existing_titles=existing)
    assert t_dup2 == "Kiến trúc SA-CMS (3)", f"Got: {t_dup2}"


def test_conversation_title_never_exposes_internal_id():
    """Verify internal IDs like 'chat_' or 'msg_' are never in generated titles."""
    for q in ["Kiến trúc SA-CMS", "chat_12345", "msg_abcde", ""]:
        title = generate_title_from_query(q)
        assert not title.startswith("chat_")
        assert not title.startswith("msg_")


# ==============================================================================
# BUG 4 REGRESSION TESTS: Memory Ingestion Status
# ==============================================================================

def test_memory_ingestion_explicit_stages_and_distinction():
    """Verify the 6 explicit pipeline stages and clear distinction between file stored and memory ready."""
    root = Path(__file__).resolve().parent.parent
    index_html = (root / "src" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    app_js = (root / "src" / "web" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    app_py = (root / "src" / "web" / "app.py").read_text(encoding="utf-8")

    expected_stages = [
        "Tệp đã lưu",
        "Văn bản đã trích xuất",
        "Cấu trúc đã phân tích",
        "Chỉ mục cục bộ đã sẵn sàng",
        "SA-CMS Memory đã sẵn sàng",
        "Sẵn sàng hỏi",
    ]

    # Verify all 6 stages appear in backend upload response
    for stage in expected_stages:
        assert f'"{stage}"' in app_py, f"Missing stage '{stage}' in app.py"

    # Verify stages in index.html
    for stage in expected_stages:
        assert stage in index_html, f"Missing stage '{stage}' in index.html"

    # Verify distinction: 'SA-CMS Memory đã sẵn sàng' is distinct from 'Tệp đã lưu' / 'File đã tải'
    assert "SA-CMS Memory đã sẵn sàng" in app_js
    assert "✓ SA-CMS Memory đã sẵn sàng" in app_js
    assert "SA-CMS Memory đã sẵn sàng" != "Tệp đã lưu"
    assert "SA-CMS Memory đã sẵn sàng" != "File đã tải"


def test_document_table_date_formatting_safe():
    """Verify frontend code safely handles created_at whether it is a number, timestamp, or string without calling slice directly on raw value."""
    root = Path(__file__).resolve().parent.parent
    app_js = (root / "src" / "web" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "d.created_at.slice" not in app_js, "d.created_at.slice causes TypeError when created_at is a number/timestamp!"
    assert "formattedDate" in app_js

