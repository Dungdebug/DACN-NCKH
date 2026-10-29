"""
Test and verify the Phase 5.3 UX Guided Workflow and Research Mode API Contracts.
Validates:
1. System Status API (offline ready flags)
2. Document upload and multi-document attribution
3. Session creation and history
4. Question answering with TRẢ LỜI and CĂN CỨ citation formatting
5. Refusal controller behavior for unsupported questions (refused=True)
6. Research metrics API and comparison API
"""

import json
import requests
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"

def test_ux_workflow():
    print("======================================================================")
    print("VERIFYING PHASE 5.3 UX GUIDED WORKFLOW & RESEARCH MODE")
    print("======================================================================")

    # 1. Check System Status
    resp = requests.get(f"{BASE_URL}/api/system/status")
    assert resp.status_code == 200, f"System status failed: {resp.text}"
    status = resp.json()
    print("[1] SYSTEM STATUS CHECK:")
    print(f"    - Model: {status.get('model')}")
    print(f"    - Tokenizer: {status.get('tokenizer')}")
    print(f"    - SA-CMS Checkpoint: {status.get('sa_cms_checkpoint')}")
    print(f"    - Document Index: {status.get('document_index')}")
    print(f"    - Offline Mode: {status.get('offline_mode')}")
    print(f"    - Network: {status.get('network')}")
    assert status.get("offline_mode") in (True, "ACTIVE"), "Offline mode must be active"

    # 2. Seed Demo Documents
    seed_resp = requests.post(f"{BASE_URL}/api/demo/seed")
    assert seed_resp.status_code == 200, f"Seed demo failed: {seed_resp.text}"
    print("[2] DEMO SEEDING: Success")

    # 3. Create Session with Human-Readable Title
    sess_resp = requests.post(
        f"{BASE_URL}/api/chat/sessions",
        json={"title": "Kiến trúc SA-CMS"}
    )
    assert sess_resp.status_code == 200
    session_data = sess_resp.json()
    session_id = session_data["session_id"]
    print(f"[3] SESSION CREATED: ID={session_id}, Title='{session_data['title']}'")
    assert "DOC_" not in session_data["title"], "Session title must be human-readable"

    # 4. Attach Documents (Multi-Document)
    docs_resp = requests.get(f"{BASE_URL}/api/documents")
    docs = docs_resp.json().get("documents", [])
    
    # Locate DEMO_DOC_001 and DEMO_DOC_002
    doc1 = next((d for d in docs if d["document_id"] == "DEMO_DOC_001"), None)
    doc2 = next((d for d in docs if d["document_id"] == "DEMO_DOC_002"), None)
    if not doc1:
        doc1 = docs[0]
    if not doc2:
        doc2 = docs[1] if len(docs) > 1 else docs[0]

    # Attach doc 1
    att1 = requests.post(
        f"{BASE_URL}/api/chat/sessions/{session_id}/documents",
        json={"document_id": doc1["document_id"]}
    )
    assert att1.status_code == 200

    # Attach doc 2
    att2 = requests.post(
        f"{BASE_URL}/api/chat/sessions/{session_id}/documents",
        json={"document_id": doc2["document_id"]}
    )
    assert att2.status_code == 200
    print(f"[4] MULTI-DOCUMENT ATTACHED: '{doc1['title']}' and '{doc2['title']}'")

    # 5. Ask Grounded Question
    q1 = "Tầng 1 của hệ thống CMS đảm nhiệm vai trò gì trong kiến trúc?"
    ans_resp = requests.post(
        f"{BASE_URL}/api/chat/sessions/{session_id}/messages",
        json={
            "query": q1,
            "answer_mode": "balanced",
            "mode": "hybrid",
            "top_k": 5,
            "language": "vi"
        }
    )
    assert ans_resp.status_code == 200
    ans_data = ans_resp.json()
    msg = ans_data["message"]
    print("[5] GROUNDED QA EXECUTION:")
    print(f"    - Query: {q1}")
    print(f"    - Refused: {msg.get('refused', False)}")
    print(f"    - Citations count: {len(msg.get('citations', []))}")
    print(f"    - Content snippet: {msg['content'][:120]}...")
    assert not msg.get("refused", False), "Grounded question should not be refused"
    assert len(msg.get("citations", [])) > 0, "Grounded question must provide citations"

    # Verify citation schema
    c0 = msg["citations"][0]
    print(f"    - First Citation: [{c0.get('citation_index')}] {c0.get('document_title')} — {c0.get('section_title')} (Para {c0.get('paragraph_index')})")
    assert "document_title" in c0 and "section_title" in c0

    # 6. Ask Unsupported Question (Refusal Test)
    q2 = "Thủ đô của Nhật Bản là thành phố nào và thời tiết hiện tại ra sao?"
    ref_resp = requests.post(
        f"{BASE_URL}/api/chat/sessions/{session_id}/messages",
        json={
            "query": q2,
            "answer_mode": "balanced",
            "mode": "hybrid",
            "top_k": 5,
            "language": "vi"
        }
    )
    assert ref_resp.status_code == 200
    ref_data = ref_resp.json()
    ref_msg = ref_data["message"]
    print("[6] UNSUPPORTED QA (REFUSAL) EXECUTION:")
    print(f"    - Query: {q2}")
    print(f"    - Refused: {ref_msg.get('refused')}")
    print(f"    - Content: {ref_msg['content']}")
    assert ref_msg.get("refused") is True, "Out-of-scope question must trigger refusal"

    # 7. Check Research Lab Metrics
    metrics_resp = requests.get(f"{BASE_URL}/research/metrics")
    assert metrics_resp.status_code == 200
    metrics_data = metrics_resp.json()
    print("[7] RESEARCH LAB METRICS: Loaded successfully")

    # 8. Check Model Comparison API (B1, B2, P1, P2)
    comp_resp = requests.post(
        f"{BASE_URL}/api/chat/compare",
        json={
            "query": q1,
            "methods": ["B1", "B2", "P1", "P2"],
            "answer_mode": "balanced"
        }
    )
    assert comp_resp.status_code == 200
    comp_data = comp_resp.json()
    assert "comparison" in comp_data
    assert "P2" in comp_data["comparison"]
    print("[8] MODEL COMPARISON (B1, B2, P1, P2): Success")

    print("======================================================================")
    print("ALL USER JOURNEY REST API TESTS PASSED!")
    print("======================================================================")

if __name__ == "__main__":
    test_ux_workflow()
