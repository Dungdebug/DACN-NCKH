"""
Test Suite for Advanced RAG Mechanisms:
1. Multi-Query Decomposition & Rewriting (Đa truy vấn)
2. Dynamic Context Window / Sentence-Window Retrieval (Cửa sổ câu)
3. Chain-of-Thought (CoT) Prompt Template with Abstractive Synthesis
4. RAG-Tuned Instruct Models (Qwen-2.5-Instruct, Phi-3-Mini)
"""

import pytest

from src.hybrid_qa.document_store import DocumentRecord, Passage
from src.hybrid_qa.multi_query import MultiQueryExpander, MultiQueryRetriever
from src.hybrid_qa.prompt_template import (
    COT_MANDATORY_PROCESS_VI,
    COT_SYSTEM_INSTRUCTION_VI,
    GroundedPromptTemplate,
)
from src.hybrid_qa.retriever import RetrievalResult
from src.hybrid_qa.sentence_window import SentenceWindowExpander, SentenceWindowRetriever
from src.inference.generator import (
    SUPPORTED_SLMS,
    format_rag_instruct_prompt,
    get_rag_tuned_profile,
)
from src.web.document_analyzer import DocumentAnalyzer


# -----------------------------------------------------------------------------
# 1. Multi-Query Decomposition & Rewriting Tests
# -----------------------------------------------------------------------------

def test_multi_query_expansion_generates_diverse_queries():
    """
    Verifies that a broad query like 'Có bao nhiêu loại mã hóa?' generates
    3-5 distinct query variations (reformulations and sub-queries) covering
    classification, symmetric/asymmetric, and entity lists.
    """
    query = "Có bao nhiêu loại mã hóa?"
    queries = MultiQueryExpander.generate_queries(query, count=4)

    assert len(queries) >= 3
    assert queries[0] == query  # Original query preserved

    # At least one query explores classification / symmetry
    q_str = " ".join(queries).lower()
    assert any(term in q_str for term in ["phân loại", "đối xứng", "danh sách", "các loại"])


def test_multi_query_retriever_fuses_results_with_rrf():
    """
    Verifies that MultiQueryRetriever executes across all generated queries
    and uses Reciprocal Rank Fusion (RRF) to merge, deduplicate, and rank passages.
    """
    class MockBaseRetriever:
        def retrieve(self, q: str, top_k: int = 5):
            if "đối xứng" in q.lower():
                return [
                    RetrievalResult(passage_id="P_SYM", document_id="D1", text="Mã hóa đối xứng dùng chung 1 khóa bí mật.", score=9.0),
                    RetrievalResult(passage_id="P_COMMON", document_id="D1", text="Khái niệm mã hóa dữ liệu.", score=7.0),
                ]
            else:
                return [
                    RetrievalResult(passage_id="P_COMMON", document_id="D1", text="Khái niệm mã hóa dữ liệu.", score=8.5),
                    RetrievalResult(passage_id="P_ASYM", document_id="D1", text="Mã hóa bất đối xứng dùng cặp khóa công khai và bí mật.", score=8.0),
                ]

    mq_retriever = MultiQueryRetriever(base_retriever=MockBaseRetriever(), num_queries=3)
    results = mq_retriever.retrieve("Có bao nhiêu loại mã hóa?", top_k=3)

    assert len(results) >= 2
    # P_COMMON appears in both query branches so its RRF score is accumulated
    pids = [r.passage_id for r in results]
    assert "P_COMMON" in pids
    assert results[0].passage_id == "P_COMMON"
    assert results[0].metadata.get("retrieval_strategy") == "multi_query_rrf"
    assert len(results[0].metadata.get("multi_queries_matched", [])) >= 2


# -----------------------------------------------------------------------------
# 2. Dynamic Context Window / Sentence-Window Retrieval Tests
# -----------------------------------------------------------------------------

def test_sentence_window_expander_pulls_surrounding_sentences():
    """
    Verifies that SentenceWindowExpander takes a target chunk and pulls
    +2 sentences before and +2 sentences after without cutting off context.
    """
    full_doc = (
        "Mật mã học là ngành khoa học bảo mật thông tin. "
        "Nó đóng vai trò cốt lõi trong an toàn mạng máy tính. "
        "Mật mã Caesar dịch chuyển mỗi ký tự đi k vị trí. "
        "Công thức giải mã là P = (C - k) mod 26. "
        "Tuy nhiên nó có không gian khóa rất nhỏ chỉ 26 giá trị. "
        "Do đó phương pháp này dễ dàng bị thám mã bằng vét cạn. "
        "Các hệ mã hiện đại như AES đã thay thế hoàn toàn hệ mã này."
    )
    passage = "Mật mã Caesar dịch chuyển mỗi ký tự đi k vị trí."

    expanded = SentenceWindowExpander.expand_passage_window(
        passage_text=passage,
        full_text=full_doc,
        window_size=2,
    )

    # Must contain 2 sentences before
    assert "Mật mã học là ngành khoa học" in expanded
    assert "an toàn mạng máy tính" in expanded
    # Must contain the core sentence
    assert "Mật mã Caesar dịch chuyển" in expanded
    # Must contain 2 sentences after
    assert "P = (C - k) mod 26" in expanded
    assert "không gian khóa rất nhỏ" in expanded


def test_sentence_window_retriever_indexing_and_window_merging():
    """
    Verifies that SentenceWindowRetriever indexes a document into sentence nodes,
    finds the winning sentence, and returns a window with full surrounding context.
    """
    doc = DocumentRecord(
        document_id="DOC_CRYPTO",
        version=1,
        title="Tổng quan Mật mã học",
        raw_text=(
            "An toàn thông tin là nhu cầu cấp thiết hiện nay. "
            "Các kỹ thuật mã hóa đóng vai trò nền tảng. "
            "Mã hóa khóa đối xứng sử dụng cùng một khóa bí mật cho cả mã hóa và giải mã. "
            "Thuật toán tiêu biểu nhất là AES và DES. "
            "Ưu điểm của nó là tốc độ xử lý rất nhanh."
        ),
    )

    retriever = SentenceWindowRetriever(window_size=1)
    retriever.index_document(doc)

    results = retriever.retrieve("khóa đối xứng sử dụng khóa gì?", top_k=1)
    assert len(results) == 1
    res = results[0]

    assert res.metadata["retrieval_strategy"] == "sentence_window"
    assert "khóa đối xứng" in res.text
    # Window size 1 should include preceding sentence and succeeding sentence
    assert "kỹ thuật mã hóa đóng vai trò nền tảng" in res.text
    assert "Thuật toán tiêu biểu nhất là AES" in res.text


# -----------------------------------------------------------------------------
# 3. Chain-of-Thought (CoT) Prompt Template Tests
# -----------------------------------------------------------------------------

def test_cot_grounded_prompt_template_exact_structure():
    """
    Verifies that the CoT Prompt Template adheres to the exact requested structure:
    [Chỉ thị Hệ thống], [Ngữ cảnh từ tài liệu], [Câu hỏi của người dùng], [Quy trình xử lý bắt buộc].
    """
    passages = [
        {
            "passage_id": "P01",
            "section_title": "Phân loại mã hóa",
            "text": "Tài liệu đề cập đến 2 loại mã hóa chính: mã hóa đối xứng và mã hóa bất đối xứng.",
        }
    ]
    query = "Có bao nhiêu loại mã hóa chính?"

    # Plain text format
    plain_cot = GroundedPromptTemplate.build_cot_prompt(
        question=query,
        context_passages=passages,
        language="vi",
        format_style="plain",
    )

    assert "[Chỉ thị Hệ thống]" in plain_cot
    assert COT_SYSTEM_INSTRUCTION_VI in plain_cot
    assert "[Ngữ cảnh từ tài liệu]" in plain_cot
    assert "[Câu hỏi của người dùng]" in plain_cot
    assert "[Quy trình xử lý bắt buộc]" in plain_cot
    assert "Tuyệt đối không chép phạt" in plain_cot
    assert "Mở bài: Trực tiếp trả lời thẳng" in plain_cot
    assert "Thân bài: Phân tích chi tiết" in plain_cot
    assert "Trích dẫn: Chèn số trích dẫn [1], [2]" in plain_cot
    assert query in plain_cot


def test_cot_grounded_prompt_chatml_and_phi3_formats():
    """
    Verifies that CoT Prompt Template correctly translates into ChatML (<|im_start|>)
    and Phi-3 (<|system|>) templates for RAG-tuned instruction models.
    """
    query = "Hệ thống SA-CMS là gì?"
    context = "[P1] Hệ thống SA-CMS là bộ nhớ tham số liên tục đa thang."

    # ChatML (Qwen-2.5-Instruct)
    chatml_prompt = GroundedPromptTemplate.build_cot_prompt(
        question=query,
        context_text=context,
        format_style="chatml",
    )
    assert "<|im_start|>system" in chatml_prompt
    assert "<|im_start|>user" in chatml_prompt
    assert "<|im_start|>assistant" in chatml_prompt
    assert "[Quy trình xử lý bắt buộc]" in chatml_prompt

    # Phi-3 (Phi-3-Mini-4K-Instruct)
    phi3_prompt = GroundedPromptTemplate.build_cot_prompt(
        question=query,
        context_text=context,
        format_style="phi3",
    )
    assert "<|system|>" in phi3_prompt
    assert "<|user|>" in phi3_prompt
    assert "<|assistant|>" in phi3_prompt


# -----------------------------------------------------------------------------
# 4. RAG-Tuned Instruct Models & Formatting Tests
# -----------------------------------------------------------------------------

def test_rag_tuned_models_profiles_and_prompt_formatting():
    """
    Verifies that Qwen-2.5-Instruct and Phi-3-Mini profiles are properly registered
    and automatically format prompts into their respective instruction formats.
    """
    # 1. Qwen-2.5-1.5B-Instruct
    qwen_prof = get_rag_tuned_profile("qwen-2.5-1.5b")
    assert qwen_prof.name == "Qwen-2.5-1.5B-Instruct"
    assert qwen_prof.chat_template_format == "chatml"

    qwen_prompt = format_rag_instruct_prompt(
        profile=qwen_prof,
        question="Mã hóa AES là gì?",
        context_text="AES là chuẩn mã hóa dữ liệu tiên tiến.",
        cot=True,
    )
    assert "<|im_start|>system" in qwen_prompt
    assert "<|im_start|>user" in qwen_prompt
    assert "<|im_start|>assistant" in qwen_prompt

    # 2. Phi-3-Mini-Instruct
    phi3_prof = get_rag_tuned_profile("phi-3")
    assert phi3_prof.name == "Phi-3-Mini-4K-Instruct"
    assert phi3_prof.chat_template_format == "phi3"

    phi3_prompt = format_rag_instruct_prompt(
        profile=phi3_prof,
        question="Mã hóa RSA hoạt động ra sao?",
        context_text="RSA là thuật toán mã hóa công khai.",
        cot=True,
    )
    assert "<|system|>" in phi3_prompt
    assert "<|assistant|>" in phi3_prompt


# -----------------------------------------------------------------------------
# 5. DocumentAnalyzer End-to-End Integration
# -----------------------------------------------------------------------------

def test_document_analyzer_integration_with_multi_query_and_window():
    """
    Verifies that DocumentAnalyzer exposes helper methods for multi-query expansion,
    sentence-window expansion, and CoT prompt synthesis.
    """
    class MockPassageObj:
        def __init__(self, pid, text):
            self.passage_id = pid
            self.text = text

    full_text = (
        "Khoa học mật mã bảo vệ thông tin trực tuyến. "
        "Mã hóa đối xứng sử dụng chung một khóa bí mật. "
        "Ví dụ điển hình là DES và AES. "
        "Nó có tốc độ rất nhanh so với mã hóa công khai."
    )
    passages = [MockPassageObj("P1", "Mã hóa đối xứng sử dụng chung một khóa bí mật.")]

    # 1. Multi-query decomposition
    queries = DocumentAnalyzer.decompose_and_expand_query("Mã hóa đối xứng là gì?", count=3)
    assert len(queries) >= 3

    # 2. Window expansion
    expanded = DocumentAnalyzer.expand_evidence_windows(passages, full_raw_text=full_text, window_size=1)
    assert len(expanded) == 1
    assert "Khoa học mật mã bảo vệ" in expanded[0].text
    assert "Ví dụ điển hình là DES" in expanded[0].text

    # 3. CoT prompt generation
    cot_prompt = DocumentAnalyzer.build_cot_grounded_prompt(
        query="Mã hóa đối xứng là gì?",
        evidence_passages=expanded,
        format_style="plain",
    )
    assert "[Chỉ thị Hệ thống]" in cot_prompt
    assert "[Ngữ cảnh từ tài liệu]" in cot_prompt
    assert "[Quy trình xử lý bắt buộc]" in cot_prompt
