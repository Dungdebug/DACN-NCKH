"""Fetch a large, strictly benchmark-disjoint online training corpus via public APIs.

Reuses HTTP and parsing helpers from ``src.evaluation.online_corpus_fetcher`` and
enforces 0% overlap (by ``document_id`` and normalized ``title``) against
``data/online_benchmark_corpus.json``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.online_corpus_fetcher import (
    _build_abstract_paper_questions,
    _build_arxiv_questions,
    _build_wiki_questions,
    _fetch_hf_rows,
    _parse_pubmedqa_rows,
    _parse_qasper_rows,
    _parse_sciq_rows,
    _parse_squad_v2_rows,
    fetch_arxiv_papers,
    fetch_europepmc_papers,
    fetch_openalex_papers,
    fetch_wikipedia_article,
)

# Strictly distinct from _VI_WIKI_TOPICS in src/evaluation/online_corpus_fetcher.py
_TRAIN_VI_WIKI_TOPICS = [
    "Mạng nơ-ron tích chập", "Mạng nơ-ron hồi quy", "Biến đổi Fourier",
    "Giải thuật Dijkstra", "Quy hoạch động", "Cây nhị phân tìm kiếm",
    "Ngôn ngữ lập trình C++", "Ngôn ngữ lập trình Java", "JavaScript",
    "Hệ quản trị cơ sở dữ liệu", "SQL", "NoSQL", "Docker", "Kubernetes",
    "Giao thức TCP/IP", "HTTP", "DNS", "Tường lửa (mạng máy tính)",
    "Mã hóa đối xứng", "AES (mật mã học)", "RSA (mật mã học)",
    "Xác thực hai yếu tố", "OAuth", "Trí tuệ nhân tạo tạo sinh",
    "Xử lý tiếng nói", "Nhận dạng giọng nói", "Dịch máy",
    "Tóm tắt văn bản", "Phân tích cảm xúc", "Biểu diễn tri thức",
    "Văn hóa Việt Nam", "Ẩm thực Việt Nam", "Địa lý Việt Nam",
    "Kinh tế Việt Nam", "Cách mạng công nghiệp lần thứ tư",
    "Vũ trụ học", "Lỗ đen", "Cơ học lượng tử", "Hóa học vô cơ",
    "Sinh học tế bào", "Thần kinh học", "Tâm lý học nhận thức",
    "Triết học", "Logic học", "Thống kê Bayes", "Phân tích hồi quy",
    "Kỹ thuật dữ liệu", "Kiến trúc máy tính", "Bộ vi xử lý", "GPU",
]

# Strictly distinct from _EN_WIKI_TOPICS in src/evaluation/online_corpus_fetcher.py
_TRAIN_EN_WIKI_TOPICS = [
    "Convolutional neural network", "Recurrent neural network", "Long short-term memory",
    "Generative pre-trained transformer", "Autoencoder", "Variational autoencoder",
    "Generative adversarial network", "Diffusion model", "Mixture of experts",
    "Low-rank adaptation", "Prompt engineering", "Zero-shot learning",
    "Few-shot learning", "Transfer learning", "Self-supervised learning",
    "Contrastive learning", "Meta-learning", "Active learning (machine learning)",
    "Feature engineering", "Hyperparameter optimization", "Cross-validation (statistics)",
    "Regularization (mathematics)", "Dropout (neural networks)", "Batch normalization",
    "Layer normalization", "Softmax function", "Cross-entropy",
    "Cosine similarity", "Euclidean distance", "Principal component analysis",
    "Singular value decomposition", "K-means clustering", "Hierarchical clustering",
    "Decision tree learning", "Gradient boosting", "XGBoost",
    "Natural language understanding", "Named-entity recognition", "Part-of-speech tagging",
    "Semantic search", "Dense retrieval", "BM25",
    "Transport Layer Security", "Advanced Encryption Standard", "Elliptic-curve cryptography",
    "Intrusion detection system", "Vulnerability (computing)", "Penetration test",
    "Microservices", "REST",
]

# Diverse AI / LLM / Memory / RAG / Security queries on arXiv
_TRAIN_ARXIV_QUERIES = [
    ("ti:\"large language model\" AND cat:cs.CL", 25),
    ("ti:\"retrieval-augmented generation\" OR ti:\"retrieval augmented\"", 25),
    ("ti:\"continual learning\" OR ti:\"catastrophic forgetting\"", 25),
    ("ti:\"memory\" AND (cat:cs.LG OR cat:cs.CL)", 25),
    ("ti:\"long context\" OR ti:\"context compression\"", 25),
    ("ti:\"machine unlearning\" OR ti:\"unlearning\"", 25),
    ("cat:cs.CR AND (\"language model\" OR \"privacy\" OR \"encryption\")", 25),
    ("ti:\"question answering\" AND cat:cs.CL", 25),
]

_TRAIN_OPENALEX_QUERIES = [
    ("retrieval augmented generation large language models", 30),
    ("continual learning neural memory architecture", 30),
    ("machine unlearning differential privacy", 30),
    ("natural language processing question answering", 30),
    ("knowledge graph vector database semantic search", 30),
    ("transformer attention long context compression", 30),
]

_TRAIN_EUROPEPMC_QUERIES = [
    ("clinical natural language processing biomedical", 30),
    ("machine learning medical diagnosis transformer", 30),
    ("biomedical question answering retrieval", 30),
    ("genomics deep learning sequence modeling", 30),
    ("drug discovery neural network prediction", 30),
    ("electronic health records language model", 30),
]


def _fetch_hf_rows_retry(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    timeout: int = 30,
    retries: int = 4,
) -> List[Dict[str, Any]]:
    for attempt in range(retries):
        rows = _fetch_hf_rows(dataset, config, split, offset, length, timeout=timeout)
        if rows:
            return rows
        time.sleep(0.8 * (attempt + 1))
    return []


def _load_benchmark_exclusions(benchmark_path: Path) -> tuple[Set[str], Set[str]]:
    if not benchmark_path.exists():
        return set(), set()
    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench = json.load(f)
    doc_ids = {str(d.get("document_id", "")).strip() for d in bench.get("documents", []) if d.get("document_id")}
    titles = {str(d.get("title", "")).strip().lower() for d in bench.get("documents", []) if d.get("title")}
    return doc_ids, titles


def build_online_training_corpus(
    output_path: Path = Path("data/online_training_corpus.json"),
    benchmark_path: Path = Path("data/online_benchmark_corpus.json"),
    max_workers: int = 10,
    qasper_rows_total: int = 200,
    squad_rows_total: int = 3500,
    sciq_rows_total: int = 600,
    pubmedqa_rows_total: int = 400,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    bench_doc_ids, bench_titles = _load_benchmark_exclusions(benchmark_path)
    print(f"[Exclusion] Loaded {len(bench_doc_ids)} doc_ids and {len(bench_titles)} titles from {benchmark_path}")

    documents: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []
    seen_doc_ids: Set[str] = set()
    seen_titles: Set[str] = set()
    skipped_overlap = 0

    def _add_doc(doc: Optional[Dict[str, Any]]) -> bool:
        nonlocal skipped_overlap
        if not doc:
            return False
        doc_id = str(doc.get("document_id", "")).strip()
        title_norm = str(doc.get("title", "")).strip().lower()
        if not doc_id or not title_norm:
            return False
        if doc_id in bench_doc_ids or title_norm in bench_titles:
            skipped_overlap += 1
            return False
        if doc_id in seen_doc_ids:
            return False
        if doc.get("source") not in {"squad_v2", "sciq"} and title_norm in seen_titles:
            return False
        seen_doc_ids.add(doc_id)
        seen_titles.add(title_norm)
        documents.append(doc)
        return True

    qasper_offsets = list(range(0, qasper_rows_total, 100))
    squad_offsets = list(range(0, squad_rows_total, 100))
    sciq_offsets = list(range(0, sciq_rows_total, 100))
    pubmedqa_offsets = list(range(0, pubmedqa_rows_total, 100))

    print(
        f"[Fetch] Launching ThreadPoolExecutor(max_workers={max_workers}): "
        f"{len(qasper_offsets)} QASPER, {len(squad_offsets)} SQuADv2, "
        f"{len(sciq_offsets)} SciQ, {len(pubmedqa_offsets)} PubMedQA, "
        f"{len(_TRAIN_ARXIV_QUERIES)} arXiv, {len(_TRAIN_OPENALEX_QUERIES)} OpenAlex, "
        f"{len(_TRAIN_EUROPEPMC_QUERIES)} EuropePMC, "
        f"{len(_TRAIN_VI_WIKI_TOPICS)} VI Wiki, {len(_TRAIN_EN_WIKI_TOPICS)} EN Wiki..."
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        qasper_futs = [
            executor.submit(_fetch_hf_rows_retry, "allenai/qasper", "qasper", "train", off, 100, 30)
            for off in qasper_offsets
        ]
        squad_futs = [
            executor.submit(_fetch_hf_rows_retry, "rajpurkar/squad_v2", "squad_v2", "train", off, 100, 25)
            for off in squad_offsets
        ]
        sciq_futs = [
            executor.submit(_fetch_hf_rows_retry, "allenai/sciq", "default", "train", off, 100, 25)
            for off in sciq_offsets
        ]
        pubmedqa_futs = [
            executor.submit(_fetch_hf_rows_retry, "qiaojin/PubMedQA", "pqa_labeled", "train", off, 100, 25)
            for off in pubmedqa_offsets
        ]
        arxiv_futs = [
            executor.submit(fetch_arxiv_papers, q, n, 25)
            for q, n in _TRAIN_ARXIV_QUERIES
        ]
        openalex_futs = [
            executor.submit(fetch_openalex_papers, q, n, 20)
            for q, n in _TRAIN_OPENALEX_QUERIES
        ]
        europepmc_futs = [
            executor.submit(fetch_europepmc_papers, q, n, 20)
            for q, n in _TRAIN_EUROPEPMC_QUERIES
        ]
        vi_futs = [executor.submit(fetch_wikipedia_article, t, "vi", 15) for t in _TRAIN_VI_WIKI_TOPICS]
        en_futs = [executor.submit(fetch_wikipedia_article, t, "en", 15) for t in _TRAIN_EN_WIKI_TOPICS]

        # 1. QASPER train
        qasper_rows: List[Dict[str, Any]] = []
        for fut in concurrent.futures.as_completed(qasper_futs):
            qasper_rows.extend(fut.result())
        qasper_docs, qasper_qs = _parse_qasper_rows(qasper_rows)
        for doc in qasper_docs:
            _add_doc(doc)
        questions.extend(q for q in qasper_qs if q["document_id"] in seen_doc_ids)
        print(f"  -> QASPER train: {len(qasper_rows)} rows -> {sum(1 for d in documents if d['source']=='qasper')} docs")

        # 2. SQuAD v2 train
        squad_rows: List[Dict[str, Any]] = []
        for fut in concurrent.futures.as_completed(squad_futs):
            squad_rows.extend(fut.result())
        squad_docs, squad_qs = _parse_squad_v2_rows(squad_rows)
        for doc in squad_docs:
            _add_doc(doc)
        questions.extend(q for q in squad_qs if q["document_id"] in seen_doc_ids)
        print(f"  -> SQuAD v2 train: {len(squad_rows)} rows -> {sum(1 for d in documents if d['source']=='squad_v2')} docs")

        # 3. SciQ train
        sciq_rows: List[Dict[str, Any]] = []
        for fut in concurrent.futures.as_completed(sciq_futs):
            sciq_rows.extend(fut.result())
        sciq_docs, sciq_qs = _parse_sciq_rows(sciq_rows)
        for doc in sciq_docs:
            _add_doc(doc)
        questions.extend(q for q in sciq_qs if q["document_id"] in seen_doc_ids)
        print(f"  -> SciQ train: {len(sciq_rows)} rows -> {sum(1 for d in documents if d['source']=='sciq')} docs")

        # 4. PubMedQA train
        pubmedqa_rows: List[Dict[str, Any]] = []
        for fut in concurrent.futures.as_completed(pubmedqa_futs):
            pubmedqa_rows.extend(fut.result())
        pubmedqa_docs, pubmedqa_qs = _parse_pubmedqa_rows(pubmedqa_rows)
        for doc in pubmedqa_docs:
            _add_doc(doc)
        questions.extend(q for q in pubmedqa_qs if q["document_id"] in seen_doc_ids)
        print(f"  -> PubMedQA train: {len(pubmedqa_rows)} rows -> {sum(1 for d in documents if d['source']=='pubmedqa')} docs")

        # 5. arXiv
        for fut in concurrent.futures.as_completed(arxiv_futs):
            for paper in fut.result():
                if _add_doc(paper):
                    questions.extend(_build_arxiv_questions(paper))
        print(f"  -> arXiv: {sum(1 for d in documents if d['source']=='arxiv')} docs")

        # 6. OpenAlex
        for fut in concurrent.futures.as_completed(openalex_futs):
            for paper in fut.result():
                if _add_doc(paper):
                    questions.extend(_build_abstract_paper_questions(paper))
        print(f"  -> OpenAlex: {sum(1 for d in documents if d['source']=='openalex')} docs")

        # 7. EuropePMC
        for fut in concurrent.futures.as_completed(europepmc_futs):
            for paper in fut.result():
                if _add_doc(paper):
                    questions.extend(_build_abstract_paper_questions(paper))
        print(f"  -> EuropePMC: {sum(1 for d in documents if d['source']=='europepmc')} docs")

        # 8. Wikipedia VI + EN
        for fut in concurrent.futures.as_completed(vi_futs + en_futs):
            doc = fut.result()
            if _add_doc(doc):
                questions.extend(_build_wiki_questions(doc))
        print(
            f"  -> Wikipedia: {sum(1 for d in documents if d['source']=='wikipedia_vi')} VI, "
            f"{sum(1 for d in documents if d['source']=='wikipedia_en')} EN"
        )

    # Strict post-check for 0% overlap
    overlap_ids = seen_doc_ids & bench_doc_ids
    overlap_titles = seen_titles & bench_titles
    assert not overlap_ids, f"Overlap document_ids found: {overlap_ids}"
    assert not overlap_titles, f"Overlap titles found: {overlap_titles}"

    elapsed_sec = round(time.perf_counter() - t0, 2)
    by_source: Dict[str, int] = {}
    for d in documents:
        by_source[d["source"]] = by_source.get(d["source"], 0) + 1
    answerable_count = sum(1 for q in questions if q.get("answerable"))

    payload: Dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_sec,
            "total_documents": len(documents),
            "total_questions": len(questions),
            "answerable_questions": answerable_count,
            "unanswerable_questions": len(questions) - answerable_count,
            "documents_by_source": by_source,
            "benchmark_overlap_doc_ids": len(overlap_ids),
            "benchmark_overlap_titles": len(overlap_titles),
            "skipped_benchmark_overlaps": skipped_overlap,
            "disjoint_from": str(benchmark_path),
        },
        "documents": documents,
        "questions": questions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"[Done] Saved {len(documents)} documents, {len(questions)} Q&A samples "
        f"({answerable_count} answerable) to {output_path} in {elapsed_sec}s "
        f"(0% benchmark overlap, skipped={skipped_overlap})"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch large online training corpus disjoint from benchmark.")
    parser.add_argument("--output", type=str, default="data/online_training_corpus.json")
    parser.add_argument("--benchmark", type=str, default="data/online_benchmark_corpus.json")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--qasper-rows", type=int, default=200)
    parser.add_argument("--squad-rows", type=int, default=3500)
    parser.add_argument("--sciq-rows", type=int, default=600)
    parser.add_argument("--pubmedqa-rows", type=int, default=400)
    args = parser.parse_args()

    build_online_training_corpus(
        output_path=Path(args.output),
        benchmark_path=Path(args.benchmark),
        max_workers=args.workers,
        qasper_rows_total=args.qasper_rows,
        squad_rows_total=args.squad_rows,
        sciq_rows_total=args.sciq_rows,
        pubmedqa_rows_total=args.pubmedqa_rows,
    )


if __name__ == "__main__":
    main()
