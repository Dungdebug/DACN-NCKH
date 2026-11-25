"""Online benchmark corpus fetcher using free, keyless public APIs.

Supported sources:
1. Wikipedia Action API (vi.wikipedia.org, en.wikipedia.org) -> Markdown ## sections
2. arXiv Atom API (export.arxiv.org/api/query) -> AI/NLP/Crypto/Systems papers
3. OpenAlex API (api.openalex.org/works) -> Academic papers with inverted-index abstracts
4. EuropePMC REST API (ebi.ac.uk/europepmc/webservices/rest) -> Biomedical & AI papers
5. HuggingFace Datasets Server (datasets-server.huggingface.co/rows) ->
   allenai/qasper, rajpurkar/squad_v2, allenai/sciq, qiaojin/PubMedQA
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

_HEADERS = {
    "User-Agent": "WaveCMS-NCKH-Benchmark/1.0 (Academic Research; +https://github.com/nckh)"
}

_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "what", "which", "who", "when", "where",
    "why", "how", "into", "about", "than", "then", "also", "been", "being",
    "their", "there", "these", "those", "such", "can", "could", "will", "would",
    "của", "và", "là", "trong", "cho", "với", "các", "những", "một", "được",
    "khi", "nào", "như", "thế", "này", "đó", "tại", "từ", "đến", "về", "hay",
    "hoặc", "không", "có", "thể", "trên", "dưới", "theo", "nhiều", "người",
}

_VI_WIKI_TOPICS = [
    "Trí tuệ nhân tạo", "Học máy", "Học sâu", "Xử lý ngôn ngữ tự nhiên",
    "Mạng nơ-ron nhân tạo", "Mô hình ngôn ngữ lớn", "Mật mã học",
    "Mật mã khóa công khai", "Hàm băm mật mã", "Chữ ký số", "Chuỗi khối",
    "Hệ phân tán", "Cơ sở dữ liệu", "Thuật toán tìm kiếm", "Cấu trúc dữ liệu",
    "Hệ điều hành", "Mạng máy tính", "An ninh mạng", "Thị giác máy tính",
    "Học tăng cường", "Khoa học dữ liệu", "Khai phá dữ liệu", "Lý thuyết thông tin",
    "Đại số tuyến tính", "Xác suất thống kê", "Lý thuyết đồ thị", "Tối ưu hóa",
    "Việt Nam", "Hà Nội", "Thành phố Hồ Chí Minh", "Lịch sử Việt Nam",
    "Văn học Việt Nam", "Truyện Kiều", "Nguyễn Du", "Hồ Chí Minh",
    "Vật lý lượng tử", "Thuyết tương đối", "Sinh học phân tử", "Di truyền học",
    "Biến đổi khí hậu", "Năng lượng tái tạo", "Kinh tế học vĩ mô",
    "Tài chính", "Thương mại điện tử", "Internet", "Điện toán đám mây",
    "Kỹ thuật phần mềm", "Kiểm thử phần mềm", "Trình biên dịch",
    "Ngôn ngữ lập trình Python", "Toán học", "Hóa học hữu cơ", "Thiên văn học",
    "Hệ Mặt Trời", "Sinh thái học", "Giáo dục", "Y học", "Miễn dịch học",
    "Công nghệ sinh học", "Robot học",
]

_EN_WIKI_TOPICS = [
    "Artificial intelligence", "Machine learning", "Deep learning",
    "Natural language processing", "Large language model", "Transformer (deep learning architecture)",
    "Attention (machine learning)", "Information retrieval", "Okapi BM25",
    "Retrieval-augmented generation", "Cryptography", "Public-key cryptography",
    "Zero-knowledge proof", "Homomorphic encryption", "Hash function",
    "Distributed computing", "Consensus (computer science)", "Database index",
    "Operating system", "Computer network", "Compiler", "Software engineering",
    "Reinforcement learning", "Computer vision", "Knowledge graph",
    "Question answering", "Vector database", "Graph theory",
    "Information theory", "Bayesian inference", "Statistical learning theory",
    "Quantum computing", "Blockchain", "Differential privacy",
    "Federated learning", "Neural Turing machine", "Memory-augmented neural network",
    "Continual learning", "Catastrophic interference", "Text segmentation",
    "Precision and recall", "F-score", "Statistical significance",
    "Monte Carlo method", "Gradient descent", "Backpropagation",
    "Support vector machine", "Random forest", "Word embedding",
    "BERT (language model)",
]

_ARXIV_QUERIES = [
    ("cat:cs.CL", 10),
    ("cat:cs.AI", 10),
    ("cat:cs.CR", 10),
    ("cat:cs.DC", 10),
    ("cat:cs.IR", 10),
    ("cat:cs.LG", 10),
]


def _extract_keywords(text: str, max_kw: int = 6) -> List[str]:
    tokens = re.findall(r"[^\W\d_]{3,}", text.lower(), flags=re.UNICODE)
    seen: set = set()
    out: List[str] = []
    for tok in tokens:
        if tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_kw:
            break
    return out


def _wiki_extract_to_markdown(title: str, extract: str) -> Tuple[str, List[Dict[str, str]]]:
    """Convert Wikipedia plaintext extract headings (== H ==) into Markdown ## headings."""
    lines = extract.splitlines()
    md_lines: List[str] = [f"# {title}", ""]
    sections: List[Dict[str, str]] = []
    cur_heading = "Overview"
    cur_buf: List[str] = []

    heading_re = re.compile(r"^(={2,5})\s*(.+?)\s*\1\s*$")
    for raw_line in lines:
        line = raw_line.strip()
        m = heading_re.match(line)
        if m:
            body = "\n".join(cur_buf).strip()
            if body:
                sections.append({"heading": cur_heading, "text": body})
            level = min(len(m.group(1)), 4)
            cur_heading = m.group(2).strip()
            md_lines.append("")
            md_lines.append(f"{'#' * level} {cur_heading}")
            cur_buf = []
        else:
            if line:
                cur_buf.append(line)
                md_lines.append(line)
            elif md_lines and md_lines[-1] != "":
                md_lines.append("")

    body = "\n".join(cur_buf).strip()
    if body:
        sections.append({"heading": cur_heading, "text": body})

    md_text = re.sub(r"\n{3,}", "\n\n", "\n".join(md_lines)).strip()
    return md_text, sections


def fetch_wikipedia_article(
    title_or_query: str,
    lang: str = "vi",
    timeout: int = 15,
) -> Optional[Dict[str, Any]]:
    """Fetch a Wikipedia article via Action API and format headings as Markdown."""
    lang = (lang or "vi").strip().lower()
    if lang not in {"vi", "en"}:
        lang = "en"
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    query = (title_or_query or "").strip()
    if not query:
        return None

    def _query_by_title(title: str) -> Optional[Dict[str, Any]]:
        params = {
            "action": "query",
            "prop": "extracts|info",
            "explaintext": 1,
            "inprop": "url",
            "redirects": 1,
            "titles": title,
            "format": "json",
            "formatversion": 2,
        }
        resp = requests.get(api_url, params=params, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return None
        return pages[0]

    try:
        page = _query_by_title(query)
        if page is None or len(page.get("extract", "")) < 200:
            search_resp = requests.get(
                api_url,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 3,
                    "format": "json",
                },
                headers=_HEADERS,
                timeout=timeout,
            )
            search_resp.raise_for_status()
            hits = search_resp.json().get("query", {}).get("search", [])
            for hit in hits:
                cand = _query_by_title(hit["title"])
                if cand and len(cand.get("extract", "")) >= 200:
                    page = cand
                    break

        if not page or not page.get("extract"):
            return None

        resolved_title = page.get("title", query)
        extract = page.get("extract", "").strip()
        if len(extract) < 150:
            return None

        md_text, sections = _wiki_extract_to_markdown(resolved_title, extract)
        page_id = page.get("pageid") or hashlib.md5(resolved_title.encode("utf-8")).hexdigest()[:8]
        url = page.get("fullurl") or (
            f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(resolved_title.replace(' ', '_'))}"
        )
        return {
            "document_id": f"WIKI_{lang.upper()}_{page_id}",
            "title": resolved_title,
            "text": md_text,
            "content": md_text,
            "source": f"wikipedia_{lang}",
            "language": lang,
            "url": url,
            "categories": [f"Wikipedia ({lang.upper()})"],
            "sections": sections,
            "word_count": len(md_text.split()),
        }
    except Exception:
        return None


def fetch_arxiv_papers(
    query: str,
    max_results: int = 5,
    timeout: int = 20,
) -> List[Dict[str, Any]]:
    """Fetch papers from arXiv Atom API with title, abstract, and categories."""
    q = (query or "").strip()
    if not q:
        return []
    search_query = q if ":" in q else f"all:{q}"
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max(1, min(int(max_results), 50)),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers: List[Dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        raw_id = (entry.findtext("atom:id", "", ns) or "").strip()
        arxiv_id = raw_id.rsplit("/", 1)[-1] if raw_id else hashlib.md5(raw_id.encode()).hexdigest()[:10]
        title = re.sub(r"\s+", " ", (entry.findtext("atom:title", "", ns) or "")).strip()
        abstract = re.sub(r"\s+", " ", (entry.findtext("atom:summary", "", ns) or "")).strip()
        published = (entry.findtext("atom:published", "", ns) or "").strip()
        authors = [
            (a.findtext("atom:name", "", ns) or "").strip()
            for a in entry.findall("atom:author", ns)
            if (a.findtext("atom:name", "", ns) or "").strip()
        ]
        categories = [
            c.attrib["term"]
            for c in entry.findall("atom:category", ns)
            if c.attrib.get("term")
        ]
        if not title or not abstract:
            continue

        md_text = (
            f"# {title}\n\n"
            f"## Abstract\n{abstract}\n\n"
            f"## Metadata\n"
            f"arXiv ID: {arxiv_id}\n"
            f"Categories: {', '.join(categories) if categories else 'cs'}\n"
            f"Authors: {', '.join(authors[:8]) if authors else 'N/A'}\n"
            f"Published: {published}"
        )
        papers.append(
            {
                "document_id": f"ARXIV_{re.sub(r'[^A-Za-z0-9_.-]', '_', arxiv_id)}",
                "title": title,
                "abstract": abstract,
                "text": md_text,
                "content": md_text,
                "categories": categories,
                "authors": authors,
                "published": published,
                "url": raw_id or f"https://arxiv.org/abs/{arxiv_id}",
                "source": "arxiv",
                "language": "en",
                "word_count": len(md_text.split()),
            }
        )
    return papers


def _fetch_hf_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    timeout: int = 20,
) -> List[Dict[str, Any]]:
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return [item.get("row", {}) for item in resp.json().get("rows", [])]
    except Exception:
        return []


def _build_wiki_questions(doc: Dict[str, Any], max_q: int = 3) -> List[Dict[str, Any]]:
    """Derive grounded answerable & unanswerable questions from a Wikipedia doc."""
    doc_id = doc["document_id"]
    title = doc["title"]
    lang = doc["language"]
    source = doc["source"]
    sections = doc.get("sections") or []
    questions: List[Dict[str, Any]] = []

    for idx, sec in enumerate(sections):
        if len(questions) >= max_q - 1:
            break
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", sec["text"]) if len(s.strip()) >= 35]
        if not sents:
            continue
        ans = sents[0][:320]
        heading = sec["heading"]
        if lang == "vi":
            q_text = (
                f"Theo tài liệu về {title}, nội dung chính trong mục '{heading}' là gì?"
                if heading != "Overview"
                else f"Tài liệu định nghĩa hoặc mô tả {title} như thế nào?"
            )
        else:
            q_text = (
                f"According to the article on {title}, what is described in the '{heading}' section?"
                if heading != "Overview"
                else f"How does the article define or introduce {title}?"
            )
        questions.append(
            {
                "question_id": f"Q_{doc_id}_{idx}",
                "document_id": doc_id,
                "question": q_text,
                "gold_answer": ans,
                "answerable": True,
                "language": lang,
                "source": source,
                "expected_keywords": _extract_keywords(ans),
            }
        )

    # Add 1 unanswerable question per doc for abstention evaluation
    if lang == "vi":
        unans_q = f"Tài liệu về {title} có công bố mã số ngân sách tài trợ cụ thể năm 2099 và chữ ký kiểm toán nội bộ không?"
    else:
        unans_q = f"Does the article on {title} specify the exact 2099 budget allocation code and internal audit serial number?"
    questions.append(
        {
            "question_id": f"Q_{doc_id}_UNANS",
            "document_id": doc_id,
            "question": unans_q,
            "gold_answer": "",
            "answerable": False,
            "language": lang,
            "source": source,
            "expected_keywords": [],
        }
    )
    return questions


def _build_abstract_paper_questions(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    doc_id = paper["document_id"]
    title = paper["title"]
    abstract = paper.get("abstract", "")
    source = paper.get("source", "arxiv")
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if len(s.strip()) >= 30]
    ans = sents[0][:320] if sents else abstract[:320]
    return [
        {
            "question_id": f"Q_{doc_id}_0",
            "document_id": doc_id,
            "question": f"What problem or method is proposed in the paper '{title}'?",
            "gold_answer": ans,
            "answerable": True,
            "language": "en",
            "source": source,
            "expected_keywords": _extract_keywords(ans),
        },
        {
            "question_id": f"Q_{doc_id}_UNANS",
            "document_id": doc_id,
            "question": f"What exact GPU hourly rental price in VND is reported in '{title}'?",
            "gold_answer": "",
            "answerable": False,
            "language": "en",
            "source": source,
            "expected_keywords": [],
        },
    ]


def _build_arxiv_questions(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _build_abstract_paper_questions(paper)


def _unpack_openalex_abstract(inv_index: Optional[Dict[str, List[int]]]) -> str:
    if not isinstance(inv_index, dict) or not inv_index:
        return ""
    positioned: List[Tuple[int, str]] = []
    for word, positions in inv_index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                positioned.append((pos, str(word)))
    if not positioned:
        return ""
    positioned.sort(key=lambda x: x[0])
    return re.sub(r"\s+", " ", " ".join(w for _, w in positioned)).strip()


def fetch_openalex_papers(
    query: str,
    max_results: int = 10,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
    """Fetch academic papers from OpenAlex API and unpack inverted-index abstracts."""
    q = (query or "").strip()
    if not q:
        return []
    url = "https://api.openalex.org/works"
    params = {
        "search": q,
        "per-page": max(1, min(int(max_results), 50)),
    }
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        results = resp.json().get("results", []) or []
    except Exception:
        return []

    papers: List[Dict[str, Any]] = []
    for item in results:
        title = re.sub(r"\s+", " ", str(item.get("title") or item.get("display_name") or "")).strip()
        abstract = _unpack_openalex_abstract(item.get("abstract_inverted_index"))
        if not title or len(abstract) < 40:
            continue

        raw_id = str(item.get("id") or "").strip()
        work_id = raw_id.rsplit("/", 1)[-1] if raw_id else hashlib.md5(title.encode("utf-8")).hexdigest()[:10]
        year = item.get("publication_year") or ""
        primary_loc = item.get("primary_location") or {}
        src_obj = primary_loc.get("source") or {}
        venue = str(src_obj.get("display_name") or "OpenAlex").strip()
        concepts = [
            str(c.get("display_name")).strip()
            for c in (item.get("concepts") or [])[:5]
            if isinstance(c, dict) and c.get("display_name")
        ]
        doi = str(item.get("doi") or "").strip()
        doc_url = doi or raw_id or f"https://openalex.org/{work_id}"
        authors = [
            str((a.get("author") or {}).get("display_name") or "").strip()
            for a in (item.get("authorships") or [])[:8]
            if isinstance(a, dict) and (a.get("author") or {}).get("display_name")
        ]

        md_text = (
            f"# {title}\n\n"
            f"## Abstract\n{abstract}\n\n"
            f"## Metadata\n"
            f"OpenAlex ID: {work_id}\n"
            f"Year: {year or 'N/A'}\n"
            f"Venue: {venue}\n"
            f"Concepts: {', '.join(concepts) if concepts else 'Research'}\n"
            f"Authors: {', '.join(authors) if authors else 'N/A'}\n"
            f"URL: {doc_url}"
        )
        papers.append(
            {
                "document_id": f"OPENALEX_{re.sub(r'[^A-Za-z0-9_.-]', '_', work_id)}",
                "title": title,
                "abstract": abstract,
                "text": md_text,
                "content": md_text,
                "categories": concepts or [venue],
                "authors": authors,
                "published": str(year),
                "url": doc_url,
                "source": "openalex",
                "language": "en",
                "word_count": len(md_text.split()),
            }
        )
    return papers


def fetch_europepmc_papers(
    query: str,
    max_results: int = 10,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
    """Fetch biomedical/scientific papers from EuropePMC REST API."""
    q = (query or "").strip()
    if not q:
        return []
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": q,
        "format": "json",
        "resultType": "core",
        "pageSize": max(1, min(int(max_results), 50)),
    }
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        results = (resp.json().get("resultList") or {}).get("result", []) or []
    except Exception:
        return []

    papers: List[Dict[str, Any]] = []
    for item in results:
        title = re.sub(r"<[^>]+>", " ", str(item.get("title") or ""))
        title = re.sub(r"\s+", " ", title).strip()
        raw_abs = re.sub(r"<[^>]+>", " ", str(item.get("abstractText") or ""))
        abstract = re.sub(r"\s+", " ", raw_abs).strip()
        if not title or len(abstract) < 40:
            continue

        pmcid = str(item.get("pmcid") or "").strip()
        pmid = str(item.get("pmid") or "").strip()
        raw_id = str(item.get("id") or "").strip()
        rec_id = pmcid or pmid or raw_id or hashlib.md5(title.encode("utf-8")).hexdigest()[:10]
        year = str(item.get("pubYear") or "").strip()
        journal_info = item.get("journalInfo") or {}
        journal = str(
            item.get("journalTitle")
            or (journal_info.get("journal") or {}).get("title")
            or "EuropePMC"
        ).strip()
        author_str = str(item.get("authorString") or "").strip()
        doc_url = (
            f"https://europepmc.org/article/PMC/{pmcid}"
            if pmcid
            else f"https://europepmc.org/article/MED/{pmid or raw_id}"
        )

        md_text = (
            f"# {title}\n\n"
            f"## Abstract\n{abstract}\n\n"
            f"## Metadata\n"
            f"Record ID: {rec_id}\n"
            f"Journal: {journal}\n"
            f"Year: {year or 'N/A'}\n"
            f"Authors: {author_str or 'N/A'}\n"
            f"URL: {doc_url}"
        )
        papers.append(
            {
                "document_id": f"EUROPEPMC_{re.sub(r'[^A-Za-z0-9_.-]', '_', rec_id)}",
                "title": title,
                "abstract": abstract,
                "text": md_text,
                "content": md_text,
                "categories": [journal],
                "authors": [a.strip() for a in author_str.split(",") if a.strip()][:8],
                "published": year,
                "url": doc_url,
                "source": "europepmc",
                "language": "en",
                "word_count": len(md_text.split()),
            }
        )
    return papers


def _parse_qasper_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    docs: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []
    for row in rows:
        paper_id = str(row.get("id") or "").strip()
        title = (row.get("title") or "").strip()
        abstract = (row.get("abstract") or "").strip()
        full_text = row.get("full_text") or {}
        sec_names = full_text.get("section_name") or []
        sec_paras = full_text.get("paragraphs") or []
        if not paper_id or not title:
            continue

        md_parts = [f"# {title}"]
        if abstract:
            md_parts.append(f"\n## Abstract\n{abstract}")
        for sname, paras in zip(sec_names, sec_paras):
            body = "\n\n".join(p.strip() for p in (paras or []) if p and p.strip())
            if body:
                heading = (sname or "Section").strip()
                md_parts.append(f"\n## {heading}\n{body}")
        md_text = "\n".join(md_parts).strip()
        doc_id = f"QASPER_{paper_id}"
        docs.append(
            {
                "document_id": doc_id,
                "title": title,
                "text": md_text,
                "content": md_text,
                "source": "qasper",
                "language": "en",
                "categories": ["QASPER"],
                "word_count": len(md_text.split()),
            }
        )

        qas = row.get("qas") or {}
        q_list = qas.get("question") or []
        qid_list = qas.get("question_id") or []
        ans_groups = qas.get("answers") or []
        for idx, (q_text, qid, ans_group) in enumerate(zip(q_list, qid_list, ans_groups)):
            ans_items = (ans_group or {}).get("answer") or []
            if not ans_items or not q_text:
                continue
            first = ans_items[0]
            is_unans = bool(first.get("unanswerable", False))
            gold = ""
            if not is_unans:
                if first.get("free_form_answer"):
                    gold = str(first["free_form_answer"]).strip()
                elif first.get("extractive_spans"):
                    gold = "; ".join(str(s) for s in first["extractive_spans"] if s).strip()
                elif first.get("yes_no") is not None:
                    gold = "Yes" if first["yes_no"] else "No"
            if not is_unans and not gold:
                continue
            questions.append(
                {
                    "question_id": f"QASPER_{qid or f'{paper_id}_{idx}'}",
                    "document_id": doc_id,
                    "question": q_text.strip(),
                    "gold_answer": gold,
                    "answerable": not is_unans,
                    "language": "en",
                    "source": "qasper",
                    "expected_keywords": _extract_keywords(gold) if not is_unans else [],
                }
            )
    return docs, questions


def _parse_squad_v2_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        title = (row.get("title") or "").strip()
        context = (row.get("context") or "").strip()
        if not title or not context:
            continue
        grouped.setdefault((title, context), []).append(row)

    docs: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []
    for (title, context), items in grouped.items():
        ctx_hash = hashlib.md5(f"{title}::{context}".encode("utf-8")).hexdigest()[:10]
        doc_id = f"SQUAD2_{ctx_hash}"
        md_text = f"# {title}\n\n## Passage\n{context}"
        docs.append(
            {
                "document_id": doc_id,
                "title": title,
                "text": md_text,
                "content": md_text,
                "source": "squad_v2",
                "language": "en",
                "categories": ["SQuAD v2"],
                "word_count": len(md_text.split()),
            }
        )
        for item in items:
            ans_texts = (item.get("answers") or {}).get("text") or []
            answerable = len(ans_texts) > 0
            gold = str(ans_texts[0]).strip() if answerable else ""
            questions.append(
                {
                    "question_id": f"SQUAD2_{item.get('id', ctx_hash)}",
                    "document_id": doc_id,
                    "question": (item.get("question") or "").strip(),
                    "gold_answer": gold,
                    "answerable": answerable,
                    "language": "en",
                    "source": "squad_v2",
                    "expected_keywords": _extract_keywords(gold) if answerable else [],
                }
            )
    return docs, questions


def _parse_sciq_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    docs: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for idx, row in enumerate(rows):
        support = str(row.get("support") or "").strip()
        question = str(row.get("question") or "").strip()
        correct = str(row.get("correct_answer") or "").strip()
        if len(support) < 40 or not question or not correct:
            continue
        ctx_hash = hashlib.md5(support.encode("utf-8")).hexdigest()[:10]
        doc_id = f"SCIQ_{ctx_hash}"
        title = f"SciQ Science Passage ({ctx_hash})"
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            md_text = f"# {title}\n\n## Support Passage\n{support}"
            docs.append(
                {
                    "document_id": doc_id,
                    "title": title,
                    "text": md_text,
                    "content": md_text,
                    "source": "sciq",
                    "language": "en",
                    "categories": ["SciQ"],
                    "word_count": len(md_text.split()),
                }
            )
        questions.append(
            {
                "question_id": f"SCIQ_{ctx_hash}_{idx}",
                "document_id": doc_id,
                "question": question,
                "gold_answer": correct,
                "answerable": True,
                "language": "en",
                "source": "sciq",
                "expected_keywords": _extract_keywords(correct),
            }
        )
    return docs, questions


def _parse_pubmedqa_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    docs: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for idx, row in enumerate(rows):
        pubid = str(row.get("pubid") or idx).strip()
        question = str(row.get("question") or "").strip()
        ctx_obj = row.get("context") or {}
        contexts = ctx_obj.get("contexts") if isinstance(ctx_obj, dict) else []
        ctx_text = "\n\n".join(str(c).strip() for c in (contexts or []) if str(c).strip())
        long_answer = str(row.get("long_answer") or "").strip()
        final_decision = str(row.get("final_decision") or "").strip()
        if len(ctx_text) < 40 or not question:
            continue
        doc_id = f"PUBMEDQA_{pubid}"
        title = f"PubMedQA Study {pubid}: {question[:80]}"
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            md_parts = [f"# {title}", f"\n## Context\n{ctx_text}"]
            if long_answer:
                md_parts.append(f"\n## Conclusion\n{long_answer}")
            md_text = "\n".join(md_parts).strip()
            docs.append(
                {
                    "document_id": doc_id,
                    "title": title,
                    "text": md_text,
                    "content": md_text,
                    "source": "pubmedqa",
                    "language": "en",
                    "categories": ["PubMedQA"],
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pubid}/",
                    "word_count": len(md_text.split()),
                }
            )
        gold = long_answer or final_decision
        if gold:
            questions.append(
                {
                    "question_id": f"PUBMEDQA_{pubid}_{idx}",
                    "document_id": doc_id,
                    "question": question,
                    "gold_answer": gold,
                    "answerable": True,
                    "language": "en",
                    "source": "pubmedqa",
                    "expected_keywords": _extract_keywords(gold),
                }
            )
    return docs, questions


def fetch_hf_dataset_sample(
    dataset_name: str,
    split: str = "train",
    offset: int = 0,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Fetch and parse a sample of documents from HuggingFace (qasper, squad_v2, sciq, pubmedqa)."""
    key = (dataset_name or "").strip().lower()
    safe_limit = max(1, min(int(limit), 50))
    safe_offset = max(0, int(offset))
    safe_split = (split or "train").strip() or "train"

    if key in ("qasper", "allenai/qasper"):
        rows = _fetch_hf_rows("allenai/qasper", "qasper", safe_split, safe_offset, safe_limit)
        docs, _ = _parse_qasper_rows(rows)
        return docs[:safe_limit]

    if key in ("squad_v2", "squadv2", "rajpurkar/squad_v2"):
        # SQuAD v2 rows group by (title, context), so fetch extra rows to yield enough distinct docs
        fetch_len = min(100, safe_limit * 4)
        rows = _fetch_hf_rows("rajpurkar/squad_v2", "squad_v2", safe_split, safe_offset, fetch_len)
        docs, _ = _parse_squad_v2_rows(rows)
        return docs[:safe_limit]

    if key in ("sciq", "allenai/sciq"):
        fetch_len = min(100, safe_limit * 2)
        rows = _fetch_hf_rows("allenai/sciq", "default", safe_split, safe_offset, fetch_len)
        docs, _ = _parse_sciq_rows(rows)
        return docs[:safe_limit]

    if key in ("pubmedqa", "qiaojin/pubmedqa"):
        rows = _fetch_hf_rows("qiaojin/PubMedQA", "pqa_labeled", safe_split, safe_offset, safe_limit)
        docs, _ = _parse_pubmedqa_rows(rows)
        return docs[:safe_limit]

    return []


def search_external_document_apis(
    query: str,
    language: str = "vi",
    max_docs: int = 3,
) -> List[Dict[str, Any]]:
    """Search Wikipedia, arXiv, OpenAlex, and EuropePMC in parallel for live chatbot grounding."""
    q = (query or "").strip()
    if not q:
        return []
    lang = (language or "vi").strip().lower()
    wiki_lang = lang if lang in {"vi", "en"} else "en"
    limit = max(1, min(int(max_docs), 10))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        wiki_fut = executor.submit(fetch_wikipedia_article, q, wiki_lang, 10)
        arxiv_fut = executor.submit(fetch_arxiv_papers, q, 2, 12)
        openalex_fut = executor.submit(fetch_openalex_papers, q, 2, 12)
        epmc_fut = executor.submit(fetch_europepmc_papers, q, 2, 12)

        candidates: List[Dict[str, Any]] = []
        try:
            wiki_doc = wiki_fut.result()
            if wiki_doc:
                candidates.append(wiki_doc)
        except Exception:
            pass
        for fut in (openalex_fut, arxiv_fut, epmc_fut):
            try:
                candidates.extend(fut.result() or [])
            except Exception:
                pass

    deduped: List[Dict[str, Any]] = []
    seen_ids: set = set()
    seen_titles: set = set()
    for doc in candidates:
        did = str(doc.get("document_id") or "").strip()
        title_norm = str(doc.get("title") or "").strip().lower()
        if not did or did in seen_ids or (title_norm and title_norm in seen_titles):
            continue
        seen_ids.add(did)
        if title_norm:
            seen_titles.add(title_norm)
        deduped.append(doc)
        if len(deduped) >= limit:
            break
    return deduped


def build_large_online_corpus(
    target_docs: int = 220,
    cache_path: str = "data/online_benchmark_corpus.json",
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Fetch >= target_docs documents & >= 550 QA pairs concurrently and cache to JSON."""
    cache_file = Path(cache_path)
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if len(cached.get("documents", [])) >= target_docs:
                return cached
        except Exception:
            pass

    t0 = time.perf_counter()
    documents: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []
    seen_doc_ids: set = set()

    def _add_doc(doc: Optional[Dict[str, Any]]) -> bool:
        if not doc or doc["document_id"] in seen_doc_ids:
            return False
        seen_doc_ids.add(doc["document_id"])
        documents.append(doc)
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        vi_futures = [executor.submit(fetch_wikipedia_article, t, "vi", 15) for t in _VI_WIKI_TOPICS]
        en_futures = [executor.submit(fetch_wikipedia_article, t, "en", 15) for t in _EN_WIKI_TOPICS]
        arxiv_futures = [executor.submit(fetch_arxiv_papers, q, n, 20) for q, n in _ARXIV_QUERIES]
        qasper_future = executor.submit(_fetch_hf_rows, "allenai/qasper", "qasper", "validation", 0, 50, 25)
        squad_futures = [
            executor.submit(_fetch_hf_rows, "rajpurkar/squad_v2", "squad_v2", "validation", off, 100, 20)
            for off in (0, 100, 200, 300)
        ]

        for fut in concurrent.futures.as_completed(vi_futures + en_futures):
            doc = fut.result()
            if _add_doc(doc):
                questions.extend(_build_wiki_questions(doc))

        for fut in concurrent.futures.as_completed(arxiv_futures):
            for paper in fut.result():
                if _add_doc(paper):
                    questions.extend(_build_arxiv_questions(paper))

        qasper_docs, qasper_qs = _parse_qasper_rows(qasper_future.result())
        for doc in qasper_docs:
            _add_doc(doc)
        questions.extend(q for q in qasper_qs if q["document_id"] in seen_doc_ids)

        squad_rows: List[Dict[str, Any]] = []
        for fut in concurrent.futures.as_completed(squad_futures):
            squad_rows.extend(fut.result())
        squad_docs, squad_qs = _parse_squad_v2_rows(squad_rows)
        for doc in squad_docs:
            _add_doc(doc)
        questions.extend(q for q in squad_qs if q["document_id"] in seen_doc_ids)

    elapsed_sec = round(time.perf_counter() - t0, 2)
    by_source: Dict[str, int] = {}
    for d in documents:
        by_source[d["source"]] = by_source.get(d["source"], 0) + 1
    answerable_count = sum(1 for q in questions if q.get("answerable"))
    unanswerable_count = len(questions) - answerable_count

    payload: Dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_sec,
            "target_docs": target_docs,
            "total_documents": len(documents),
            "total_questions": len(questions),
            "answerable_questions": answerable_count,
            "unanswerable_questions": unanswerable_count,
            "documents_by_source": by_source,
        },
        "documents": documents,
        "questions": questions,
    }

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload
