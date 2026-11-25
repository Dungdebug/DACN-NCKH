"""
Practice, Similar Formula & Exercise Generator (Dual Mode: Offline Symbolic + Online LLM).

- Offline mode (Hướng 1): 100% local, deterministic/randomized Python symbolic & template
  generators grounded in formulas and examples extracted from the active document.
- Online mode (Hướng 2): Uses an external/local LLM (Gemini / OpenAI-compatible / Ollama)
  seeded with the document's extracted formulas and examples, with automatic fallback
  to Offline mode if no API key/endpoint is configured or reachable.
"""

from __future__ import annotations

import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.web.document_analyzer import strip_accents


PRACTICE_INTENT_MARKERS = (
    "sinh them vi du",
    "cho them vi du",
    "them vi du",
    "vi du tuong tu",
    "vi du giong",
    "vi du khac",
    "cong thuc tuong tu",
    "cong thuc giong",
    "bien doi cong thuc",
    "sinh bai tap",
    "tao bai tap",
    "them bai tap",
    "cho bai tap",
    "bai tap tuong tu",
    "bai tap van dung",
    "bai tap luyen tap",
    "bai tap ve",
    "ra de",
    "generate example",
    "similar example",
    "generate exercise",
    "practice problem",
)


def is_practice_generation_query(query: str) -> bool:
    """Returns True if the user is asking to generate similar examples, formulas, or exercises."""
    q_norm = strip_accents((query or "").lower())
    if any(m in q_norm for m in PRACTICE_INTENT_MARKERS):
        return True
    has_gen_verb = any(v in q_norm for v in ("sinh", "tao", "them", "cho them", "ra de", "viet them"))
    has_target = any(t in q_norm for t in ("vi du", "bai tap", "luyen tap", "de toan", "de hoa", "de sinh"))
    return has_gen_verb and has_target


def is_online_llm_enabled() -> bool:
    """Returns True when online LLM generation is enabled via environment configuration."""
    online_flag = os.environ.get("SA_CMS_ONLINE_LLM", "").strip().lower()
    if online_flag in ("1", "true", "yes", "on"):
        return True
    if os.environ.get("SA_CMS_OFFLINE_MODE", "").strip() == "1":
        return False
    return bool(
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OLLAMA_MODEL")
    )


def extract_formulas_and_examples(text: str, max_items: int = 8) -> Tuple[List[str], List[str]]:
    """Extracts formula-like lines and example sentences from document text."""
    formulas: List[str] = []
    examples: List[str] = []
    raw_chunks = re.split(r"(?:\r?\n)+|(?<=[.!?])\s+(?=[A-ZÀ-Ỹ0-9])", text or "")
    for chunk in raw_chunks:
        c = chunk.strip()
        if len(c) < 6:
            continue
        c_norm = strip_accents(c.lower())
        is_ex = (
            c_norm.startswith(("vi du", "bai toan", "de bai", "chang han"))
            or any(k in c_norm for k in ("vi du ", "de bai:", "buoc 1", "ap dung:"))
        )
        has_formula = (
            ("=" in c or "→" in c or "⇌" in c or "⇒" in c)
            and len(c) <= 260
        )
        if is_ex and len(examples) < max_items and c not in examples:
            examples.append(c)
        elif has_formula and len(formulas) < max_items and c not in formulas:
            formulas.append(c)
    return formulas, examples


# ---------------------------------------------------------------------------
# Hướng 1: Offline Symbolic & Template Generators
# ---------------------------------------------------------------------------

def _fmt_num(val: float) -> str:
    """Formats floats cleanly without trailing zeros (Vietnamese comma optional -> dot/clean)."""
    rounded = round(val, 3)
    if abs(rounded - round(rounded)) < 1e-6:
        return str(int(round(rounded)))
    return f"{rounded:g}"


def _generate_redox_offline(seed: Optional[int] = None) -> str:
    """Generates a similar redox formula set, worked example, and 2 exercises with exact Python math."""
    rng = random.Random(seed)
    metals = [
        {"name": "Magie", "sym": "Mg", "M": 24, "n_e": 2, "salt": "Mg(NO₃)₂", "M_salt": 148},
        {"name": "Kẽm", "sym": "Zn", "M": 65, "n_e": 2, "salt": "Zn(NO₃)₂", "M_salt": 189},
        {"name": "Nhôm", "sym": "Al", "M": 27, "n_e": 3, "salt": "Al(NO₃)₃", "M_salt": 213},
        {"name": "Sắt", "sym": "Fe", "M": 56, "n_e": 3, "salt": "Fe(NO₃)₃", "M_salt": 242},
    ]
    gases = [
        {"sym": "NO", "name": "Nitơ monoxit", "n_e": 3, "half": "N⁺⁵ + 3e → N⁺² (trong NO)"},
        {"sym": "NO₂", "name": "Nitơ đioxit", "n_e": 1, "half": "N⁺⁵ + 1e → N⁺⁴ (trong NO₂)"},
        {"sym": "N₂O", "name": "Đinitơ oxit", "n_e": 8, "half": "2N⁺⁵ + 8e → 2N⁺¹ (trong N₂O)"},
    ]

    ex_m = rng.choice(metals)
    ex_g = rng.choice(gases)
    n_metal = rng.choice([0.1, 0.15, 0.2, 0.3, 0.4])
    m_metal = round(n_metal * ex_m["M"], 2)
    n_e = round(n_metal * ex_m["n_e"], 4)
    n_gas = round(n_e / ex_g["n_e"], 4)
    v_gas = round(n_gas * 24.79, 3)
    m_salt = round(n_metal * ex_m["M_salt"], 2)

    # Exercise 1: different metal/gas
    q1_m = rng.choice([m for m in metals if m["sym"] != ex_m["sym"]])
    q1_g = rng.choice(gases)
    q1_n = rng.choice([0.1, 0.2, 0.25, 0.3])
    q1_mass = round(q1_n * q1_m["M"], 2)
    q1_ne = round(q1_n * q1_m["n_e"], 4)
    q1_ngas = round(q1_ne / q1_g["n_e"], 4)
    q1_v = round(q1_ngas * 24.79, 3)
    q1_msalt = round(q1_n * q1_m["M_salt"], 2)

    # Exercise 2: reverse problem (given V -> find m)
    q2_m = rng.choice(metals)
    q2_g = gases[0]  # NO for clean numbers
    q2_n = rng.choice([0.1, 0.2, 0.3])
    q2_ne = round(q2_n * q2_m["n_e"], 4)
    q2_ngas = round(q2_ne / q2_g["n_e"], 4)
    q2_v = round(q2_ngas * 24.79, 3)
    q2_mass = round(q2_n * q2_m["M"], 2)

    return (
        "### 🧪 Công thức suy dẫn & biến đổi tương tự (Chế độ Offline — Symbolic)\n"
        "- **Bảo toàn electron tổng quát (1 kim loại hóa trị $a$ + HNO₃ tạo 1 sản phẩm khử nhận $b$ electron):**\n"
        "  $$a \\cdot n_{\\text{kim loại}} = b \\cdot n_{\\text{sản phẩm khử}}$$\n"
        "- **Công thức tính nhanh khối lượng muối nitrat (không tạo NH₄NO₃):**\n"
        "  $$m_{\\text{muối}} = m_{\\text{kim loại}} + 62 \\cdot \\sum n(e\\text{ trao đổi})$$\n"
        "- **Quy đổi thể tích khí ở ĐKC (25°C, 1 bar) và nồng độ mol:**\n"
        "  $$n_{\\text{khí}} = \\frac{V}{24,79}, \\qquad n = \\frac{m}{M}, \\qquad V_{dd} = \\frac{n}{C_M}$$\n\n"
        "### 📝 Ví dụ tương tự (tự động sinh & tính toán chính xác)\n"
        f"**Đề bài:** Hòa tan hoàn toàn **{_fmt_num(m_metal)} gam** {ex_m['name']} ({ex_m['sym']}, M = {ex_m['M']}) "
        f"vào dung dịch HNO₃ dư, thu được dung dịch muối {ex_m['salt']} và **V lít** khí {ex_g['name']} ({ex_g['sym']} — sản phẩm khử duy nhất ở ĐKC 25°C, 1 bar). Tính V và khối lượng muối thu được.\n"
        f"- **Bước 1:** $n({ex_m['sym']}) = \\frac{{{_fmt_num(m_metal)}}}{{{ex_m['M']}}} = {_fmt_num(n_metal)}\\text{{ mol}}$.\n"
        f"- **Bước 2:** Quá trình oxi hóa: ${ex_m['sym']}^0 \\to {ex_m['sym']}^{{+{ex_m['n_e']}}} + {ex_m['n_e']}e$ "
        f"($n(e\\text{{ nhường}}) = {ex_m['n_e']} \\times {_fmt_num(n_metal)} = {_fmt_num(n_e)}\\text{{ mol}}$).\n"
        f"  Quá trình khử: {ex_g['half']}.\n"
        f"- **Bước 3 (Bảo toàn e):** $n({ex_g['sym']}) = \\frac{{{_fmt_num(n_e)}}}{{{ex_g['n_e']}}} = {_fmt_num(n_gas)}\\text{{ mol}}$.\n"
        f"- **Bước 4:** $V({ex_g['sym']}) = {_fmt_num(n_gas)} \\times 24,79 = \\mathbf{{{_fmt_num(v_gas)}\\text{{ lít}}}}$; "
        f"$m_{{{ex_m['salt']}}} = {_fmt_num(n_metal)} \\times {ex_m['M_salt']} = \\mathbf{{{_fmt_num(m_salt)}\\text{{ gam}}}}$.\n\n"
        "### ✍️ Bài tập tự luyện\n"
        f"1. **Bài 1 (Thuận):** Hòa tan hoàn toàn **{_fmt_num(q1_mass)} gam** {q1_m['name']} ({q1_m['sym']}, M = {q1_m['M']}) vào dung dịch HNO₃ dư, thu được V lít khí {q1_g['sym']} (sản phẩm khử duy nhất ở ĐKC 25°C, 1 bar). Tính V và khối lượng muối {q1_m['salt']}?\n"
        f"   - *Đáp án kiểm tra:* $n({q1_m['sym']}) = {_fmt_num(q1_n)}$ mol; $n(e) = {_fmt_num(q1_ne)}$ mol; **$V = {_fmt_num(q1_v)}$ lít**; **$m_{{\\text{{muối}}}} = {_fmt_num(q1_msalt)}$ gam**.\n"
        f"2. **Bài 2 (Nghịch):** Hòa tan hết $m$ gam {q2_m['name']} ({q2_m['sym']}, M = {q2_m['M']}) trong dung dịch HNO₃ loãng dư thu được **{_fmt_num(q2_v)} lít** khí {q2_g['sym']} (đkc, sản phẩm khử duy nhất). Tìm giá trị của $m$?\n"
        f"   - *Đáp án kiểm tra:* $n({q2_g['sym']}) = {_fmt_num(q2_ngas)}$ mol $\\Rightarrow n(e) = {_fmt_num(q2_ne)}$ mol $\\Rightarrow n({q2_m['sym']}) = {_fmt_num(q2_n)}$ mol $\\Rightarrow$ **$m = {_fmt_num(q2_mass)}$ gam**."
    )


def _generate_dna_offline(seed: Optional[int] = None) -> str:
    """Generates similar DNA/replication formulas, worked example, and 2 exercises."""
    rng = random.Random(seed)

    # Pick clean L multiple of 340 Å so N is multiple of 200
    n_mult = rng.choice([10, 12, 15, 20])
    N = n_mult * 200
    L = int((N // 2) * 3.4)
    C = N // 20
    diff_pct = rng.choice([10, 20, 30])  # %G - %A = diff_pct
    pct_G = (50 + diff_pct) // 2
    pct_A = 50 - pct_G
    G = int(N * pct_G / 100)
    A = int(N * pct_A / 100)
    H = 2 * A + 3 * G
    k = rng.choice([2, 3, 4])
    factor = (2 ** k) - 1
    N_mt = N * factor
    A_mt = A * factor
    G_mt = G * factor

    # Exercise 1
    q1_mult = rng.choice([m for m in (8, 10, 15, 18) if m != n_mult])
    q1_N = q1_mult * 200
    q1_L = int((q1_N // 2) * 3.4)
    q1_diff = rng.choice([10, 20])
    q1_pctA = (50 - q1_diff) // 2
    q1_pctG = 50 - q1_pctA
    q1_A = int(q1_N * q1_pctA / 100)
    q1_G = int(q1_N * q1_pctG / 100)
    q1_H = 2 * q1_A + 3 * q1_G
    q1_k = rng.choice([2, 3])
    q1_Nmt = q1_N * ((2 ** q1_k) - 1)

    # Exercise 2 (given H and C)
    q2_N = rng.choice([1200, 1800, 2400])
    q2_C = q2_N // 20
    q2_G = int(q2_N * 0.3)
    q2_A = (q2_N // 2) - q2_G
    q2_H = q2_N + q2_G
    q2_k = 3
    q2_Gmt = q2_G * ((2 ** q2_k) - 1)

    return (
        "### 🧬 Công thức suy dẫn & biến đổi tương tự (Chế độ Offline — Symbolic)\n"
        "- **Từ chiều dài $L$ (Å) suy ra tổng số nucleotide và số chu kỳ xoắn:**\n"
        "  $$N = \\frac{2L}{3,4}, \\qquad C = \\frac{N}{20} = \\frac{L}{34}$$\n"
        "- **Tính nhanh số từng loại Nu và liên kết hydro:**\n"
        "  $$A = T = \\frac{N}{2} - G, \\qquad H = 2A + 3G = N + G$$\n"
        "- **Nguyên liệu môi trường cung cấp qua $k$ lần nhân đôi:**\n"
        "  $$N_{\\text{mt}} = N \\cdot (2^k - 1), \\qquad A_{\\text{mt}} = A \\cdot (2^k - 1), \\qquad G_{\\text{mt}} = G \\cdot (2^k - 1)$$\n\n"
        "### 📝 Ví dụ tương tự (tự động sinh & tính toán chính xác)\n"
        f"**Đề bài:** Một gen có chiều dài **{L} Å** và có hiệu số giữa Guanin với loại không bổ sung với nó bằng **{diff_pct}%** tổng số nucleotide. "
        f"Tính số nucleotide từng loại, số liên kết hydro và số nucleotide môi trường cung cấp khi gen nhân đôi **{k} lần**.\n"
        f"- **Bước 1:** Tổng số nucleotide $N = \\frac{{2 \\times {L}}}{{3,4}} = {N}$ Nu; chu kỳ xoắn $C = \\frac{{{N}}}{{20}} = {C}$.\n"
        f"- **Bước 2:** Ta có $\\%G + \\%A = 50\\%$ và $\\%G - \\%A = {diff_pct}\\% \\Rightarrow \\%G = \\%X = {pct_G}\\%,\\ \\%A = \\%T = {pct_A}\\%$.\n"
        f"  Suy ra $A = T = {A}$, $G = X = {G}$.\n"
        f"- **Bước 3:** Số liên kết hydro $H = 2A + 3G = 2({A}) + 3({G}) = \\mathbf{{{H}}}$.\n"
        f"- **Bước 4:** Nhân đôi $k = {k}$ lần: $N_{{\\text{{mt}}}} = {N} \\times (2^{k} - 1) = \\mathbf{{{N_mt}}}$ Nu "
        f"($A_{{\\text{{mt}}}} = T_{{\\text{{mt}}}} = {A_mt}$; $G_{{\\text{{mt}}}} = X_{{\\text{{mt}}}} = {G_mt}$).\n\n"
        "### ✍️ Bài tập tự luyện\n"
        f"1. **Bài 1:** Một gen có chiều dài **{q1_L} Å**, trong đó $\\%A - \\%G = {q1_diff}\\%$. Khi gen nhân đôi liên tiếp **{q1_k} lần**, hãy tính $A, T, G, X$, số liên kết hydro $H$ và $N_{{\\text{{mt}}}}$.\n"
        f"   - *Đáp án kiểm tra:* $N = {q1_N}$, $A = T = {q1_G}$? Không, vì $\\%A > \\%G$ nên $\\%A = {q1_pctG}\\%, \\%G = {q1_pctA}\\%$ $\\Rightarrow$ **$A = T = {q1_G}$**, **$G = X = {q1_A}$**, **$H = {2 * q1_G + 3 * q1_A}$**, **$N_{{\\text{{mt}}}} = {q1_Nmt}$**.\n"
        f"2. **Bài 2:** Một gen có **{q2_C} chu kỳ xoắn** và có **{q2_H} liên kết hydro**. Tính số nucleotide loại G môi trường cung cấp khi gen nhân đôi **{q2_k} lần**.\n"
        f"   - *Đáp án kiểm tra:* $N = 20 \\times {q2_C} = {q2_N}$; $G = H - N = {q2_G}$ $\\Rightarrow$ **$G_{{\\text{{mt}}}} = {q2_G} \\times (2^{{{q2_k}}} - 1) = {q2_Gmt}$ Nu**."
    )


def _generate_generic_offline(
    query: str,
    context_text: str,
    formulas: List[str],
    examples: List[str],
    seed: Optional[int] = None,
) -> str:
    """Fallback offline generator for any document containing formulas or numeric examples."""
    rng = random.Random(seed)
    lines: List[str] = [
        "### 📐 Công thức & Mẫu trích xuất từ tài liệu (Chế độ Offline — Template)"
    ]
    if formulas:
        for f in formulas[:4]:
            lines.append(f"- **Công thức gốc:** `{f}`")
    else:
        lines.append("- *Chưa phát hiện biểu thức `=` tường minh trong đoạn ngữ cảnh hiện tại; sử dụng trực tiếp dữ kiện văn bản.*")

    if examples:
        base_ex = examples[0]
        scale = rng.choice([2, 3, 0.5])

        def _scale_num(m: re.Match) -> str:
            raw = m.group(0).replace(",", ".")
            try:
                v = float(raw) * scale
                return _fmt_num(v)
            except ValueError:
                return m.group(0)

        mutated_ex = re.sub(r"\b\d+(?:[.,]\d+)?\b", _scale_num, base_ex, count=2)
        lines.append("\n### 📝 Ví dụ tương tự (biến đổi tham số từ ví dụ gốc)")
        lines.append(f"- **Ví dụ gốc trong tài liệu:** {base_ex}")
        lines.append(f"- **Biến thể tham số (hệ số ×{_fmt_num(scale)}):** {mutated_ex}")
        lines.append("- **Hướng dẫn áp dụng:** Thay bộ số liệu mới ở trên vào đúng công thức gốc để tính kết quả tương ứng.")

    lines.append("\n### ✍️ Bài tập tự luyện gợi ý")
    if formulas:
        lines.append(f"1. **Bài 1 (Biến đổi công thức):** Từ biểu thức `{formulas[0]}`, hãy rút ra công thức tính từng đại lượng còn lại và nêu đơn vị đo chuẩn.")
        lines.append(f"2. **Bài 2 (Vận dụng):** Tự chọn một bộ giá trị đầu vào hợp lệ cho `{formulas[0]}` (khác với ví dụ trong tài liệu) và trình bày lời giải từng bước.")
    else:
        lines.append("1. **Bài 1:** Dựa vào quy tắc trong đoạn tài liệu trên, hãy đặt một tình huống số liệu mới và giải thích từng bước thực hiện.")
    return "\n".join(lines)


def generate_practice_offline(
    query: str,
    context_text: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generates similar formulas, worked examples, and exercises 100% offline."""
    combined_norm = strip_accents(f"{query}\n{context_text}".lower())
    formulas, examples = extract_formulas_and_examples(context_text)

    if any(k in combined_norm for k in ("oxi hoa", "chat khu", "bao toan electron", "n(e ", "24,79", "hno3")):
        body = _generate_redox_offline(seed=seed)
        domain = "chemistry_redox"
    elif any(k in combined_norm for k in ("adn", "nucleotide", "adenin", "guanin", "lien ket hydro", "chu ky xoan", "nhan doi")):
        body = _generate_dna_offline(seed=seed)
        domain = "biology_dna"
    else:
        body = _generate_generic_offline(query, context_text, formulas, examples, seed=seed)
        domain = "generic_formula"

    return {
        "answer": body,
        "engine": "offline_symbolic_template",
        "domain": domain,
        "formulas_found": len(formulas),
        "examples_found": len(examples),
    }


# ---------------------------------------------------------------------------
# Hướng 2: Online LLM Generator (with automatic Offline fallback)
# ---------------------------------------------------------------------------

def _build_online_prompt(query: str, context_text: str, formulas: List[str], examples: List[str]) -> str:
    f_block = "\n".join(f"- {f}" for f in formulas[:5]) if formulas else "- (Trích trực tiếp từ ngữ cảnh bên dưới)"
    e_block = "\n".join(f"- {e}" for e in examples[:3]) if examples else "- (Chưa có ví dụ mẫu tường minh)"
    return (
        "Bạn là trợ giảng học thuật. Chỉ dựa trên kiến thức, công thức và ví dụ mẫu trong tài liệu dưới đây, "
        "hãy thực hiện yêu cầu của người học bằng tiếng Việt rõ ràng, chính xác về mặt tính toán:\n"
        "1) Đưa ra 1-2 công thức biến đổi / công thức liên quan tương tự từ công thức gốc.\n"
        "2) Sinh 1 ví dụ minh họa MỚI (khác số liệu/tình huống với ví dụ gốc) kèm lời giải chi tiết từng bước.\n"
        "3) Sinh 2 bài tập tự luyện MỚI (kèm đáp án kiểm tra ngắn gọn ở cuối mỗi bài).\n\n"
        f"Yêu cầu người học: {query}\n\n"
        f"Công thức trích từ tài liệu:\n{f_block}\n\n"
        f"Ví dụ mẫu trích từ tài liệu:\n{e_block}\n\n"
        f"Ngữ cảnh tài liệu liên quan:\n{context_text[:2500]}"
    )


def _call_online_llm(prompt: str, timeout_sec: float = 20.0) -> Optional[Tuple[str, str]]:
    """Calls configured Online LLM (Gemini, OpenAI-compatible, or Ollama). Returns (text, provider) or None."""
    import httpx

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1200},
        }
        try:
            resp = httpx.post(url, json=payload, timeout=timeout_sec)
            if resp.status_code == 200:
                data = resp.json()
                text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    .strip()
                )
                if text:
                    return text, f"online_gemini:{model}"
        except Exception:
            pass

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    openai_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    if openai_key:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
        try:
            resp = httpx.post(
                f"{openai_base}/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,
                    "max_tokens": 1200,
                },
                timeout=timeout_sec,
            )
            if resp.status_code == 200:
                text = (
                    resp.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if text:
                    return text, f"online_openai:{model}"
        except Exception:
            pass

    ollama_model = os.environ.get("OLLAMA_MODEL", "").strip()
    if ollama_model:
        ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        try:
            resp = httpx.post(
                f"{ollama_host}/v1/chat/completions",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,
                },
                timeout=timeout_sec,
            )
            if resp.status_code == 200:
                text = (
                    resp.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if text:
                    return text, f"online_ollama:{ollama_model}"
        except Exception:
            pass

    return None


def generate_practice_content(
    query: str,
    context_text: str,
    mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Unified entry point:
    - mode='offline' (Hướng 1): Symbolic + template generator.
    - mode='online' (Hướng 2): Online LLM seeded with document formulas/examples,
      automatically falling back to Hướng 1 if no provider is configured/reachable.
    - mode=None: Auto-detects via `is_online_llm_enabled()`.
    """
    resolved_mode = mode or ("online" if is_online_llm_enabled() else "offline")
    if resolved_mode == "online":
        formulas, examples = extract_formulas_and_examples(context_text)
        prompt = _build_online_prompt(query, context_text, formulas, examples)
        llm_res = _call_online_llm(prompt)
        if llm_res is not None:
            text, provider = llm_res
            return {
                "answer": f"{text}\n\n> *Nội dung ví dụ & bài tập được sinh tự động ({provider}) bám sát công thức trong tài liệu.*",
                "engine": provider,
                "domain": "online_llm",
                "formulas_found": len(formulas),
                "examples_found": len(examples),
            }
        # Fallback to Hướng 1 if online LLM unavailable
        fallback = generate_practice_offline(query, context_text, seed=seed)
        fallback["engine"] = "offline_symbolic_fallback"
        fallback["answer"] += (
            "\n\n> *Lưu ý: Chế độ Online LLM chưa có API key (`GEMINI_API_KEY` / `OPENAI_API_KEY` / `OLLAMA_MODEL`) "
            "hoặc không kết nối được, hệ thống đã tự động dùng bộ sinh Offline (Hướng 1).*"
        )
        return fallback

    return generate_practice_offline(query, context_text, seed=seed)
