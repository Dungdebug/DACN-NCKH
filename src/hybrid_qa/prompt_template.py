"""
Prompt Engineering & Precision Control Module for RAG / Hybrid QA.

Enforces 3 Hard Constraints via Rigorous Prompt Templates:
1. Strict Grounding: System is explicitly barred from using external or pre-trained knowledge.
   Answers must be synthesized exclusively from provided context passages.
2. Anti-Hallucination & Legitimate Refusal: Grants the model explicit permission and command
   to reply with a standardized refusal message ("Tài liệu được cung cấp không chứa thông tin...")
   if the provided passages do not supply verified evidence.
3. Citation Provenance: Enforces bracketed citation markers ([DOC...::P...] or [1], [2])
   for every factual claim made in the generated answer.

Supported Formats:
- Plain Text (Optimal for Base SLMs like SmolLM2-135M)
- ChatML (Optimal for Qwen-2.5-1.5B Instruct)
- Phi-3 Instruct (<|system|>...<|user|>...)
- Llama-3 Instruct (<|begin_of_text|><|start_header_id|>...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


# -----------------------------------------------------------------------------
# HARD CONSTRAINT DEFINITIONS (RÀNG BUỘC CỨNG)
# -----------------------------------------------------------------------------

COT_SYSTEM_INSTRUCTION_VI = (
    "Bạn là một chuyên gia phân tích tài liệu. Nhiệm vụ của bạn là đọc các ngữ cảnh được cung cấp "
    "và trả lời câu hỏi của người dùng một cách mạch lạc, chi tiết và đi thẳng vào trọng tâm."
)

COT_MANDATORY_PROCESS_VI = (
    "[Quy trình xử lý bắt buộc]\n\n"
    "1. Tuyệt đối không chép phạt: Không liệt kê lại các tiêu đề, không copy y nguyên định dạng gạch đầu dòng (*, -) của tài liệu gốc. Hãy dùng ngôn ngữ tự nhiên để diễn đạt lại (Abstractive).\n\n"
    "2. Cấu trúc trả lời:\n"
    "   - Mở bài: Trực tiếp trả lời thẳng vào câu hỏi (Ví dụ: \"Tài liệu đề cập đến X loại mã hóa chính...\").\n"
    "   - Thân bài: Phân tích chi tiết từng loại/từng ý dựa trên ngữ cảnh. Gộp các ý trùng lặp lại với nhau.\n\n"
    "3. Bám sát trọng tâm: Chỉ trả lời những gì người dùng hỏi. Nếu tài liệu chứa thông tin dư thừa (như bài tập, ví dụ không liên quan), hãy mạnh dạn bỏ qua.\n\n"
    "4. Trích dẫn: Chèn số trích dẫn [1], [2] một cách tự nhiên vào cuối mỗi câu hoặc mỗi đoạn hoàn chỉnh."
)

STRICT_GROUNDING_VI = (
    "1. RÀNG BUỘC NGỮ CẢNH (Strict Grounding): Chỉ được phép trả lời dựa trên những thông tin "
    "được cung cấp rõ ràng trong phần 'NGỮ CẢNH TÀI LIỆU' dưới đây. Tuyệt đối KHÔNG suy đoán, "
    "không tự ý mở rộng và không sử dụng kiến thức bên ngoài tài liệu."
)

ANTI_HALLUCINATION_VI = (
    "2. CHỐNG ẢO GIÁC (Anti-Hallucination): Nếu ngữ cảnh được cung cấp không chứa thông tin hoặc "
    "không đủ bằng chứng chắc chắn để trả lời câu hỏi, bạn BẮT BUỘC phải trả lời chính xác câu sau: "
    "\"Tài liệu được cung cấp không chứa thông tin để trả lời câu hỏi này.\" "
    "Tuyệt đối không được tự bịa đặt, suy đoán hoặc trả lời sai sự thật."
)

CITATION_VI = (
    "3. BẮT BUỘC TRÍCH DẪN (Citation): Mỗi câu trả lời hoặc mệnh đề sự kiện PHẢI kèm mã trích dẫn "
    "nguồn trong dấu ngoặc vuông (ví dụ: [DOC...::P...] hoặc [1]) tương ứng chính xác với đoạn văn "
    "được dùng làm căn cứ."
)

ABSTRACTIVE_SYNTHESIS_VI = (
    "4. TỔNG HỢP VĂN PHONG TỰ NHIÊN (Abstractive Synthesis): Tuyệt đối KHÔNG sao chép nguyên văn "
    "định dạng thô của ngữ cảnh (không chép lại các dấu hoa thị *, gạch đầu dòng -, tiêu đề in hoa cụt lủn). "
    "Hãy tự tổng hợp và viết lại thành một đoạn văn hoàn chỉnh, mạch lạc bằng giọng văn của bạn. "
    "Toàn bộ mã trích dẫn nguồn [id] BẮT BUỘC phải đặt ở CUỐI mỗi câu hoàn chỉnh, không chèn chen ngang giữa câu."
)

STRICT_GROUNDING_EN = (
    "1. STRICT GROUNDING: Answer ONLY based on the facts directly stated in the 'CONTEXT' below. "
    "Do NOT assume, extrapolate, or bring in any external background knowledge."
)

ANTI_HALLUCINATION_EN = (
    "2. ANTI-HALLUCINATION: If the context does not contain sufficient facts to answer the question, "
    "you MUST state strictly: \"The provided document does not contain sufficient information to answer this question.\" "
    "Do NOT hallucinate or guess."
)

CITATION_EN = (
    "3. MANDATORY CITATIONS: Every claim must be attributed with its exact passage citation identifier "
    "in square brackets (e.g. [DOC...::P...] or [1])."
)

ABSTRACTIVE_SYNTHESIS_EN = (
    "4. ABSTRACTIVE SYNTHESIS: Do NOT verbatim copy raw markdown formatting (no '*', '-', or all-caps headings). "
    "Synthesize the evidence into a fluent, cohesive prose paragraph in your own words. "
    "Place all citation markers [id] strictly at the END of complete sentences."
)


class GroundedPromptTemplate:
    """
    Template engine constructing strictly calibrated prompts for Small Language Models (SLMs).
    """

    @classmethod
    def format_context_passages(cls, passages: List[Union[Dict[str, Any], Any]]) -> str:
        """Format passages into an identifiable indexed context block."""
        formatted_blocks = []
        for idx, p in enumerate(passages, start=1):
            if isinstance(p, dict):
                p_id = p.get("passage_id") or f"DOC::P{idx:03d}"
                text = p.get("text") or ""
                sec = p.get("section_title") or ""
            else:
                p_id = getattr(p, "passage_id", f"DOC::P{idx:03d}")
                text = getattr(p, "text", "")
                sec = getattr(p, "section_title", "")

            sec_header = f" [{sec}]" if sec else ""
            formatted_blocks.append(f"[{p_id}]{sec_header}\n{text.strip()}")

        return "\n\n".join(formatted_blocks)

    @classmethod
    def build_system_instruction(cls, language: str = "vi") -> str:
        """Build the authoritative system prompt containing the hard constraints."""
        if language in ("vi", "vietnamese"):
            return (
                "Bạn là Trợ lý AI Chuyên gia Nghiên cứu Tài liệu. Hãy tuân thủ nghiêm ngặt 4 nguyên tắc sau:\n"
                f"{STRICT_GROUNDING_VI}\n"
                f"{ANTI_HALLUCINATION_VI}\n"
                f"{CITATION_VI}\n"
                f"{ABSTRACTIVE_SYNTHESIS_VI}\n"
                "Hãy trả lời ngắn gọn, trực diện, mạch lạc và chính xác tuyệt đối."
            )
        else:
            return (
                "You are an Academic Document Intelligence AI Assistant. Strictly adhere to these 4 rules:\n"
                f"{STRICT_GROUNDING_EN}\n"
                f"{ANTI_HALLUCINATION_EN}\n"
                f"{CITATION_EN}\n"
                f"{ABSTRACTIVE_SYNTHESIS_EN}\n"
                "Provide a direct, concise, factual, and strictly grounded response."
            )

    @classmethod
    def build_prompt(
        cls,
        question: str,
        context_passages: Optional[List[Any]] = None,
        context_text: Optional[str] = None,
        language: str = "vi",
        format_style: str = "plain",  # "plain", "chatml", "phi3", "llama3"
    ) -> str:
        """
        Assemble the full prompt with hard grounding constraints.
        """
        system_instr = cls.build_system_instruction(language=language)

        if context_text is None:
            if context_passages:
                ctx_body = cls.format_context_passages(context_passages)
            else:
                ctx_body = ""
        else:
            ctx_body = context_text.strip()

        is_vi = language in ("vi", "vietnamese")
        ctx_label = "NGỮ CẢNH TÀI LIỆU" if is_vi else "DOCUMENT CONTEXT"
        q_label = "CÂU HỎI" if is_vi else "QUESTION"
        a_label = "CÂU TRẢ LỜI" if is_vi else "ANSWER"

        # 1. Plain Text Format (Suitable for Base SLMs like SmolLM2-135M Base)
        if format_style == "plain":
            if ctx_body:
                prompt = (
                    f"### QUY TẮC BẮT BUỘC:\n{system_instr}\n\n"
                    f"### {ctx_label}:\n{ctx_body}\n\n"
                    f"### {q_label}: {question}\n"
                    f"### {a_label}:"
                )
            else:
                prompt = (
                    f"### QUY TẮC BẮT BUỘC:\n{system_instr}\n\n"
                    f"### {q_label}: {question}\n"
                    f"### {a_label}:"
                )
            return prompt

        # 2. ChatML Format (Qwen-2.5-1.5B Instruct)
        elif format_style == "chatml":
            user_content = f"{ctx_label}:\n{ctx_body}\n\n{q_label}: {question}" if ctx_body else question
            return (
                f"<|im_start|>system\n{system_instr}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        # 3. Phi-3 Format (microsoft/Phi-3-mini-4k-instruct)
        elif format_style == "phi3":
            user_content = f"{ctx_label}:\n{ctx_body}\n\n{q_label}: {question}" if ctx_body else question
            return (
                f"<|system|>\n{system_instr}<|end|>\n"
                f"<|user|>\n{user_content}<|end|>\n"
                f"<|assistant|>\n"
            )

        # 4. Llama-3 Format (meta-llama/Llama-3.2-1B-Instruct)
        elif format_style == "llama3":
            user_content = f"{ctx_label}:\n{ctx_body}\n\n{q_label}: {question}" if ctx_body else question
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_instr}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_content}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

        # Fallback default
        return f"{system_instr}\n\n{ctx_label}:\n{ctx_body}\n\n{q_label}: {question}\n{a_label}:"

    @classmethod
    def build_cot_prompt(
        cls,
        question: str,
        context_passages: Optional[List[Any]] = None,
        context_text: Optional[str] = None,
        language: str = "vi",
        format_style: str = "plain",
    ) -> str:
        """
        Builds the Chain-of-Thought (CoT) structured prompt:
        - [Chỉ thị Hệ thống]: Document analyst persona and direct reasoning mandate.
        - [Ngữ cảnh từ tài liệu]: Evidence passages.
        - [Câu hỏi của người dùng]: User query.
        - [Quy trình xử lý bắt buộc]: Abstractive synthesis, Mở bài, Thân bài, Bám sát trọng tâm, Trích dẫn [1], [2].
        """
        if context_text is None:
            if context_passages:
                ctx_body = cls.format_context_passages(context_passages)
            else:
                ctx_body = ""
        else:
            ctx_body = context_text.strip()

        is_vi = language in ("vi", "vietnamese")
        sys_directive = COT_SYSTEM_INSTRUCTION_VI if is_vi else (
            "You are a Document Analysis Specialist. Read the provided context passages carefully "
            "and answer the user's question directly, cohesively, thoroughly, and strictly to the point."
        )
        proc_directive = COT_MANDATORY_PROCESS_VI if is_vi else (
            "[Mandatory Reasoning Process]\n\n"
            "1. Abstractive Synthesis: Do not verbatim copy raw markdown bullets (*, -) or disconnected headings. Rewrite in natural language.\n"
            "2. Answer Structure:\n"
            "   - Opening: Directly state the core answer to the question.\n"
            "   - Body: Analyze each key aspect/category based on context, merging duplicates.\n"
            "3. Grounded Focus: Focus strictly on what was asked; omit irrelevant noise.\n"
            "4. Citations: Insert bracketed citations [1], [2] naturally at sentence ends."
        )

        sys_header = "[Chỉ thị Hệ thống]" if is_vi else "[System Instruction]"
        ctx_header = "[Ngữ cảnh từ tài liệu]" if is_vi else "[Document Context]"
        q_header = "[Câu hỏi của người dùng]" if is_vi else "[User Question]"

        if format_style == "plain":
            return (
                f"{sys_header}\n{sys_directive}\n\n"
                f"{ctx_header}\n{ctx_body}\n\n"
                f"{q_header}\n{question}\n\n"
                f"{proc_directive}\n\n"
                f"{STRICT_GROUNDING_VI if is_vi else STRICT_GROUNDING_EN}\n"
                f"{ANTI_HALLUCINATION_VI if is_vi else ANTI_HALLUCINATION_EN}\n"
                f"{CITATION_VI if is_vi else CITATION_EN}\n\n"
                f"[{'Câu trả lời hoàn chỉnh' if is_vi else 'Synthesized Answer'}]:"
            )
        elif format_style == "chatml":
            user_content = (
                f"{ctx_header}\n{ctx_body}\n\n"
                f"{q_header}\n{question}\n\n"
                f"{proc_directive}"
            )
            return (
                f"<|im_start|>system\n{sys_directive}\n"
                f"{STRICT_GROUNDING_VI if is_vi else STRICT_GROUNDING_EN}\n"
                f"{ANTI_HALLUCINATION_VI if is_vi else ANTI_HALLUCINATION_EN}\n"
                f"{CITATION_VI if is_vi else CITATION_EN}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        elif format_style == "phi3":
            user_content = (
                f"{ctx_header}\n{ctx_body}\n\n"
                f"{q_header}\n{question}\n\n"
                f"{proc_directive}"
            )
            return (
                f"<|system|>\n{sys_directive}\n"
                f"{STRICT_GROUNDING_VI if is_vi else STRICT_GROUNDING_EN}\n"
                f"{ANTI_HALLUCINATION_VI if is_vi else ANTI_HALLUCINATION_EN}\n"
                f"{CITATION_VI if is_vi else CITATION_EN}<|end|>\n"
                f"<|user|>\n{user_content}<|end|>\n"
                f"<|assistant|>\n"
            )
        elif format_style == "llama3":
            user_content = (
                f"{ctx_header}\n{ctx_body}\n\n"
                f"{q_header}\n{question}\n\n"
                f"{proc_directive}"
            )
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{sys_directive}\n"
                f"{STRICT_GROUNDING_VI if is_vi else STRICT_GROUNDING_EN}\n"
                f"{ANTI_HALLUCINATION_VI if is_vi else ANTI_HALLUCINATION_EN}\n"
                f"{CITATION_VI if is_vi else CITATION_EN}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_content}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        return cls.build_prompt(question, context_passages, context_text, language, format_style)
