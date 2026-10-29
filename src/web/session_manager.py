"""
Phase 5.2 — Chat Session Manager & Conversational Context Resolver.
Manages multi-turn conversation sessions, document attachments, and conversational memory.

STRICT RULE: NO TRAINING, NO GRADIENT UPDATES, FROZEN INFERENCE ONLY.
"""

import os
import re
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Set


def generate_title_from_query(query: str, existing_titles: Optional[Set[str]] = None) -> str:
    """
    Generates a concise, meaningful session title from a user query, avoiding duplicates.
    Examples:
      'Kiến trúc SA-CMS là gì?' -> 'Kiến trúc SA-CMS'
      'Cơ chế bộ nhớ L1 L2 L3 hoạt động thế nào?' -> 'Cơ chế bộ nhớ L1 L2 L3'
      'Giải thích về Nested Learning và CMS' -> 'Nested Learning và CMS'
    Avoids duplicate titles by adding a numeric suffix '(2)', '(3)' if necessary.
    Never exposes internal IDs.
    """
    clean = query.strip()
    clean = re.sub(r"\s+", " ", clean)

    # Conversational prefixes in Vietnamese and English
    prefixes = [
        "cho tôi biết về", "cho tôi biết", "cho em biết về", "cho em biết",
        "hãy giải thích về", "giải thích về", "giải thích",
        "hãy phân tích về", "phân tích về", "phân tích",
        "hãy tóm tắt về", "hãy tóm tắt", "tóm tắt về", "tóm tắt",
        "cho mình hỏi về", "cho mình hỏi", "xin hỏi về", "xin hỏi",
        "tìm hiểu về", "trình bày về", "trình bày",
        "hãy cho biết về", "hãy cho biết", "chi tiết về", "thông tin về",
        "bạn có thể cho biết", "bạn có biết",
        "what is", "explain", "tell me about", "summarize",
    ]
    for p in prefixes:
        if clean.lower().startswith(p):
            clean = clean[len(p):].strip()
            break

    # Strip trailing punctuation before checking question suffixes
    clean = clean.strip(" ?:.,-!\"'’“”[](){}")

    # Trailing question phrases
    suffixes = [
        "là gì vậy", "là gì", "như thế nào vậy", "như thế nào",
        "ra sao vậy", "ra sao", "hoạt động thế nào", "hoạt động ra sao",
        "thế nào", "đúng không", "phải không", "được không",
        "is it", "what is it",
    ]
    for _ in range(2):
        clean = clean.strip(" ?:.,-!\"'’“”[](){}")
        clean_lower = clean.lower()
        matched = False
        for s in suffixes:
            if clean_lower.endswith(s):
                clean = clean[:-len(s)].strip()
                matched = True
                break
        if not matched:
            break

    # Strip punctuation again
    clean = clean.strip(" ?:.,-!\"'’“”[](){}")

    if not clean:
        words = query.strip().split()
        clean = " ".join(words[:5]) if words else "Cuộc trò chuyện mới"

    # Truncate cleanly up to ~36 characters
    if len(clean) > 36:
        truncated = clean[:36]
        last_space = truncated.rfind(" ")
        if last_space > 15:
            clean = truncated[:last_space] + "..."
        else:
            clean = truncated + "..."

    # Capitalize first character
    clean = clean[:1].upper() + clean[1:]

    # Deduplicate against existing titles if provided
    if existing_titles:
        base_title = clean
        counter = 2
        while clean in existing_titles:
            clean = f"{base_title} ({counter})"
            counter += 1

    return clean


class ChatSession:
    """Represents an active multi-turn conversation session."""

    def __init__(
        self,
        session_id: str,
        title: str = "Cuộc trò chuyện mới",
        document_ids: Optional[List[str]] = None,
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ):
        self.session_id = session_id
        self.title = title
        self.document_ids = document_ids or []
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or self.created_at
        self.messages = messages or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "document_ids": self.document_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        return cls(
            session_id=data["session_id"],
            title=data.get("title", "Cuộc trò chuyện mới"),
            document_ids=data.get("document_ids", []),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            messages=data.get("messages", []),
        )

    def add_message(
        self,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        routing: Optional[str] = None,
        refused: bool = False,
        refusal_reason: Optional[str] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        existing_titles: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        msg_id = f"msg_{uuid.uuid4().hex[:10]}"
        msg = {
            "id": msg_id,
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "citations": citations or [],
            "evidence": evidence or [],
            "telemetry": telemetry or {},
            "routing": routing or "DIRECT_LOOKUP",
            "refused": refused,
            "refusal_reason": refusal_reason,
            "diagnostics": diagnostics or {},
        }
        self.messages.append(msg)
        self.updated_at = time.time()

        # Auto-update session title based on first user message if default
        if role == "user" and (self.title in ("Cuộc trò chuyện mới", "New Chat", "") or not self.title):
            self.title = generate_title_from_query(content, existing_titles)

        return msg

    def add_system_event(
        self,
        event_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        msg_id = f"msg_{uuid.uuid4().hex[:10]}"
        msg = {
            "id": msg_id,
            "role": "system",
            "type": event_type,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
            "citations": [],
            "evidence": [],
            "telemetry": {},
            "routing": "SYSTEM_EVENT",
            "refused": False,
            "refusal_reason": None,
            "diagnostics": {},
        }
        self.messages.append(msg)
        self.updated_at = time.time()
        return msg

    def attach_document(self, doc_id: str) -> bool:
        if doc_id not in self.document_ids:
            self.document_ids.append(doc_id)
            self.updated_at = time.time()
            return True
        return False

    def detach_document(self, doc_id: str) -> bool:
        if doc_id in self.document_ids:
            self.document_ids.remove(doc_id)
            self.updated_at = time.time()
            return True
        return False


class SessionManager:
    """Manages chat session lifecycle, persistence, and conversation context."""

    def __init__(self, storage_dir: str = "data/chat_sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_file = self.storage_dir / "sessions.json"
        self._sessions: Dict[str, ChatSession] = {}
        self._load_sessions()

    def _load_sessions(self):
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        session = ChatSession.from_dict(item)
                        self._sessions[session.session_id] = session
            except Exception:
                self._sessions = {}

    def _save_sessions(self):
        try:
            data = [s.to_dict() for s in self._sessions.values()]
            with open(self.sessions_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def create_session(
        self,
        title: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> ChatSession:
        session_id = f"chat_{uuid.uuid4().hex[:12]}"
        session = ChatSession(
            session_id=session_id,
            title=title or "Cuộc trò chuyện mới",
            document_ids=document_ids or [],
        )
        self._sessions[session_id] = session
        self._save_sessions()
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    def get_existing_titles(self, exclude_session_id: Optional[str] = None) -> Set[str]:
        return {
            s.title for s_id, s in self._sessions.items()
            if s_id != exclude_session_id and s.title and s.title not in ("Cuộc trò chuyện mới", "New Chat")
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        # Sort newest first
        sorted_sessions = sorted(
            self._sessions.values(), key=lambda s: s.updated_at, reverse=True
        )
        now = time.time()
        one_day = 86400

        result = []
        for s in sorted_sessions:
            diff = now - s.updated_at
            if diff < one_day:
                group = "Hôm nay"
            elif diff < 2 * one_day:
                group = "Hôm qua"
            else:
                group = "Trước đó"

            result.append({
                "session_id": s.session_id,
                "title": s.title,
                "document_ids": s.document_ids,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "message_count": len(s.messages),
                "time_group": group,
            })
        return result

    def rename_session(self, session_id: str, new_title: str) -> bool:
        session = self.get_session(session_id)
        if session:
            session.title = new_title.strip() or "Cuộc trò chuyện"
            session.updated_at = time.time()
            self._save_sessions()
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save_sessions()
            return True
        return False

    def attach_document(self, session_id: str, doc_id: str) -> bool:
        session = self.get_session(session_id)
        if session:
            attached = session.attach_document(doc_id)
            if attached:
                self._save_sessions()
            return True
        return False

    def record_document_event(
        self,
        session_id: str,
        doc_id: str,
        title: str,
        file_name: Optional[str] = None,
        file_size: Optional[str] = None,
        source: str = "upload",
    ) -> Optional[Dict[str, Any]]:
        session = self.get_session(session_id)
        if not session:
            return None

        display_title = title or file_name or doc_id
        content = f"Đã thêm tài liệu \"{display_title}\" vào cuộc trò chuyện."
        event = session.add_system_event(
            event_type="document_attached",
            content=content,
            metadata={
                "document_id": doc_id,
                "title": display_title,
                "file_name": file_name or display_title,
                "file_size": file_size,
                "source": source,
            },
        )
        self._save_sessions()
        return event

    def detach_document(self, session_id: str, doc_id: str) -> bool:
        session = self.get_session(session_id)
        if session:
            detached = session.detach_document(doc_id)
            if detached:
                self._save_sessions()
            return True
        return False

    def save(self):
        self._save_sessions()

    def resolve_conversation_context(
        self, session_id: str, current_query: str
    ) -> str:
        """
        Task 12: Resolves follow-up pronouns and elliptical questions using conversation history.
        E.g.:
        Q1: 'CMS là gì?' -> AI: '...'
        Q2: 'Nó khác RAG thế nào?' -> Context-aware query: 'CMS khác RAG thế nào'
        """
        session = self.get_session(session_id)
        if not session or not session.messages:
            return current_query

        # Get prior user query (skip current query if already appended to session)
        user_msgs = [m for m in session.messages if m.get("role") == "user"]
        if not user_msgs:
            return current_query

        if len(user_msgs) >= 2 and user_msgs[-1].get("content", "").strip() == current_query.strip():
            last_user_query = user_msgs[-2].get("content", "").strip()
        elif len(user_msgs) >= 1 and user_msgs[-1].get("content", "").strip() != current_query.strip():
            last_user_query = user_msgs[-1].get("content", "").strip()
        else:
            return current_query

        # Pronouns and contextual indicators in Vietnamese and English
        pronouns_vi = ["nó", "chúng", "điều này", "phương pháp này", "công nghệ này", "mô hình này", "hệ thống này", "thứ này"]
        pronouns_en = ["it", "they", "this method", "this system", "this model", "these"]

        q_lower = current_query.lower().strip()

        # Check if query starts with or contains follow-up indicators
        is_follow_up = False
        target_pronoun = None

        import re
        for p in pronouns_vi + pronouns_en:
            # Word boundary matching
            pat = r"\b" + re.escape(p) + r"\b"
            if re.search(pat, q_lower):
                is_follow_up = True
                target_pronoun = p
                break

        # Check if elliptical question (e.g., "Thế còn RAG?", "Còn về ưu điểm thì sao?")
        elliptical_starters = ["còn", "thế còn", "thế thì", "vậy còn", "what about", "how about"]
        for starter in elliptical_starters:
            if q_lower.startswith(starter):
                is_follow_up = True
                break

        if not is_follow_up:
            return current_query

        # Extract primary subject from last user query
        clean_last = last_user_query
        for stop in ["ở phần 2,", "ở phần 1,", "ở chương", "trong phần", "là gì", "la gi", "như thế nào", "ra sao", "thế nào", "giải thích", "cho biết", "what is", "explain"]:
            clean_last = re.sub(re.escape(stop), "", clean_last, flags=re.IGNORECASE)
        subject = clean_last.strip(" ?:.,-!\"'")

        if not subject:
            subject = last_user_query.strip(" ?:.,-!\"'")

        if target_pronoun:
            # Replace pronoun with resolved subject using word boundary
            pat = r"\b" + re.escape(target_pronoun) + r"\b"
            resolved = re.sub(pat, subject, current_query, count=1, flags=re.IGNORECASE)
            return resolved
        else:
            # Elliptical combination
            return f"{subject}: {current_query}"
