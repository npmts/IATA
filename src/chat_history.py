import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chat_history.sqlite")


@dataclass
class ConversationMeta:
    id: str
    title: Optional[str]
    created_at: str


class ChatHistory:
    """Lightweight helper around SQLite for storing chat transcripts."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_meta (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    conversation_id TEXT NOT NULL,
                    message_order INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, message_order),
                    FOREIGN KEY (conversation_id) REFERENCES conversation_meta(id)
                )
                """
            )

    def create_conversation(
        self,
        *,
        title: Optional[str] = None,
        initial_message: Optional[Dict[str, str]] = None,
    ) -> str:
        conversation_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversation_meta (id, title, created_at) VALUES (?, ?, ?)",
                (conversation_id, title, created_at),
            )
        if initial_message:
            self.save_message(
                conversation_id,
                role=initial_message.get("role", "assistant"),
                content=initial_message.get("content", ""),
                created_at=created_at,
            )
        return conversation_id

    def save_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        created_at: Optional[str] = None,
    ) -> None:
        timestamp = created_at or datetime.utcnow().isoformat()
        with self._connect() as conn:
            next_order = conn.execute(
                "SELECT COALESCE(MAX(message_order) + 1, 0) FROM conversation_messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0]
            conn.execute(
                """
                INSERT OR REPLACE INTO conversation_messages
                (conversation_id, message_order, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, next_order, role, content, timestamp),
            )
            if role == "user":
                current_title = conn.execute(
                    "SELECT title FROM conversation_meta WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if current_title and (current_title[0] is None or current_title[0].strip() == ""):
                    summary = content.strip().splitlines()[0][:80]
                    conn.execute(
                        "UPDATE conversation_meta SET title = ? WHERE id = ?",
                        (summary or "Conversation", conversation_id),
                    )

    def list_conversations(self, limit: int = 20) -> List[ConversationMeta]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, COALESCE(NULLIF(title, ''), 'Conversation'), created_at
                FROM conversation_meta
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [ConversationMeta(*row) for row in rows]

    def load_conversation(self, conversation_id: str) -> List[Dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM conversation_messages
                WHERE conversation_id = ?
                ORDER BY message_order ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

