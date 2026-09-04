# -*- coding: utf-8 -*-
"""SQLite хранилище истории чатов."""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = Path(DATA_DIR) / "chat_history.db"


class Message:
    """Сообщение чата."""

    def __init__(self, id_: int, role: str, content: str, timestamp: str):
        self.id = id_
        self.role = role  # "user", "assistant"
        self.content = content
        self.timestamp = timestamp


class ChatRepo:
    """Репозиторий для работы с историей чата в SQLite."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Создать таблицы если их нет."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS session_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
        """)
        conn.commit()
        conn.close()

    def add_message(self, role: str, content: str, session_id: Optional[int] = None) -> None:
        """Добавить сообщение в общий поток и в сессию если указана."""
        timestamp = datetime.now().isoformat()
        conn = self._get_conn()
        cur = conn.cursor()
        
        # Добавляем в общий поток
        cur.execute(
            "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
            (role, content, timestamp)
        )
        
        # Если указана сессия - добавляем и туда
        if session_id is not None:
            cur.execute(
                "INSERT INTO session_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, timestamp)
            )
            # Обновляем время обновления сессии
            cur.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (timestamp, session_id)
            )
        
        conn.commit()
        conn.close()

    def get_recent(self, limit: int = 50) -> List[Message]:
        """Получить последние сообщения из общего потока."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, role, content, timestamp FROM messages ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = cur.fetchall()
        conn.close()
        return [Message(row["id"], row["role"], row["content"], row["timestamp"]) 
                for row in reversed(rows)]

    # ── Управление сессиями (новыми диалогами) ──

    def create_session(self, title: Optional[str] = None) -> int:
        """Создать новую сессию чата. Возвращает ID сессии."""
        timestamp = datetime.now().isoformat()
        if title is None:
            title = f"Диалог {timestamp[:16]}"
        
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_sessions (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, timestamp, timestamp)
        )
        session_id = cur.lastrowid
        conn.commit()
        conn.close()
        logger.info(f"Создана новая сессия чата: id={session_id}, title={title}")
        return session_id

    def get_session_messages(self, session_id: int) -> List[Message]:
        """Получить все сообщения конкретной сессии."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, role, content, timestamp FROM session_messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        )
        rows = cur.fetchall()
        conn.close()
        return [Message(row["id"], row["role"], row["content"], row["timestamp"]) 
                for row in rows]

    def get_all_sessions(self) -> List[dict]:
        """Получить список всех сессий."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, created_at, updated_at FROM chat_sessions ORDER BY updated_at DESC"
        )
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
            for row in rows
        ]

    def delete_session(self, session_id: int) -> None:
        """Удалить сессию и все её сообщения."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
        cur.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()
        logger.info(f"Удалена сессия: id={session_id}")

    def update_session_title(self, session_id: int, title: str) -> None:
        """Обновить заголовок сессии."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE chat_sessions SET title = ? WHERE id = ?",
            (title, session_id)
        )
        conn.commit()
        conn.close()


# Глобальный экземпляр
_chat_repo: Optional[ChatRepo] = None


def init_db():
    """Инициализировать базу данных."""
    global _chat_repo
    _chat_repo = ChatRepo()
    logger.info(f"Chat database initialized at {_chat_repo.db_path}")


def get_repo() -> ChatRepo:
    """Получить репозиторий чата."""
    global _chat_repo
    if _chat_repo is None:
        _chat_repo = ChatRepo()
    return _chat_repo


# Алиас для совместимости с existing кодом
chat_repo = get_repo()
