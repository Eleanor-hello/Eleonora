# -*- coding: utf-8 -*-
"""
LanceDB обёртка — хранение и поиск векторов.

Две таблицы в одной БД:
  - "messages"         — полная история диалога (user + assistant)
  - "personal_facts"   — строгие личные факты пользователя

Схема messages:
  msg_id       — уникальный ID (msg_YYYYMMDD_HHMMSS_microsec)
  role         — "user" | "assistant"
  content      — текст сообщения
  timestamp    — ISO строка времени
  session_id   — ID сессии (для группировки)
  vector       — эмбеддинг (float32 × 2048)

Схема personal_facts:
  msg_id           — уникальный ID (fact_YYYYMMDD_HHMMSS_microsec)
  fact_text        — текст факта (в третьем лице от имени пользователя)
  source_msg_ids   — JSON-массив msg_id исходных сообщений
  confidence       — float 0.0-1.0 уверенность консолидатора
  created_at       — unix timestamp
  vector           — эмбеддинг (float32 × 2048)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)

# Размерность GigaEmbeddings-instruct
EMBEDDING_DIM = 2048


class VectorStore:
    """LanceDB-хранилище для эмбеддингов."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
    ):
        """
        Args:
            db_path: Путь к директории LanceDB (по умолчанию data/lance_db)
        """
        self.db_path = db_path or Path("data/lance_db")
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(self.db_path))
        self.messages_table = self._get_or_create_messages_table()
        self.facts_table = self._get_or_create_facts_table()

        logger.info(
            f"VectorStore: path={self.db_path}, "
            f"messages={self.messages_table.count_rows()}, "
            f"facts={self.facts_table.count_rows()}"
        )

    # ─────────────────────────────────────────────────────────────
    # Таблицы
    # ─────────────────────────────────────────────────────────────

    def _get_or_create_messages_table(self):
        table_name = "messages"
        if table_name in self.db.table_names():
            return self.db.open_table(table_name)

        schema = pa.schema([
            pa.field("msg_id", pa.string()),
            pa.field("role", pa.string()),          # "user" | "assistant"
            pa.field("content", pa.string()),
            pa.field("timestamp", pa.string()),     # ISO format
            pa.field("session_id", pa.string()),    # для группировки сессий
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
        ])
        table = self.db.create_table(table_name, schema=schema)
        logger.info(f"Created table '{table_name}' (dim={EMBEDDING_DIM})")
        return table

    def _get_or_create_facts_table(self):
        table_name = "personal_facts"
        if table_name in self.db.table_names():
            return self.db.open_table(table_name)

        schema = pa.schema([
            pa.field("msg_id", pa.string()),
            pa.field("fact_text", pa.string()),
            pa.field("source_msg_ids", pa.string()),  # JSON array
            pa.field("confidence", pa.float32()),
            pa.field("created_at", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
        ])
        table = self.db.create_table(table_name, schema=schema)
        logger.info(f"Created table '{table_name}' (dim={EMBEDDING_DIM})")
        return table

    # ─────────────────────────────────────────────────────────────
    # Messages API
    # ─────────────────────────────────────────────────────────────

    def add_message(
        self,
        msg_id: str,
        role: str,
        content: str,
        timestamp: str,
        session_id: str,
        vector: np.ndarray,
    ):
        """Добавить сообщение в историю."""
        self.messages_table.add([{
            "msg_id": str(msg_id),
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "session_id": session_id,
            "vector": vector.tolist() if isinstance(vector, np.ndarray) else vector,
        }])

    def add_messages_batch(self, records: List[dict]):
        """Пакетное добавление сообщений."""
        if not records:
            return
        data = []
        for r in records:
            data.append({
                "msg_id": str(r["msg_id"]),
                "role": r["role"],
                "content": r["content"],
                "timestamp": r["timestamp"],
                "session_id": r.get("session_id", "default"),
                "vector": r["vector"].tolist() if isinstance(r["vector"], np.ndarray) else r["vector"],
            })
        self.messages_table.add(data)
        logger.info(f"Added {len(data)} messages to history")

    def search_messages(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        role_filter: Optional[str] = None,
        session_filter: Optional[str] = None,
    ) -> List[Tuple[str, str, str, float]]:
        """
        Поиск в истории диалога.

        Returns:
            [(msg_id, role, content, distance), ...]
        """
        query = self.messages_table.search(query_vector.tolist()).metric("cosine")

        where_parts = []
        if role_filter:
            where_parts.append(f"role = '{role_filter}'")
        if session_filter:
            where_parts.append(f"session_id = '{session_filter}'")
        if where_parts:
            query = query.where(" AND ".join(where_parts))

        results = query.limit(top_k).to_list()

        output = []
        for row in results:
            output.append((
                row["msg_id"],
                row["role"],
                row["content"],
                float(row.get("_distance", 0.0)),
            ))
        return output

    # ─────────────────────────────────────────────────────────────
    # Personal Facts API
    # ─────────────────────────────────────────────────────────────

    def add_fact(
        self,
        msg_id: str,
        fact_text: str,
        source_msg_ids: List[str],
        confidence: float,
        vector: np.ndarray,
        created_at: Optional[int] = None,
    ):
        """Добавить личный факт."""
        ts = int(created_at) if created_at is not None else int(time.time())
        self.facts_table.add([{
            "msg_id": str(msg_id),
            "fact_text": fact_text,
            "source_msg_ids": json.dumps(source_msg_ids, ensure_ascii=False),
            "confidence": float(confidence),
            "created_at": ts,
            "vector": vector.tolist() if isinstance(vector, np.ndarray) else vector,
        }])

    def add_facts_batch(self, records: List[dict]):
        """Пакетное добавление фактов."""
        if not records:
            return
        now = int(time.time())
        data = []
        for r in records:
            data.append({
                "msg_id": str(r["msg_id"]),
                "fact_text": r["fact_text"],
                "source_msg_ids": json.dumps(r.get("source_msg_ids", []), ensure_ascii=False),
                "confidence": float(r.get("confidence", 1.0)),
                "created_at": int(r.get("created_at", now)),
                "vector": r["vector"].tolist() if isinstance(r["vector"], np.ndarray) else r["vector"],
            })
        self.facts_table.add(data)
        logger.info(f"Added {len(data)} facts to personal_facts")

    def search_facts(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, str, List[str], float, float]]:
        """
        Поиск личных фактов.

        Returns:
            [(msg_id, fact_text, source_msg_ids, confidence, distance), ...]
        """
        query = self.facts_table.search(query_vector.tolist()).metric("cosine")
        results = query.limit(top_k).to_list()

        output = []
        for row in results:
            try:
                source_ids = json.loads(row.get("source_msg_ids", "[]"))
            except json.JSONDecodeError:
                source_ids = []
            output.append((
                row["msg_id"],
                row["fact_text"],
                source_ids,
                float(row.get("confidence", 1.0)),
                float(row.get("_distance", 0.0)),
            ))
        return output

    def fact_exists(self, fact_text: str, vector: np.ndarray, threshold: float = 0.15) -> bool:
        """Проверить есть ли похожий факт в БД (батч-дедуп)."""
        if self.facts_table.count_rows() == 0:
            return False
        try:
            neighbors = self.facts_table.search(vector.tolist()).metric("cosine").limit(1).to_list()
        except Exception as e:
            logger.warning(f"Fact dedup check failed: {e}")
            return False

        if not neighbors:
            return False
        _, existing_text, distance = neighbors[0]["msg_id"], neighbors[0]["fact_text"], float(neighbors[0].get("_distance", 1.0))
        if distance <= threshold:
            logger.info(f"Skip duplicate fact [db, dist={distance:.4f}]: \"{fact_text[:60]}\" ≈ \"{existing_text[:60]}\"")
            return True
        return False

    # ─────────────────────────────────────────────────────────────
    # Утилиты
    # ─────────────────────────────────────────────────────────────

    def count_messages(self) -> int:
        return self.messages_table.count_rows()

    def count_facts(self) -> int:
        return self.facts_table.count_rows()

    def iter_facts(self) -> List[Dict[str, Any]]:
        """Вернуть список всех фактов из personal_facts (без векторов — для миграций).

        Возвращает [{msg_id, fact_text, source_msg_ids, confidence, created_at}, ...].
        """
        try:
            rows = self.facts_table.to_pandas().to_dict("records")
        except Exception as e:
            logger.error(f"iter_facts: failed to read table: {e}")
            return []
        out = []
        for row in rows:
            try:
                sources = json.loads(row.get("source_msg_ids", "[]"))
            except (json.JSONDecodeError, TypeError):
                sources = []
            out.append({
                "msg_id": row.get("msg_id", ""),
                "fact_text": row.get("fact_text", ""),
                "source_msg_ids": sources,
                "confidence": float(row.get("confidence", 1.0)),
                "created_at": row.get("created_at", 0),
            })
        return out

    def delete_fact(self, msg_id: str):
        self.facts_table.delete(f"msg_id = '{msg_id}'")

    def delete_message(self, msg_id: str):
        self.messages_table.delete(f"msg_id = '{msg_id}'")