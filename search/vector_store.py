# -*- coding: utf-8 -*-
"""Векторное хранилище LanceDB для поиска по памяти.

Две таблицы в одной БД:
  - "messages"        — история диалога (user + assistant), индексируется из SQLite
  - "personal_facts"  — строгие личные факты пользователя (Этап 4)

Для messages используется синхронизация из SQLite (db.chat_repo): сообщения,
которых ещё нет в индексе, эмбеддятся и добавляются. msg_id = строка SQL id.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)


class VectorStore:
    """LanceDB-хранилище эмбеддингов."""

    def __init__(self, db_path: Path, embed_dim: int = 1024):
        self.db_path = Path(db_path)
        self.embed_dim = embed_dim
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(self.db_path))
        self.messages_table = self._get_or_create_messages_table()
        self.facts_table = self._get_or_create_facts_table()
        logger.info(
            f"VectorStore: path={self.db_path}, "
            f"messages={self.messages_table.count_rows()}, "
            f"facts={self.facts_table.count_rows()}"
        )

    # ── Таблицы ──

    def _get_or_create_messages_table(self):
        name = "messages"
        if name in self.db.table_names():
            return self.db.open_table(name)
        schema = pa.schema([
            pa.field("msg_id", pa.string()),
            pa.field("role", pa.string()),
            pa.field("content", pa.string()),
            pa.field("timestamp", pa.string()),   # ISO
            pa.field("vector", pa.list_(pa.float32(), self.embed_dim)),
        ])
        table = self.db.create_table(name, schema=schema)
        logger.info(f"Created table '{name}' (dim={self.embed_dim})")
        return table

    def _get_or_create_facts_table(self):
        name = "personal_facts"
        if name in self.db.table_names():
            return self.db.open_table(name)
        schema = pa.schema([
            pa.field("msg_id", pa.string()),
            pa.field("fact_text", pa.string()),
            pa.field("source_msg_ids", pa.string()),  # JSON array
            pa.field("confidence", pa.float32()),
            pa.field("created_at", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), self.embed_dim)),
        ])
        table = self.db.create_table(name, schema=schema)
        logger.info(f"Created table '{name}' (dim={self.embed_dim})")
        return table

    # ── Синхронизация из SQLite ──

    def sync_from_sql(self, embedder) -> int:
        """Добавить в индекс сообщения из SQLite, которых там ещё нет.

        Возвращает число добавленных записей. Пропускает уже
        проиндексированные (по msg_id = строке SQL id).
        """
        from db import chat_repo

        messages = chat_repo.get_recent(limit=100000)  # вся история
        if not messages:
            return 0

        # Какие msg_id уже есть в индексе
        existing = {
            row["msg_id"]
            for row in self.messages_table.search().limit(self.messages_table.count_rows()).to_list()
        } if self.messages_table.count_rows() > 0 else set()

        to_add = []
        for m in messages:
            mid = str(m.id)
            if mid in existing:
                continue
            to_add.append((mid, m.role, m.content, m.created_at.isoformat() if m.created_at else ""))

        if not to_add:
            return 0

        texts = [c for _, _, c, _ in to_add]
        vectors = embedder.embed_texts(texts)

        records = []
        for (mid, role, content, ts), vec in zip(to_add, vectors):
            records.append({
                "msg_id": mid,
                "role": role,
                "content": content,
                "timestamp": ts,
                "vector": vec.tolist(),
            })
        self.messages_table.add(records)
        logger.info(f"VectorStore: добавил {len(records)} сообщений в индекс")
        return len(records)

    # ── Добавление вручную (факты / при необходимости) ──

    def add_messages(self, records: List[dict]):
        if not records:
            return
        data = []
        for r in records:
            vec = r["vector"]
            data.append({
                "msg_id": str(r["msg_id"]),
                "role": r["role"],
                "content": r["content"],
                "timestamp": r.get("timestamp", ""),
                "vector": vec.tolist() if isinstance(vec, np.ndarray) else vec,
            })
        self.messages_table.add(data)

    # ── Поиск ──

    def search_messages(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        role_filter: Optional[str] = None,
    ) -> List[Tuple[str, str, str, float]]:
        """Косинусный поиск по истории. → [(msg_id, role, content, distance), ...]"""
        q = self.messages_table.search(query_vector.tolist()).metric("cosine")
        if role_filter:
            q = q.where(f"role = '{role_filter}'")
        results = q.limit(top_k).to_list()
        return [
            (row["msg_id"], row["role"], row["content"], float(row.get("_distance", 0.0)))
            for row in results
        ]

    def add_fact(
        self,
        msg_id: str,
        fact_text: str,
        source_msg_ids: List[str],
        confidence: float,
        vector: np.ndarray,
        created_at: Optional[int] = None,
    ):
        import json
        ts = int(created_at) if created_at is not None else int(time.time())
        self.facts_table.add([{
            "msg_id": str(msg_id),
            "fact_text": fact_text,
            "source_msg_ids": json.dumps(source_msg_ids, ensure_ascii=False),
            "confidence": float(confidence),
            "created_at": ts,
            "vector": vector.tolist() if isinstance(vector, np.ndarray) else vector,
        }])

    def search_facts(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, str, float, float]]:
        import json
        results = self.facts_table.search(query_vector.tolist()).metric("cosine").limit(top_k).to_list()
        out = []
        for row in results:
            try:
                src = json.loads(row.get("source_msg_ids", "[]"))
            except json.JSONDecodeError:
                src = []
            out.append((row["msg_id"], row["fact_text"], src, float(row.get("confidence", 1.0)),
                        float(row.get("_distance", 0.0))))
        return out

    # ── Утилиты ──

    def count_messages(self) -> int:
        return self.messages_table.count_rows()

    def count_facts(self) -> int:
        return self.facts_table.count_rows()

    def fact_exists(self, fact_text: str, vector: np.ndarray, threshold: float = 0.15) -> bool:
        """Проверить, есть ли похожий факт в БД (дедуп по cosine distance).

        Возвращает True, если ближайший существующий факт ближе threshold.
        """
        if self.facts_table.count_rows() == 0:
            return False
        try:
            neighbors = self.facts_table.search(vector.tolist()).metric("cosine").limit(1).to_list()
        except Exception as e:
            logger.warning(f"Fact dedup check failed: {e}")
            return False
        if not neighbors:
            return False
        dist = float(neighbors[0].get("_distance", 1.0))
        if dist <= threshold:
            logger.info(f"Дубль факта [dist={dist:.4f}]: \"{fact_text[:60]}\"")
            return True
        return False