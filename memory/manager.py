# -*- coding: utf-8 -*-
"""Фасад памяти: объединяет поиск (MemorySearch) и консолидацию фактов.

Единая точка доступа для worker: поиск воспоминаний (recall) и запись личных
фактов в LanceDB (через Consolidator).
"""

import logging
from typing import List, Optional

from config import (
    CONSOLIDATOR_BATCH_SIZE,
    CONSOLIDATOR_BUFFER_FILE,
    CONSOLIDATOR_DEDUP_THRESHOLD,
    CONSOLIDATOR_USER_NAME,
)
from memory.consolidator import Consolidator

logger = logging.getLogger(__name__)


class MemoryManager:
    """Владеет эмбеддером, векторным хранилищем, поиском и консолидатором."""

    def __init__(self, embedder, store, llm):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.consolidator = Consolidator(
            llm=llm,
            embedder=embedder,
            vector_store=store,
            user_name=CONSOLIDATOR_USER_NAME,
            batch_size=CONSOLIDATOR_BATCH_SIZE,
            buffer_file=CONSOLIDATOR_BUFFER_FILE,
            dedup_threshold=CONSOLIDATOR_DEDUP_THRESHOLD,
        )

    # ── индексация истории ──

    def ensure_indexed(self):
        self.store.sync_from_sql(self.embedder)

    # ── консолидация фактов ──

    def note_user_message(self, content: str, msg_id: str = "") -> None:
        """Засчитать user-сообщение в буфер консолидации фактов.

        При заполнении буфера консолидатор сам запустит извлечение фактов
        и запишет их в LanceDB (personal_facts).
        """
        try:
            self.consolidator.add_message(content, msg_id=msg_id)
        except Exception as e:
            logger.warning(f"Consolidation feed failed: {e}")

    def consolidate_now(self) -> List[dict]:
        try:
            return self.consolidator.consolidate()
        except Exception as e:
            logger.warning(f"Consolidation failed: {e}")
            return []

    # ── поиск ──

    def recall(self, user_text: str, user_hint: Optional[str] = None) -> Optional[str]:
        from search.manager import MemorySearch
        ms = MemorySearch(self.embedder, self.store)
        return ms.recall(user_text, user_hint=user_hint)