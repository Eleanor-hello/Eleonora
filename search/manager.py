# -*- coding: utf-8 -*-
"""Поиск по памяти: связывает эмбеддер + векторное хранилище.

Используется поисковым агентом (agents/search_check.py): когда тот выдает
user_hint, вызываем recall() и получаем блок «ВОСПОМИНАНИЯ» для подстановки
в контекст диалога.
"""

import logging
from typing import Optional

import numpy as np

from config import (
    EMBEDDING_INSTRUCTION,
    SEARCH_MIN_SCORE,
    SEARCH_TOP_K,
)

logger = logging.getLogger(__name__)


class MemorySearch:
    """Поиск по личным сообщениям и фактам пользователя."""

    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store
        self._initialized = False

    def ensure_indexed(self):
        """Один раз синхронизировать историю из SQLite в индекс."""
        if self._initialized:
            return
        try:
            self.store.sync_from_sql(self.embedder)
        except Exception as e:
            logger.warning(f"MemorySearch: sync index failed: {e}")
        self._initialized = True

    def recall(self, user_text: str, user_hint: Optional[str] = None,
               top_k: int = SEARCH_TOP_K, min_score: float = SEARCH_MIN_SCORE) -> Optional[str]:
        """Поиск релевантных воспоминаний. → строка блока «ВОСПОМИНАНИЯ» или None."""
        self.ensure_indexed()
        if self.store.count_messages() == 0 and self.store.count_facts() == 0:
            return None

        instruction = user_hint if user_hint else EMBEDDING_INSTRUCTION
        query_vector = self.embedder.embed_query(user_text, instruction=instruction)

        # Ищем и в сообщениях, и в фактах
        facts = self.store.search_facts(query_vector, top_k=top_k)
        messages = self.store.search_messages(query_vector, top_k=top_k)

        # Факты: distance -> similarity; порог
        fact_lines = []
        for _, text, _, conf, dist in facts:
            sim = 1.0 - dist
            if sim >= min_score:
                fact_lines.append(text)

        # Сообщения: берём только user-сообщения (история от Элеоноры не нужна)
        msg_lines = []
        seen_texts = set(fact_lines)
        for _, role, content, dist in messages:
            sim = 1.0 - dist
            if sim < min_score:
                continue
            if role == "assistant":
                continue
            if content in seen_texts:
                continue
            seen_texts.add(content)
            msg_lines.append(content)

        if not fact_lines and not msg_lines:
            return None

        parts = []
        if fact_lines:
            parts.append("Факты о пользователе:\n" + "\n".join(f"- {t}" for t in fact_lines))
        if msg_lines:
            parts.append("Релевантные сообщения из истории:\n" + "\n".join(f"- {t}" for t in msg_lines))
        return "\n\n".join(parts)