# -*- coding: utf-8 -*-
"""Фоновая обработка одного хода диалога (GUI не замирает).

Последовательность:
  1. Агент ударений — если пользователь исправил произношение,
     сохраняем слово и отвечаем заготовкой без LLM.
  2. Поисковый агент — если нужен поиск в памяти, ищем воспоминания.
  3. Генерация ответа моделью (с учётом найденных воспоминаний).
"""

import logging
from threading import Lock
from typing import Dict, List, Optional

from PySide6.QtCore import QThread, Signal

from agents.search_check import detect as search_detect
from agents.stress_check import detect_full
from llm.client import LLMClient

logger = logging.getLogger(__name__)

# Ленивый одиночный менеджер памяти (эмбеддинги грузятся при первом вызове).
# Изолируем с помощью Lock, чтобы одновременно работал только один доступ.
_memory_lock = Lock()
_memory_manager = None


def _get_memory_manager(llm=None):
    """Создать (один раз) менеджер памяти (поиск + консолидация фактов)."""
    global _memory_manager
    if _memory_manager is None:
        from config import EMBEDDING_MODEL_PATH, LANCE_DB_PATH
        from memory.manager import MemoryManager
        from search.embeddings import Embedder
        from search.vector_store import VectorStore
        embedder = Embedder(EMBEDDING_MODEL_PATH, device="cpu")
        store = VectorStore(LANCE_DB_PATH)
        _memory_manager = MemoryManager(embedder, store, llm)
    return _memory_manager


class ResponseWorker(QThread):
    """Один ход: проверка ударения + генерация ответа."""

    finished = Signal(str)        # готовый ответ модели
    failed = Signal(str)          # текст ошибки
    learned_stress = Signal(str)  # слово БЕЗ '+', которое выучили

    def __init__(
        self,
        llm: LLMClient,
        agent_llm: LLMClient,
        user_text: str,
        context: List[Dict[str, str]],
        system_prompt: str,
        parent=None,
    ):
        super().__init__(parent)
        self._llm = llm
        self._agent_llm = agent_llm
        self._user_text = user_text
        self._context = context
        self._system_prompt = system_prompt

    def run(self) -> None:
        try:
            # ── 1. Агент ударений ──
            logger.info("[stress_check] проверяю сообщение...")
            marked = detect_full(self._user_text, self._agent_llm)
            if marked:
                logger.info(f"[stress_check] найдено ударение: {marked!r}")
                from tts.preprocessor import add_stress_override
                try:
                    add_stress_override(marked)
                    self.learned_stress.emit(marked.replace("+", ""))
                    return
                except ValueError as e:
                    logger.warning(f"stress override skipped: {e}")
            logger.info("[stress_check] исправлений нет")

            # ── 2. Поисковый агент: нужен ли поиск в памяти ──
            logger.info("[search_check] проверяю, нужен ли поиск в памяти...")
            user_hint = search_detect(self._user_text, self._agent_llm)

            memories = None
            try:
                with _memory_lock:
                    mgr = _get_memory_manager(self._llm)
                    # Фиксируем пользовательское сообщение для консолидации фактов:
                    # при накоплении batch-сообщений факты извлекутся и запишутся в LanceDB.
                    mgr.note_user_message(self._user_text, msg_id="")
            except Exception as e:
                logger.warning(f"[memory] консолидация/feed не удалась: {e}")

            if user_hint:
                logger.info(f"[search] ищу воспоминания по запросу: {user_hint!r}")
                try:
                    with _memory_lock:
                        mgr = _get_memory_manager(self._llm)
                        memories = mgr.recall(self._user_text, user_hint=user_hint)
                    if memories:
                        logger.info(f"[search] найдено воспоминаний: {len(memories.splitlines())} строк")
                    else:
                        logger.info("[search] релевантных воспоминаний не найдено")
                except Exception as e:
                    logger.warning(f"[search] поиск не удался: {e}")

            # ── 3. Ответ модели (с воспоминаниями) ──
            final_system = self._system_prompt
            if memories:
                final_system = (
                    f"{self._system_prompt}\n\n"
                    f"ВОСПОМИНАНИЯ (личные данные пользователя, используй если релевантно):\n{memories}"
                )
            logger.info(f"[llm] генерирую ответ ({len(self._context)} сообщений в контексте)...")
            result = self._llm.generate(
                self._context, system_prompt=final_system
            )
            if result:
                logger.info(f"[llm] ответ готов ({len(result)} символов)")
                self.finished.emit(result)
            else:
                self.failed.emit("llama-server не ответил. Проверь, что сервер запущен.")
        except Exception as e:
            logger.exception("ResponseWorker упал")
            self.failed.emit(f"Ошибка обработки: {e}")
