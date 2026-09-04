# -*- coding: utf-8 -*-
"""Эмбеддинги для поиска по памяти.

Модель: Giga-Embeddings-instruct-480M-0826 (BF16, dim=1024, локальный путь).
Загружается лениво при первом вызове. Поддерживает instruction-формат
"IInstruct: {инструкция}\nQuery: {запрос}" — он повышает точность поиска,
когда у нас есть user_hint от search_check агента.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Обёртка над Giga-Embeddings для создания векторов."""

    def __init__(self, model_path, device: str = "cpu"):
        """
        Args:
            model_path: путь к локальной папке с моделью (BERTained-based, sentence-transformers)
            device: устройство инференса (по умолчанию cpu)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None

    def _load_model(self):
        """Ленивая загрузка модели при первом вызове."""
        if self.model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель эмбеддингов не найдена: {self.model_path}")

        from sentence_transformers import SentenceTransformer

        logger.info(
            f"Embedder: загружаю модель из {self.model_path} (device={self.device})..."
        )
        self.model = SentenceTransformer(
            str(self.model_path),
            trust_remote_code=True,
            device=self.device,
            local_files_only=True,
        )
        dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedder: модель загружена (dim={dim})")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Векторы для списка текстов. Форма (N, 1024), усеченно нормализовано."""
        self._load_model()
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed_query(self, query: str, instruction: Optional[str] = None) -> np.ndarray:
        """Вектор для поискового запроса.

        Если задана инструкция (user_hint от search_check) — используем
        инструктивный формат для лучшего качества поиска.
        """
        self._load_model()
        if instruction:
            text = f"Instruct: {instruction}\nQuery: {query}"
        else:
            text = query
        return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)