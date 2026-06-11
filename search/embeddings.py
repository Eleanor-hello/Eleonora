# -*- coding: utf-8 -*-
"""
GigaEmbeddings обёртка — создание эмбеддингов для текста.

Модель: ai-sage/Giga-Embeddings-instruct (2048 dim, русский)
Использует sentence-transformers для загрузки и инференса.
Поддерживает instruction-following формат для повышения точности поиска.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from config import resolve_device

logger = logging.getLogger(__name__)

# Размерность эмбеддингов GigaEmbeddings-instruct
EMBEDDING_DIM = 2048


class Embedder:
    """Обёртка над GigaEmbeddings для создания векторов."""

    def __init__(
        self,
        model_name: str = "ai-sage/Giga-Embeddings-instruct",
        cache_dir: Optional[Path] = None,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or Path("data/models_cache")
        self.device = resolve_device(device)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        logger.info(f"Embedder: model={model_name}, device={self.device}, cache={self.cache_dir}")

    def _load_model(self):
        """Lazy-загрузка модели при первом вызове."""
        if self.model is not None:
            return

        from sentence_transformers import SentenceTransformer

        # Если модель уже в кэше — не лезем в интернет даже для HEAD-проверки
        model_path = self.cache_dir / f"models--{self.model_name.replace('/', '--')}"
        local_only = model_path.exists()

        logger.info(f"Loading {self.model_name}... (local_only={local_only})")
        self.model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True,
            cache_folder=str(self.cache_dir),
            device=self.device,
            local_files_only=local_only,
        )
        logger.info(f"Model loaded (dim={self.model.get_sentence_embedding_dimension()})")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Создать эмбеддинги для списка текстов.

        Args:
            texts: Список строк

        Returns:
            numpy array shape (len(texts), 2048)
        """
        self._load_model()
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed_query(self, query: str, instruction: Optional[str] = None) -> np.ndarray:
        """
        Создать эмбеддинг для поискового запроса.

        GigaEmbeddings-instruct поддерживает формат:
        "Instruct: <инструкция>\nQuery: <запрос>"
        Это повышает точность поиска когда есть hint от сварма
        (user_hint от search_check агента).

        Args:
            query: Текст запроса
            instruction: Инструкция для модели (user_hint от свары)

        Returns:
            numpy array shape (2048,)
        """
        self._load_model()
        if instruction:
            text = f"Instruct: {instruction}\nQuery: {query}"
        else:
            text = query
        return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
