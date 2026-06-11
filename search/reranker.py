# -*- coding: utf-8 -*-
"""
Jina Reranker v2 — переранжирование результатов поиска.

Cross-encoder видит пару (запрос, документ) целиком,
поэтому точнее bi-encoder (GigaEmbeddings).
Используется для фильтрации top-K результатов из VectorStore.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

from config import resolve_device

logger = logging.getLogger(__name__)


class Reranker:
    """Jina Reranker v2 Base Multilingual."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        device: str = "cpu",
        cache_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.cache_dir = cache_dir or Path("data/models_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        logger.info(f"Reranker: model={model_name}, device={self.device}")

    def _load_model(self):
        """Lazy-загрузка модели."""
        if self.model is not None:
            return

        from transformers import AutoModelForSequenceClassification
        import torch

        # Если модель уже в кэше — не лезем в интернет
        model_path = self.cache_dir / f"models--{self.model_name.replace('/', '--')}"
        local_only = model_path.exists()

        logger.info(f"Loading Jina Reranker... (local_only={local_only})")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir),
            use_flash_attn=False,
            local_files_only=local_only,
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("Reranker loaded")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str, float]],
        top_n: int = 3,
    ) -> List[Tuple[str, str, float]]:
        """
        Переранжировать кандидатов через cross-encoder.

        Args:
            query: Запрос пользователя
            candidates: [(msg_id, text, distance), ...] из VectorStore
            top_n: Сколько лучших вернуть

        Returns:
            [(msg_id, text, reranker_score), ...] — top_n лучших
        """
        if not candidates:
            return []

        if len(candidates) <= top_n:
            return candidates

        self._load_model()

        if self.model is None:
            return candidates[:top_n]

        import torch

        documents = [text for _, text, _ in candidates]

        with torch.no_grad():
            result = self.model.rerank(
                query=query,
                documents=documents,
                max_query_length=256,
                max_length=512,
                top_n=top_n,
            )

        reranked = []
        for item in result:
            if isinstance(item, dict):
                idx = item.get("corpus_id", item.get("index", 0))
                score = item.get("relevance_score", item.get("score", 0.0))
            elif hasattr(item, "corpus_id"):
                idx = item.corpus_id
                score = getattr(item, "relevance_score", getattr(item, "score", 0.0))
            else:
                continue

            msg_id, text, _ = candidates[idx]
            reranked.append((msg_id, text, float(score)))

        return reranked
