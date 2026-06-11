# -*- coding: utf-8 -*-
"""
Обёртка над SigLIP 2 (google/siglip2-large-patch16-256).

Кросс-модальная модель: image-encoder и text-encoder в одном пространстве (1024d).
По дизайну SigLIP контрастно обучен на парах image↔text, поэтому косинус между
эмбеддингами осмысленно сравнивает текст с картинкой.

Используем оба encoder'а:
  embed_image(path)      — для записи в БД и для image→image поиска
  embed_text(description) — для записи (text-on-text поиск надёжнее на именах)
                            и для text→image запросов пользователя

Lazy load: модель тянется при первом вызове, не на импорте.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import numpy as np

from config import SIGLIP_MODEL, SIGLIP_DIM, SIGLIP_DEVICE, MODELS_CACHE_DIR, resolve_device

logger = logging.getLogger(__name__)


class SigLIPEncoder:
    """SigLIP 2 — image + text эмбеддинги в общем пространстве 1024d."""

    def __init__(
        self,
        model_name: str = SIGLIP_MODEL,
        device: str = SIGLIP_DEVICE,
        cache_dir: Path = MODELS_CACHE_DIR,
    ):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._processor = None
        logger.info(f"SigLIPEncoder: model={model_name}, device={self.device}")

    # ─────────────────────────────────────────────────────────────
    # Lazy load
    # ─────────────────────────────────────────────────────────────

    def _load(self):
        if self._model is not None:
            return

        from transformers import AutoModel, AutoProcessor
        import torch

        # Если модель уже в кэше — не лезем в интернет
        model_path = self.cache_dir / f"models--{self.model_name.replace('/', '--')}"
        local_only = model_path.exists()

        logger.info(f"Loading {self.model_name}... (local_only={local_only}, ~1.1GB)")
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            local_files_only=local_only,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            torch_dtype=torch.float32,
            local_files_only=local_only,
        )
        self._model.to(self.device)
        self._model.eval()
        logger.info(f"SigLIP 2 loaded (dim={SIGLIP_DIM})")

    # ─────────────────────────────────────────────────────────────
    # Image
    # ─────────────────────────────────────────────────────────────

    def embed_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Эмбеддинг одной картинки.

        Returns:
            numpy array shape (1024,), L2-нормализован.
        """
        from PIL import Image
        import torch

        self._load()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        with Image.open(path) as img:
            img = img.convert("RGB")
            inputs = self._processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            features = self._model.get_image_features(**inputs)

        vec = features[0].cpu().numpy().astype(np.float32)
        return self._l2_normalize(vec)

    def embed_images(self, paths: List[Union[str, Path]]) -> np.ndarray:
        """Батч-вариант."""
        from PIL import Image
        import torch

        self._load()
        images = []
        for p in paths:
            with Image.open(p) as img:
                images.append(img.convert("RGB"))

        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)

        vecs = features.cpu().numpy().astype(np.float32)
        return np.stack([self._l2_normalize(v) for v in vecs])

    # ─────────────────────────────────────────────────────────────
    # Text
    # ─────────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """
        Эмбеддинг текста в том же 1024d пространстве, что и картинки.

        SigLIP 2 поддерживает многоязычный текст (включая русский).
        """
        import torch

        self._load()
        text = (text or "").strip()
        if not text:
            return np.zeros(SIGLIP_DIM, dtype=np.float32)

        inputs = self._processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            features = self._model.get_text_features(**inputs)

        vec = features[0].cpu().numpy().astype(np.float32)
        return self._l2_normalize(vec)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Батч-вариант."""
        import torch

        self._load()
        cleaned = [(t or "").strip() or " " for t in texts]

        inputs = self._processor(
            text=cleaned,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            features = self._model.get_text_features(**inputs)

        vecs = features.cpu().numpy().astype(np.float32)
        return np.stack([self._l2_normalize(v) for v in vecs])

    # ─────────────────────────────────────────────────────────────
    # Утилиты
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        """L2-нормализация → cosine = dot product."""
        n = np.linalg.norm(v)
        if n == 0:
            return v
        return v / n
