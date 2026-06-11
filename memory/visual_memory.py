# -*- coding: utf-8 -*-
"""
LanceDB-хранилище для visual memory.

Таблица "visual_memories":
  msg_id        — уникальный ID, совпадает с именем файла фото без расширения
                  (vis_YYYYMMDD_HHMMSS_NNN).
  description   — текст-описание ("Жужа — кошка Сергея")
  image_path    — относительный путь к копии фото в visual_photos/
  image_vector  — SigLIP image-embedding (1024 f32)
  text_vector   — SigLIP text-embedding описания (1024 f32)
  created_at    — unix timestamp

Две колонки векторов: LanceDB ищет по одному "vector"-полю за раз.
Одна таблица, два named vector-поля. LanceDB ≥0.5 поддерживает
.search(query, vector_column_name=...).
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import lancedb
import numpy as np
import pyarrow as pa

from config import VISUAL_DB_PATH, VISUAL_PHOTOS_DIR, SIGLIP_DIM, VISUAL_SEARCH_TOP_K, BASE_DIR

logger = logging.getLogger(__name__)


def _gen_msg_id() -> str:
    """vis_YYYYMMDD_HHMMSS_microseconds — гарантированно уникально."""
    from datetime import datetime
    return "vis_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")


class VisualMemory:
    """LanceDB-хранилище визуальных воспоминаний."""

    TABLE_NAME = "visual_memories"

    def __init__(
        self,
        db_path: Path = VISUAL_DB_PATH,
        photos_dir: Path = VISUAL_PHOTOS_DIR,
        dim: int = SIGLIP_DIM,
    ):
        self.db_path = db_path
        self.photos_dir = photos_dir
        self.dim = dim

        self.db_path.mkdir(parents=True, exist_ok=True)
        self.photos_dir.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(self.db_path))
        self.table = self._get_or_create_table()
        logger.info(
            f"VisualMemory: db={self.db_path}, photos={self.photos_dir}, "
            f"records={self.table.count_rows()}"
        )

    def _get_or_create_table(self):
        if self.TABLE_NAME in self.db.table_names():
            return self.db.open_table(self.TABLE_NAME)

        schema = pa.schema([
            pa.field("msg_id", pa.string()),
            pa.field("description", pa.string()),
            pa.field("image_path", pa.string()),
            pa.field("image_vector", pa.list_(pa.float32(), self.dim)),
            pa.field("text_vector", pa.list_(pa.float32(), self.dim)),
            pa.field("created_at", pa.int64()),
        ])
        table = self.db.create_table(self.TABLE_NAME, schema=schema)
        logger.info(f"Created table '{self.TABLE_NAME}' (dim={self.dim})")
        return table

    # ─────────────────────────────────────────────────────────────
    # Запись (memorize)
    # ─────────────────────────────────────────────────────────────

    def add(
        self,
        source_image: Union[str, Path],
        description: str,
        image_vector: np.ndarray,
        text_vector: np.ndarray,
    ) -> dict:
        """
        Сохранить визуальное воспоминание.

        Шаги:
          1. Сгенерить msg_id.
          2. Скопировать source_image в photos_dir/<msg_id>.<ext>.
          3. Положить запись в LanceDB.

        Returns:
            dict с msg_id, image_path, description.
        """
        if image_vector.shape != (self.dim,):
            raise ValueError(f"image_vector shape {image_vector.shape}, expected ({self.dim},)")
        if text_vector.shape != (self.dim,):
            raise ValueError(f"text_vector shape {text_vector.shape}, expected ({self.dim},)")

        src = Path(source_image)
        if not src.exists():
            raise FileNotFoundError(f"Source image not found: {src}")

        msg_id = _gen_msg_id()
        ext = src.suffix.lower() or ".jpg"
        dst = self.photos_dir / f"{msg_id}{ext}"
        shutil.copy2(src, dst)

        # Путь относительно корня проекта
        try:
            rel_path = str(dst.relative_to(BASE_DIR))
        except ValueError:
            rel_path = str(dst)

        record = {
            "msg_id": msg_id,
            "description": description.strip(),
            "image_path": rel_path,
            "image_vector": image_vector.tolist(),
            "text_vector": text_vector.tolist(),
            "created_at": int(time.time()),
        }
        self.table.add([record])
        logger.info(
            f"Saved visual memory [{msg_id}] desc='{description[:60]}' "
            f"file='{dst.name}'"
        )
        return {
            "msg_id": msg_id,
            "image_path": str(dst),
            "rel_image_path": rel_path,
            "description": description.strip(),
        }

    # ─────────────────────────────────────────────────────────────
    # Поиск
    # ─────────────────────────────────────────────────────────────

    def search_by_text(
        self,
        query_vector: np.ndarray,
        top_k: int = VISUAL_SEARCH_TOP_K,
    ) -> List[Tuple[str, str, str, float]]:
        """
        Поиск по тексту: query — text-emb, ищем в text_vector.

        Returns:
            [(msg_id, description, image_path, distance), ...] — меньше = лучше.
        """
        return self._search(query_vector, "text_vector", top_k)

    def search_by_image(
        self,
        query_vector: np.ndarray,
        top_k: int = VISUAL_SEARCH_TOP_K,
    ) -> List[Tuple[str, str, str, float]]:
        """
        Поиск по картинке: query — image-emb, ищем в image_vector.
        """
        return self._search(query_vector, "image_vector", top_k)

    def _search(
        self,
        query_vector: np.ndarray,
        vector_column: str,
        top_k: int,
    ) -> List[Tuple[str, str, str, float]]:
        if self.table.count_rows() == 0:
            return []

        results = (
            self.table.search(query_vector.tolist(), vector_column_name=vector_column)
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )

        out: List[Tuple[str, str, str, float]] = []
        for row in results:
            out.append((
                row["msg_id"],
                row["description"],
                row["image_path"],
                float(row.get("_distance", 0.0)),
            ))
        return out

    # ─────────────────────────────────────────────────────────────
    # Утилиты
    # ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self.table.count_rows()

    def get_by_id(self, msg_id: str) -> Optional[dict]:
        rows = (
            self.table.search()
            .where(f"msg_id = '{msg_id}'")
            .limit(1)
            .to_list()
        )
        return rows[0] if rows else None

    def list_all(self, limit: int = 100) -> List[dict]:
        """Все записи без векторов — для UI/дебага."""
        rows = (
            self.table.search()
            .select(["msg_id", "description", "image_path", "created_at"])
            .limit(limit)
            .to_list()
        )
        return rows
