# -*- coding: utf-8 -*-
"""
Vision-вызов LLM для финального шага recall'а визуальной памяти.

Когда нашли картинку из visual memory, даём модели "увидеть" её
и дать осмысленный ответ. Требует multimodal модель с vision-поддержкой
(напр. Gemma 4 multi-modal). Text-only модели (gemma-4-E4B) — vision
отключён, вызов возвращает None.

LLM сервер (llama.cpp) поддерживает OpenAI-compatible content blocks:

  messages = [{
    "role": "user",
    "content": [
      {"type": "text", "text": "..."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
  }]
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Optional

import requests

from config import (
    LLM_HOST,
    RESPONSE_MODEL_ID,
    USE_VISION_LLM,
    VISION_MODEL_ID,
    VISION_LLM_TIMEOUT,
)

logger = logging.getLogger(__name__)


def _image_to_data_url(image_path: Path) -> str:
    """Файл → base64 data URL для image_url-блока."""
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/jpeg"
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def describe_recall(
    user_question: str,
    description: str,
    image_path: Path,
    host: str = LLM_HOST,
    model_id: Optional[str] = VISION_MODEL_ID,
    timeout: int = VISION_LLM_TIMEOUT,
) -> Optional[str]:
    """
    Финальный шаг recall: спросить vision-LLM "помнишь Жужу?", показав ей
    найденную в памяти картинку и подпись.

    Args:
        user_question: исходный вопрос ("помнишь как выглядит Жужа?")
        description:   описание из visual memory ("Жужа — кошка Сергея")
        image_path:    путь к найденной картинке

    Returns:
        Текст ответа модели, либо None при ошибке/недоступности.
    """
    if not USE_VISION_LLM:
        logger.debug("Vision LLM disabled (USE_VISION_LLM=False)")
        return None

    if not model_id:
        logger.debug("Vision LLM disabled (VISION_MODEL_ID=None)")
        return None

    if not image_path.exists():
        logger.error(f"Image for vision LLM not found: {image_path}")
        return None

    try:
        data_url = _image_to_data_url(image_path)
    except OSError as e:
        logger.error(f"Failed to read image: {e}")
        return None

    system = (
        "Ты — Элеонора, AI-компаньон. Из визуальной памяти подняли запись и "
        "связанное с ней фото. Ответь пользователю естественно: подтверди, "
        "что помнишь, опиши кратко что видишь на фото и сошлись на подпись "
        "из памяти, если она есть. Не выдумывай деталей сверх того, что "
        "реально видишь."
    )

    user_content = [
        {
            "type": "text",
            "text": (
                f"Вопрос пользователя: {user_question}\n"
                f"Подпись из памяти: {description}\n"
                f"Прикреплено фото из памяти — посмотри на него и ответь."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": data_url},
        },
    ]

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "max_tokens": 600,
    }

    try:
        r = requests.post(
            f"{host.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        message = data["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        if not content:
            logger.warning(f"Vision LLM empty content: keys={list(message.keys())}")
        return content
    except requests.ConnectionError:
        logger.error(f"LLM сервер недоступен ({host})")
    except requests.Timeout:
        logger.error(f"Vision LLM timeout ({timeout}s)")
    except Exception as e:
        logger.error(f"Vision LLM error: {e}")
    return None


def is_vision_available(host: str = LLM_HOST) -> bool:
    """Проверка: vision-модель настроена и сервер доступен."""
    if not USE_VISION_LLM or not VISION_MODEL_ID:
        return False
    try:
        r = requests.get(f"{host.rstrip('/')}/v1/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False
