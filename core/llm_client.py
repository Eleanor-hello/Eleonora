# -*- coding: utf-8 -*-
"""Клиент для LLM сервера (llama.cpp) — генерация ответов через chat completions API."""

import logging
import time
from typing import Any, List, Dict, Optional, Union

import requests

logger = logging.getLogger(__name__)


class LLMClient:
    """Минимальный клиент для llama.cpp chat completions."""

    def __init__(
        self,
        host: str = "http://localhost:8080",
        model_id: str = "gemma-4-E4B-it-Q4_K_M.gguf",
        temperature: float = 0.6,
        max_tokens: int = 15000,
        timeout: int = 120,
        max_retries: int = 2,
        retry_delay: float = 2.0,
    ):
        self.host = host.rstrip("/")
        self.url = f"{self.host}/v1/chat/completions"
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"LLMClient: model={model_id}, url={self.url}")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """
        Генерация ответа через LLM сервер.

        Args:
            messages: История диалога. content может быть строкой (text-only)
                      или списком блоков (multimodal: text + image_url).
                      Пример multimodal:
                        {"role": "user", "content": [
                          {"type": "text", "text": "..."},
                          {"type": "image_url", "image_url": {"url": "data:..."}}
                        ]}
            system_prompt: Системный промпт (добавляется первым)
            max_tokens: Переопределение max_tokens для этого вызова

        Returns:
            Текст ответа или None при ошибке.
        """
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model": self.model_id,
            "messages": full_messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code >= 400:
                    error_body = response.text[:500]
                    logger.warning(
                        f"LLM HTTP {response.status_code} (attempt {attempt}/{self.max_retries}): "
                        f"{error_body}"
                    )
                    last_error = f"HTTP {response.status_code}: {error_body}"
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay)
                        continue
                    logger.error(f"LLM: все {self.max_retries} попыток исчерпаны")
                    return None

                data = response.json()
                message = data["choices"][0]["message"]
                content = (message.get("content") or "").strip()
                if not content:
                    reasoning = message.get("reasoning_content") or ""
                    finish_reason = data["choices"][0].get("finish_reason", "?")
                    usage = data.get("usage", {})
                    logger.warning(
                        f"LLM empty content. "
                        f"finish_reason={finish_reason}, "
                        f"usage={usage}, "
                        f"reasoning_chars={len(reasoning)}, "
                        f"message_keys={list(message.keys())}"
                    )
                    if reasoning:
                        logger.warning(
                            f"LLM reasoning preview: {reasoning[:200]!r}"
                        )
                return content

            except requests.ConnectionError:
                logger.error(f"LLM сервер недоступен: {self.url}")
                return None
            except requests.Timeout:
                logger.error(f"Таймаут ({self.timeout}s) при запросе к LLM")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                return None
            except Exception as e:
                logger.error(f"Ошибка LLM: {e}")
                return None

        return None

    def is_available(self) -> bool:
        """Проверка доступности LLM сервера."""
        try:
            r = requests.get(f"{self.host}/v1/models", timeout=5)
            return r.status_code == 200
        except Exception:
            return False
