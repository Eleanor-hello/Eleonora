# -*- coding: utf-8 -*-
"""Клиент LLM: HTTP-запросы к llama.cpp (llama-server, OpenAI-compatible API)."""

import logging
import time
from typing import List, Dict, Optional, Any

import requests

logger = logging.getLogger(__name__)


class LLMClient:
    """Минимальный клиент chat completions для llama-server (llama.cpp)."""

    def __init__(
        self,
        host: str,
        model_id: str,
        temperature: float = 0.1,
        max_tokens: int = 32000,
        timeout: int = 900,
        max_retries: int = 2,
        retry_delay: float = 2.0,
    ):
        self.url = f"{host.rstrip('/')}/v1/chat/completions"
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """
        Сгенерировать ответ.

        messages: история диалога [{"role": "user"/"assistant", "content": str}]
        system_prompt: необязательный системный промпт

        Возвращает текст ответа или None при неудаче.
        """
        full = []
        if system_prompt:
            full.append({"role": "system", "content": system_prompt})
        full.extend(messages)

        payload = {
            "model": self.model_id,
            "messages": full,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(self.url, json=payload, timeout=self.timeout)
                if resp.status_code >= 400:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                else:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return content.strip() if content else ""
            except (requests.RequestException, KeyError, ValueError) as e:
                last_error = str(e)

            if attempt < self.max_retries:
                logger.warning(
                    f"LLM request failed ({last_error}), retry {attempt}/{self.max_retries}"
                )
                time.sleep(self.retry_delay)

        logger.error(f"LLM request failed: {last_error}")
        return None

    def is_available(self) -> bool:
        """Проверить, отвечает ли llama-server."""
        base = self.url.rsplit("/", 2)[0]  # http://host:port
        try:
            resp = requests.get(f"{base}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        try:
            resp = requests.get(f"{base}/v1/models", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
