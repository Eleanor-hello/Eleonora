# -*- coding: utf-8 -*-
"""Достаём JSON-объект из ответа LLM (модели любят добавлять мусор вокруг)."""

import json
from typing import Optional


def extract_json_object(text: str) -> Optional[dict]:
    """Найти и распарсить первый {...} в тексте.

    Переносит строки/пояснения вокруг JSON не мешают.
    Возвращает None если валидного объекта нет.
    """
    if not text:
        return None

    text = text.replace("```json", "```")
    if "`" in text:
        # берём содержимое первого code-fence, если он есть
        parts = text.split("`")
        for part in parts:
            if "{" in part:
                text = part
                break

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None
