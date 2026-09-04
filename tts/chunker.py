# -*- coding: utf-8 -*-
"""Разрезка длинного текста на чанки для Silero.

Зачем: в Silero v5.5 захардкожен лимит позиционного кодирования (~5000
символов после фонетизации) — длинный текст валяет модель. Режем на куски
и синтезируем по очереди (движок стримит их в воспроизведение).

Уровни разрезки — от естественного к грубому:
  1) по предложениям (. ! ? …), знак остаётся в конце чанка;
  2) длинное предложение — по клаузам (, ; : —);
  3) остаток — по словам, сверхдлинные слова режутся посимвольно.
"""

import re
from typing import List

_SENT_END = re.compile(r'(?<=[.!?…])["»”\')\]]?\s+')
_CLAUSE_SEP = re.compile(r"[,;:—–]\s+")


def chunk_for_tts(text: str, max_chars: int = 800) -> List[str]:
    """Разрезать текст на чанки, каждый <= max_chars. Пустой вход → []."""
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")

    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    return _pack(_split_sentences(text), max_chars)


def _split_sentences(text: str) -> List[str]:
    return [p for p in _SENT_END.split(text) if p.strip()]


def _split_part(part: str, max_chars: int) -> List[str]:
    """Разрезать одну слишком длинную часть: клаузы → слова."""
    clauses = [c for c in _CLAUSE_SEP.split(part) if c.strip()]
    if len(clauses) <= 1:
        return _split_by_words(part, max_chars)
    return _pack(clauses, max_chars)


def _split_by_words(text: str, max_chars: int) -> List[str]:
    """Крайний случай: набираем слова до лимита, монстров режем посимвольно."""
    chunks: List[str] = []
    buf = ""
    for word in text.split():
        if len(word) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(word[i:i + max_chars] for i in range(0, len(word), max_chars))
            continue
        candidate = f"{buf} {word}".strip() if buf else word
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            chunks.append(buf)
            buf = word
    if buf:
        chunks.append(buf)
    return chunks


def _pack(parts: List[str], max_chars: int) -> List[str]:
    """Собрать части в чанки: короткие склеиваем, длинные режем глубже."""
    chunks, buf = [], ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_split_part(part, max_chars))
            continue
        candidate = f"{buf} {part}".strip() if buf else part
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            chunks.append(buf)
            buf = part
    if buf:
        chunks.append(buf)
    return chunks
