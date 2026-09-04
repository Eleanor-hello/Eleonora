# -*- coding: utf-8 -*-
"""Чистка текста перед озвучкой.

Убираем то, что Silero читает плохо или странно:
markdown-разметку, блоки кода, URL, эмодзи.
Пунктуацию не трогаем — она нужна для интонации.
"""

import re

_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_CODE_INLINE = re.compile(r"`([^`]+?)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_BOLD_UNDERSCORE = re.compile(r"__(.+?)__", re.DOTALL)
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.DOTALL)
_ITALIC_UNDERSCORE = re.compile(r"(?<![\w_])_(?!_)(.+?)(?<!_)_(?![\w_])", re.DOTALL)
_HEADER = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*]\s+", re.MULTILINE)
_URL = re.compile(r"https?://\S+")
_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # эмодзи и пиктограммы
    "\U00002600-\U000027BF"  # символы и дингбаты
    "\uFE0F\u200D"           # variation selector, zero-width joiner
    "]+",
    flags=re.UNICODE,
)
_WS = re.compile(r"\s+")


def sanitize_for_tts(text: str) -> str:
    """Очистить текст для TTS. Идемпотентна."""
    if not text:
        return ""

    text = _CODE_BLOCK.sub(" ", text)   # блоки кода целиком
    text = _CODE_INLINE.sub(r"\1", text)

    text = _BOLD.sub(r"\1", text)
    text = _BOLD_UNDERSCORE.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    text = _ITALIC_UNDERSCORE.sub(r"\1", text)

    text = _HEADER.sub("", text)
    text = _BULLET.sub("", text)

    text = _URL.sub("ссылка", text)
    text = _EMOJI.sub(" ", text)        # пробел, чтобы слова не склеились

    return _WS.sub(" ", text).strip()
