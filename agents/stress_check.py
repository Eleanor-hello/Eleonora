# -*- coding: utf-8 -*-
"""Агент ударений: ловит от пользователя исправления произношения.

Примеры, которые понимает:
  «элеон+ора»                          — слово уже с '+'
  «говори не Элеонора, а ЭлеонОра»     — заглавная гласная = ударение
  «ударение в слове молоко на третью о» — нужен LLM (fuzzy-формулировка)

Возвращает слово с '+' перед ударной гласной («элеон+ора») или None.
Слово затем уходит в tts.preprocessor.add_stress_override().
"""

import logging
import re
from typing import Optional

from llm.parsing import extract_json_object

logger = logging.getLogger(__name__)

VOWELS = "аеёиоуыэюя"

# Быстрый путь 1: в сообщении уже есть размеченное слово («молок+о»)
_MARKED_RE = re.compile(r"\b([а-яё]+\+[а-яё]+)\b")

# Быстрый путь 2: «не СЛОВО1, а слОво2» — заглавная гласная внутри слова 2
_NOT_A_RE = re.compile(
    r"[Нн]е\s+([а-яёА-ЯЁ]+)\s*,?\s*а\s+(?:правильно\s+)?([а-яёА-ЯЁ]+)"
)

# Маркеры того, что сообщение вообще про произношение (для LLM-fallback)
_KEYWORDS = ("ударени", "произнос", "произнес", "перепроизнос", "говор", "озвуч")

STRESS_PROMPT = """Пользователь учит ИИ правильному ударению в русском слове.
Извлеки слово, ударную букву и её номер в слове.

OUTPUT: EXACTLY ONE LINE JSON.
Если юзер назвал И слово И куда падает ударение:
{{"word": "<слово lowercase>", "stress_letter": "<одна русская гласная>", "occurrence": N}}
occurrence — номер вхождения буквы в слове слева (1 = первая).
Если буква одна в слове — occurrence=1.
Если букв несколько и непонятно какая — occurrence=0.

Если пользователь НЕ исправляет ударение:
{{"word": null}}

Examples:
IN: "ударение в слове молоко на третью о"
OUT: {{"word": "молоко", "stress_letter": "о", "occurrence": 3}}

IN: "в слове звОнит правильно, а не звонит"
OUT: {{"word": "звонит", "stress_letter": "о", "occurrence": 1}}

IN: "Расскажи про молоко"
OUT: {{"word": null}}"""


def build_marked_word(word: str, letter: str, occurrence: int) -> Optional[str]:
    """«молоко» + («о», 3) → «молок+о». Неверные данные → None."""
    word = word.strip().lower()
    letter = letter.strip().lower()
    if not word.isalpha() or letter not in VOWELS or occurrence < 1:
        return None
    positions = [i for i, ch in enumerate(word) if ch == letter]
    if occurrence > len(positions):
        return None
    pos = positions[occurrence - 1]
    return word[:pos] + "+" + word[pos:]


def detect(text: str) -> Optional[str]:
    """Быстрые паттерны без LLM. Мгновенно, вызывать всегда."""
    m = _MARKED_RE.search(text)
    if m:
        return m.group(1).lower()

    m = _NOT_A_RE.search(text)
    if m:
        wrong, right = m.group(1), m.group(2)
        # Заглавная гласная НЕ на первой позиции указывает ударение
        # (первая позиция — просто имя собственное).
        accents = [
            i for i, ch in enumerate(right[1:], start=1) if ch in VOWELS.upper()
        ]
        if len(right) >= 3 and len(accents) == 1:
            idx = accents[0]
            low = right.lower()
            return low[:idx] + "+" + low[idx:]
    return None


def looks_like_stress_request(text: str) -> bool:
    """Похоже ли сообщение на обучение ударению (грубый фильтр для LLM)."""
    low = text.lower()
    return any(k in low for k in _KEYWORDS)


def detect_via_llm(text: str, llm) -> Optional[str]:
    """LLM-fallback для нестандартных формулировок. llm — LLMClient."""
    try:
        raw = llm.generate([{"role": "user", "content": text}], STRESS_PROMPT)
        data = extract_json_object(raw or "")
        if not data or not data.get("word"):
            return None
        marked = build_marked_word(
            data.get("word", ""), data.get("stress_letter", ""),
            int(data.get("occurrence", 0)),
        )
        return marked
    except Exception as e:
        logger.warning(f"stress_check LLM fallback failed: {e}")
        return None


def detect_full(text: str, llm=None) -> Optional[str]:
    """Полная проверка: быстрые паттерны → при подозрении LLM."""
    fast = detect(text)
    if fast:
        return fast
    if llm is not None and looks_like_stress_request(text):
        result = detect_via_llm(text, llm)
        if result:
            logger.info(f"stress_check: LLM извлёк {result!r}")
        return result
    return None
