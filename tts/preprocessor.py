# -*- coding: utf-8 -*-
"""Препроцессор: ёфикация и пользовательские ударения.

Зачем ёфикация:
  Silero с put_yo=False игнорирует правила чтения «е» как «ё».
  Мы отключаем авто-ё в модели и расставляем ё сами — по словарю
  обязательных форм (eyo-kernel safe.txt), без омографов.

Ударения:
  Silero ставит ударения через '+': знак перед ударной гласной
  («молок+о»). Пользовательские ударения лежат в data/stress_overrides.txt,
  по одной словоформе на строку, и применяются поверх всего.

Публичный API:
    yoficate(text) -> str
    apply_stress(text) -> str
    add_stress_override("молок+о") -> None   (пишет в файл)
    stats() -> dict
"""

import re
from pathlib import Path
from typing import Dict, Iterator

_BASE = Path(__file__).parent
DICT_PATH = _BASE / "data" / "yo_safe.txt"
OVERRIDES_PATH = _BASE / "data" / "yo_overrides.txt"
STRESS_OVERRIDES_PATH = _BASE / "data" / "stress_overrides.txt"

# Строка словаря: основа [+ (суффикс1|суффикс2|...)]
_LINE_RE = re.compile(r"^([^(]+)(?:\(([^)]*)\))?\s*$")
_WORD_RE = re.compile(r"[А-Яа-яЁё]+")

_YO_MAP: Dict[str, str] = {}     # ключ: слово lowercase (ё→е); значение: форма с ё
_STRESS_MAP: Dict[str, str] = {} # ключ: слово lowercase; значение: слово с '+'


def _expand(line: str) -> Iterator[str]:
    """Развернуть строку словаря в словоформы."""
    line = line.strip()
    if not line or line.startswith("#"):
        return
    m = _LINE_RE.match(line)
    if not m:
        return
    base, endings = m.group(1), m.group(2)
    if endings is None:
        yield base
    else:
        for suf in endings.split("|"):
            yield base + suf


def _load_yo() -> None:
    with DICT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            for form in _expand(line):
                low = form.lower()
                _YO_MAP[low.replace("ё", "е")] = low
    if OVERRIDES_PATH.is_file():
        # Overrides перекрывают словарь при конфликте
        with OVERRIDES_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                for form in _expand(line):
                    low = form.lower()
                    _YO_MAP[low.replace("ё", "е")] = low


def _load_stress() -> None:
    _STRESS_MAP.clear()
    if not STRESS_OVERRIDES_PATH.is_file():
        return
    with STRESS_OVERRIDES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if not word or word.startswith("#"):
                continue
            bare = word.replace("+", "")
            if bare.isalpha():
                _STRESS_MAP[bare] = word


_load_yo()
_load_stress()


def _match_case(template: str, original: str) -> str:
    """Скопировать регистр original посимвольно на template (длины равны)."""
    if len(template) != len(original):
        return template
    return "".join(
        t.upper() if o.isupper() else t for t, o in zip(template, original)
    )


def _match_case_with_plus(template: str, original: str) -> str:
    """Как _match_case, но template содержит '+' которых нет в original."""
    if len(template) - template.count("+") != len(original):
        return template
    out, i = [], 0
    for ch in template:
        if ch == "+":
            out.append("+")
            continue
        out.append(ch.upper() if original[i].isupper() else ch)
        i += 1
    return "".join(out)


def yoficate(text: str) -> str:
    """Расставить обязательные ё. Регистр исходника сохраняется."""

    def repl(m: re.Match) -> str:
        word = m.group(0)
        template = _YO_MAP.get(word.lower().replace("ё", "е"))
        return word if template is None else _match_case(template, word)

    return _WORD_RE.sub(repl, text)


def apply_stress(text: str) -> str:
    """Расставить '+' перед ударными гласными из пользовательской базы.

    Вызывать ПОСЛЕ yoficate: ключи базы хранятся в ё-форме.
    """

    def repl(m: re.Match) -> str:
        word = m.group(0)
        template = _STRESS_MAP.get(word.lower())
        return word if template is None else _match_case_with_plus(template, word)

    if not _STRESS_MAP:
        return text
    return _WORD_RE.sub(repl, text)


def add_stress_override(marked_word: str) -> None:
    """Добавить/обновить ударение ('молок+о').

    Обновляет карту в памяти и дописывает строку в файл.
    """
    word = marked_word.strip().lower()
    bare = word.replace("+", "")
    if "+" not in word or not bare.isalpha():
        raise ValueError(f"ожидалось слово с '+' перед ударной гласной: {marked_word!r}")
    _STRESS_MAP[bare] = word
    STRESS_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STRESS_OVERRIDES_PATH.open("a", encoding="utf-8") as f:
        f.write(word + "\n")


def stats() -> dict:
    return {
        "yo_forms": len(_YO_MAP),
        "stress_words": len(_STRESS_MAP),
        "stress_file": str(STRESS_OVERRIDES_PATH),
    }
