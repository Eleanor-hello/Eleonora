# -*- coding: utf-8 -*-
"""Консолидация личных фактов о пользователе.

Буферизирует user-сообщения, через LLM извлекает из них СТРОГИЕ личные факты
(имя пользователя, отношения, питомцы, работа, предпочтения...) и сохраняет
их в LanceDB (таблица personal_facts) для векторного поиска.

Контроль качества: извлекаем только явные личные факты, ничего лишнего
(0-3 факта на батч — лучше меньше, да лучше).
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm.parsing import extract_json_object

logger = logging.getLogger(__name__)

CONSOLIDATION_PROMPT = """Ты — система извлечения ЛИЧНЫХ фактов о пользователе для ИИ-компаньона «Элеонора».

ТЕКУЩАЯ ДАТА: {current_date}
ИМЯ ПОЛЬЗОВАТЕЛЯ: {user_name}

══ КРИТИЧЕСКИ ВАЖНО ══
Извлекай ТОЛЬКО факты о личности пользователя — его индивидуальные характеристики,
отношения, предпочтения. НЕ извлекай общие знания, факты о природе/науке/космосе.

══ ЧТО ИЗВЛЕКАТЬ ══
1. ИМЕНА И ОТНОШЕНИЯ: имена близких, родственников, друзей, коллег, питомцев
2. РАБОТА И НАВЫКИ: профессия, место работы, навыки, образование
3. ПРЕДПОЧТЕНИЯ: любимая еда/музыка/фильмы/книги, хобби, увлечения
4. ХАРАКТЕР: черты характера, привычки, особенности поведения
5. ЗДОРОВЬЕ: аллергии, хронические болезни, особенности
6. МЕЧТЫ И ЦЕЛИ: жизненные планы, желания, страхи
7. ВАЖНЫЕ СОБЫТИЯ: переезд, смена работы, свадьба, рождение детей

══ ЧЕГО НЕ ИЗВЛЕКАТЬ ══
- Общие знания («думает что на Венере жарко»)
- Погоду, приветствия, «ок/спасибо», междометия
- Разовая бытовая рутина («купил хлеб»)
- Вопросы пользователя («а что такое чёрная дыра?»)
- Ответы ИИ

══ ПРАВИЛО КАЧЕСТВА ══
Лучше не извлечь вообще ничего, чем извлечь мусор.
Если из сообщений не следует ЯВНЫЙ личный факт — пропусти.

══ ФОРМАТ ОТВЕТА ══
Верни СТРОГО JSON, без markdown, без пояснений:
{{"facts": [{{"text": "У {user_name} есть кошка по имени Жужа", "confidence": 0.95}}]}}

Если извлекать нечего — верни {{"facts": []}}

══ СООБЩЕНИЯ ДЛЯ АНАЛИЗА ══
{messages}"""


class Consolidator:
    """Извлечение строгих личных фактов из user-сообщений в LanceDB."""

    def __init__(
        self,
        llm,
        embedder,
        vector_store,
        user_name: str = "Сергей",
        batch_size: int = 5,
        buffer_file: Optional[Path] = None,
        dedup_threshold: float = 0.15,
    ):
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.user_name = user_name
        self.batch_size = batch_size
        self.dedup_threshold = dedup_threshold
        self._buffer_file = Path(buffer_file) if buffer_file else Path("data/consolidator_buffer.json")
        self.max_parse_retries = 3
        self._parse_errors = 0

        # Буфер пустых/уже извлечённых сообщений
        self._buffer: List[Dict[str, str]] = self._load_buffer()
        logger.info(
            f"Consolidator: user={user_name}, batch={batch_size}, "
            f"buffered={len(self._buffer)}"
        )

    # ── буфер ──

    def _load_buffer(self) -> List[Dict[str, str]]:
        if self._buffer_file.exists():
            try:
                data = json.loads(self._buffer_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except (OSError, json.JSONDecodeError):
                pass
        return []

    def _save_buffer(self) -> None:
        try:
            self._buffer_file.parent.mkdir(parents=True, exist_ok=True)
            self._buffer_file.write_text(
                json.dumps(self._buffer, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as e:
            logger.error(f"Failed to save consolidator buffer: {e}")

    def add_message(self, content: str, msg_id: str = "") -> None:
        """Добавить user-сообщение в буфер; при заполнении — консолидировать."""
        self._buffer.append({
            "content": content,
            "msg_id": msg_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        self._save_buffer()
        if len(self._buffer) >= self.batch_size:
            self.consolidate()

    # ── консолидация ──

    def consolidate(self) -> List[Dict[str, Any]]:
        """Запустить консолидацию буфера. Возвращает сохранённые факты.

        При сбое LLM/парсинга буфер НЕ сбрасывается — сообщения не теряются.
        После max_parse_retries подряд неудач буфер сбрасывается (не держим вечно).
        """
        if not self._buffer:
            return []

        logger.info(f"[consolidator] консолидирую {len(self._buffer)} сообщений...")
        start = time.time()

        messages_text = self._format_messages(self._buffer)
        prompt = CONSOLIDATION_PROMPT.format(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            user_name=self.user_name,
            messages=messages_text,
        )

        try:
            response = self.llm.generate(
                [{"role": "user", "content": "Извлеки личные факты."}],
                system_prompt=prompt,
            )
        except Exception as e:
            logger.error(f"consolidator LLM failed: {e}")
            return []
        if not response:
            logger.error("[consolidator] нет ответа LLM — буфер оставлен на повтор")
            return []

        facts = self._parse_and_save(response)
        if facts is None:
            # парсинг сломался
            self._parse_errors += 1
            if self._parse_errors >= self.max_parse_retries:
                logger.error("[consolidator] парсинг сломался многократно, сбрасываю буфер")
                self._buffer.clear()
                self._parse_errors = 0
                self._save_buffer()
            else:
                logger.warning(
                    f"[consolidator] парсинг не удался "
                    f"({self._parse_errors}/{self.max_parse_retries}); буфер сохранён"
                )
            return []

        self._parse_errors = 0
        self._buffer.clear()
        self._save_buffer()
        logger.info(f"[consolidator] сохранено {len(facts)} фактов за {time.time()-start:.1f}s")
        return facts

    # ── парсинг и сохранение ──

    def _parse_and_save(self, response: str) -> Optional[List[Dict[str, Any]]]:
        parsed = extract_json_object(response)
        if parsed is None:
            return None

        facts_raw = parsed.get("facts", [])
        if not isinstance(facts_raw, list):
            return []

        valid = []
        for f in facts_raw:
            if not isinstance(f, dict):
                continue
            text = (f.get("text") or "").strip()
            conf = f.get("confidence", 1.0)
            if not text or not isinstance(conf, (int, float)) or conf < 0.5:
                continue
            valid.append({"text": text, "confidence": float(conf)})

        if not valid:
            return []

        return self._save_facts(valid)

    def _save_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            vectors = self.embedder.embed_texts([f["text"] for f in facts])
        except Exception as e:
            logger.error(f"consolidator embed failed: {e}")
            return []

        base_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        source_ids = [m.get("msg_id", "") for m in self._buffer if m.get("msg_id")]

        saved = []
        for idx, (fact, vector) in enumerate(zip(facts, vectors)):
            text = fact["text"]
            try:
                if self.vector_store.fact_exists(text, vector, self.dedup_threshold):
                    logger.info(f"[consolidator] дубль: {text[:60]}")
                    continue
            except Exception as e:
                logger.warning(f"consolidator dedup failed: {e}")

            msg_id = f"fact_{base_ts}_{idx}"
            try:
                self.vector_store.add_fact(
                    msg_id=msg_id,
                    fact_text=text,
                    source_msg_ids=source_ids,
                    confidence=fact["confidence"],
                    vector=vector,
                )
                saved.append({
                    "msg_id": msg_id,
                    "fact_text": text,
                    "confidence": fact["confidence"],
                })
                logger.info(f"[consolidator] факт сохранён: {text[:80]}")
            except Exception as e:
                logger.error(f"consolidator add_fact failed: {e}")
        return saved

    # ── формат ──

    def _format_messages(self, messages: List[Dict]) -> str:
        lines = []
        for m in messages:
            mid = f" [{m['msg_id']}]" if m.get("msg_id") else ""
            lines.append(f"[{m.get('timestamp','')}]{mid} {m['content']}")
        return "\n".join(lines)