# -*- coding: utf-8 -*-
"""
Консолидатор памяти — управление графом знаний о пользователе.

Два режима работы:
  - С user_profile (новый): LLM возвращает операции add_node / update_attr /
    delete_node / merge_nodes для графа знаний. Факты в personal_facts
    генерируются как побочный продукт операций (для векторного поиска).
  - Без user_profile (legacy): LLM возвращает плоский список фактов, как
    раньше. Сохранён для backwards-compat и для миграции.

Аудит всех изменений графа ведётся в user_profile.audit_log.

Контроль качества: ~0-3 операций на 100 сообщений (очень строгий отбор).
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from memory.user_profile import UserProfile

logger = logging.getLogger(__name__)


# ── Промпт для режима с user_profile (операции над графом) ─────────

CONSOLIDATION_PROMPT = """Ты — система управления ГРАФОМ ЗНАНИЙ о пользователе для AI-компаньона "Элеонора".

Граф знаний — это дерево узлов с типизированными связями parent → child.
Узлы могут быть:
  - "user"   (только один, id="root")
  - "category" (pets, people, work — фиксированные категории)
  - "pet"     (питомцы: кошка, собака и т.п.)
  - "person"  (друзья, родные, коллеги)
  - "fact"    (произвольные факты-узлы)

═══ ДАННЫЕ ═══
ТЕКУЩАЯ ДАТА: {current_date}
ИМЯ ПОЛЬЗОВАТЕЛЯ: {user_name}

═══ ТЕКУЩИЙ ГРАФ (source of truth) ═══
{graph}

═══ НОВЫЕ СООБЩЕНИЯ ПОЛЬЗОВАТЕЛЯ ═══
{messages}

═══ ЗАДАЧА ═══
Проанализируй новые сообщения и верни СПИСОК ОПЕРАЦИЙ для обновления графа.
Цель — отразить в графе ВСЁ новое, что можно извлечь как устойчивый факт о личности.

═══ ДОСТУПНЫЕ ОПЕРАЦИИ ═══

1) add_node — добавить новый узел
   {{
     "op": "add_node",
     "parent": "pets" | "people" | "work" | "<existing_node_id>",
     "type": "pet" | "person" | "fact",
     "label": "Краткое имя узла (опционально)",
     "attrs": {{"name": "Жужа", "species": "кошка", "appearance": "пушистая"}},
     "sources": ["msg_id_1"],
     "confidence": 0.0-1.0,
     "msg_id": "msg_id_на_котором_оперируем"
   }}
   - parent: для pet → "pets", для person → "people", для fact в work → "work".
   - attrs.name обязательно для pet/person (используется для генерации id).
   - sources: список msg_id из новых сообщений, на которых основан факт.

2) update_attr — обновить/исправить атрибут существующего узла
   {{
     "op": "update_attr",
     "node_id": "pet_zhuzha",
     "attr_key": "name",
     "new_value": "Жужа",
     "reason": "исправление: ранее было 'Лужа'",
     "sources": ["msg_id"],
     "confidence": 0.95,
     "msg_id": "msg_id"
   }}
   - ВАЖНО: используй это для ИСПРАВЛЕНИЙ ("не Лужа, а Жужа", "работаю теперь в Google").
   - Для смены локации/статуса — это тоже update_attr, не delete_node.

3) delete_node — удалить узел целиком
   {{
     "op": "delete_node",
     "node_id": "pet_olddog",
     "reason": "...",
     "msg_id": "msg_id"
   }}
   - ТОЛЬКО для явных "у меня больше нет собаки", "Юля — бывшая, не общаемся".
   - НЕ удаляй root или category (это запрещено, операция будет отклонена).

4) merge_nodes — слить два узла (для дубликатов)
   {{
     "op": "merge_nodes",
     "from_id": "person_yuliya",
     "into_id": "person_yulya",
     "reason": "одна и та же Юля, разные формы имени",
     "msg_id": "msg_id"
   }}
   - В "into_id" попадают все attrs from_id (без потерь).
   - from_id удаляется.
   - Если "Юлия" и "Юля" — один человек (а не разные люди) — используй merge.

═══ ПРАВИЛА ОТБОРА ═══
Извлекай ТОЛЬКО устойчивые личные факты. НЕ извлекай:
  - Общие знания ("на Венере жарко", "чёрные дыры — это...")
  - Бытовую рутину ("купил хлеб", "почистил зубы")
  - Погоду, разовые ощущения ("устал", "голоден")
  - Вопросы пользователя
  - Приветствия, междометия
  - Ответы AI (только user-сообщения)

Лучше вернуть {{"operations": []}} чем шуметь.

═══ ФОРМАТ ОТВЕТА ═══
ВЕРНИ СТРОГО JSON (без markdown, без пояснений):
{{
  "operations": [
    {{"op": "add_node", "parent": "pets", "type": "pet",
     "attrs": {{"name": "Жужа", "species": "кошка"}}, ...}},
    ...
  ]
}}

Если извлекать нечего → {{"operations": []}}

═══ ПРИМЕРЫ ═══

ВХОД (новые сообщения):
  "У меня есть кошка Жужа, она пушистая"
  "Я работаю программистом на Python, делаю бэкенд на FastAPI"
ВЫХОД:
{{
  "operations": [
    {{"op": "add_node", "parent": "pets", "type": "pet",
     "attrs": {{"name": "Жужа", "species": "кошка", "appearance": "пушистая"}},
     "sources": ["msg_1"], "confidence": 0.95, "msg_id": "msg_1"}},
    {{"op": "add_node", "parent": "work", "type": "fact",
     "attrs": {{"text": "{user_name} работает программистом на Python, специализируется на бэкенде FastAPI"}},
     "sources": ["msg_2"], "confidence": 0.95, "msg_id": "msg_2"}}
  ]
}}

ВХОД (новые сообщения):
  "Не Лужа а Жужа, извини"
  (В графе уже есть pet_luzha)
ВЫХОД:
{{
  "operations": [
    {{"op": "update_attr", "node_id": "pet_luzha", "attr_key": "name",
     "new_value": "Жужа", "reason": "исправление: пользователь уточнил имя",
     "sources": ["msg_3"], "confidence": 0.95, "msg_id": "msg_3"}}
  ]
}}

ВХОД (новые сообщения):
  "У меня больше нет собаки, отдал бабушке"
  (В графе есть pet_dog_bim)
ВЫХОД:
{{
  "operations": [
    {{"op": "delete_node", "node_id": "pet_dog_bim",
     "reason": "пользователь избавился от собаки",
     "msg_id": "msg_4"}}
  ]
}}

ВХОД (новые сообщения):
  "Подругу Юлию теперь зовут Юлей, это одно лицо"
ВЫХОД:
{{
  "operations": [
    {{"op": "merge_nodes", "from_id": "person_yuliya", "into_id": "person_yulya",
     "reason": "одна и та же подруга",
     "msg_id": "msg_5"}}
  ]
}}
"""


# ── Legacy-промпт (плоские факты, без графа) ───────────────────────

LEGACY_CONSOLIDATION_PROMPT = """Ты — система извлечения ЛИЧНЫХ фактов о пользователе для AI-компаньона "Элеонора".

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
- "думает что на Венере очень жарко" (это общее знание, не личный факт)
- "сегодня хорошая погода" (погода, не личный факт)
- "смотрел фильм" без контекста отношения к нему
- Вопросы пользователя ("а что такое чёрная дыра?")
- Ответы AI
- Приветствия, "ок/спасибо", междометия
- Разовая бытовая рутина ("купил хлеб", "почистил зубы")
- Физические ощущения без смысла ("устал", "замёрз", "голоден")

══ ПРАВИЛО КАЧЕСТВА ══
Лучше не извлечь вообще ничего, чем извлечь мусор.
Если из сообщений не следует ЯВНЫЙ личный факт — пропусти.
Ожидание: 0-3 факта на 100 сообщений.

══ ФОРМАТ ОТВЕТА ══
Верни СТРОГО JSON, без markdown, без пояснений:
{{
  "facts": [
    {{
      "text": "У {user_name} есть кошка по имени Жужа",
      "confidence": 0.95
    }}
  ]
}}

Если извлекать нечего — верни {{ "facts": [] }}

══ СООБЩЕНИЯ ДЛЯ АНАЛИЗА ══
{messages}"""


class Consolidator:
    """
    Консолидатор памяти с поддержкой knowledge graph.

    Args:
        user_profile: UserProfile для режима с графом. Если None — legacy
                      режим с плоскими фактами.
    """

    def __init__(
        self,
        llm_client,
        embedder,
        vector_store,
        user_profile: Optional["UserProfile"] = None,
        user_name: str = "Сергей",
        batch_size: int = 100,
        buffer_file: Optional[Path] = None,
        dedup_threshold: float = 0.15,
    ):
        self.llm = llm_client
        self.embedder = embedder
        self.vector_store = vector_store
        self.user_profile = user_profile
        self.user_name = user_name
        self.batch_size = batch_size
        self.dedup_threshold = dedup_threshold
        self._buffer_file = buffer_file or Path("data/consolidator_buffer.json")

        self._buffer: List[Dict[str, str]] = self._load_buffer()
        self._messages_since_last = len(self._buffer)

        mode = "graph" if user_profile else "legacy"
        logger.info(
            f"Consolidator v3 (mode={mode}): user={user_name}, batch={batch_size}, "
            f"dedup={dedup_threshold}, buffered={self._messages_since_last}"
        )
        if user_profile is None:
            logger.warning(
                "Consolidator: user_profile not provided, falling back to "
                "legacy flat-facts mode. Knowledge graph will NOT be updated."
            )

    # ── буфер сообщений (без изменений) ────────────────────────────

    def _load_buffer(self) -> List[Dict[str, str]]:
        if self._buffer_file.exists():
            try:
                data = json.loads(self._buffer_file.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    logger.info(f"Restored consolidator buffer: {len(data)} messages")
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Failed to load consolidator buffer: {e}")
        return []

    def _save_buffer(self):
        try:
            self._buffer_file.parent.mkdir(parents=True, exist_ok=True)
            self._buffer_file.write_text(
                json.dumps(self._buffer, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"Failed to save consolidator buffer: {e}")

    def add_message(self, role: str, content: str, msg_id: str = ""):
        if role == "user":
            self._buffer.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "msg_id": msg_id,
            })
            self._messages_since_last += 1
            self._save_buffer()

            if self._messages_since_last >= self.batch_size:
                self.consolidate()

    # ── точка входа: consolidate ────────────────────────────────────

    def consolidate(self) -> List[Dict[str, Any]]:
        """Запустить консолидацию буфера. Возвращает список сохранённых фактов."""
        if not self._buffer:
            return []

        logger.info(f"Consolidating {len(self._buffer)} messages...")
        start = time.time()

        messages_text = self._format_messages(self._buffer)
        current_date = datetime.now().strftime("%Y-%m-%d")

        if self.user_profile is not None:
            prompt = CONSOLIDATION_PROMPT.format(
                current_date=current_date,
                user_name=self.user_name,
                graph=self.user_profile.to_prompt_text(max_chars=2000),
                messages=messages_text,
            )
        else:
            prompt = LEGACY_CONSOLIDATION_PROMPT.format(
                current_date=current_date,
                user_name=self.user_name,
                messages=messages_text,
            )

        response = self.llm.generate(
            messages=[{"role": "user", "content": "Обнови граф знаний."}],
            system_prompt=prompt,
        )

        if not response:
            logger.error("Consolidation failed: no LLM response")
            return []

        if self.user_profile is not None:
            saved = self._consolidate_via_graph(response)
        else:
            saved = self._consolidate_via_facts(response)

        elapsed = time.time() - start
        logger.info(f"Consolidated: {len(saved)} saved in {elapsed:.1f}s")

        self._buffer.clear()
        self._messages_since_last = 0
        self._save_buffer()
        return saved

    def force_consolidate(self):
        if self._buffer:
            self.consolidate()

    # ── режим с графом (новый) ──────────────────────────────────────

    def _consolidate_via_graph(self, response: str) -> List[Dict[str, Any]]:
        """Распарсить операции, применить через user_profile, сохранить
        производные факты в personal_facts для векторного поиска."""
        ops = self._parse_operations(response)
        if not ops:
            return []

        applied_ops: List[Dict[str, Any]] = []
        fact_records: List[Dict[str, Any]] = []

        for op in ops:
            result, fact_text = self._apply_op(op)
            if result == "applied" and fact_text:
                fact_records.append({
                    "text": fact_text,
                    "op": op.get("op"),
                    "msg_id": op.get("msg_id"),
                    "sources": op.get("sources") or [],
                    "confidence": op.get("confidence", 1.0),
                })
            applied_ops.append({"op": op, "result": result})

        saved = self._save_facts_to_vector(fact_records)

        rejected = sum(1 for o in applied_ops if o["result"] != "applied")
        logger.info(
            f"Graph ops: {len(applied_ops)} total, "
            f"{len(applied_ops) - rejected} applied, "
            f"{rejected} rejected, "
            f"{len(saved)} facts saved to vector store"
        )
        return saved

    def _apply_op(self, op: Dict[str, Any]) -> tuple[str, Optional[str]]:
        """Применить одну операцию к user_profile. Возвращает (result, fact_text).
        result ∈ {'applied', 'rejected', 'noop'}."""
        op_name = op.get("op")
        if op_name not in ("add_node", "update_attr", "delete_node", "merge_nodes"):
            logger.warning(f"Unknown op: {op_name!r}")
            return ("rejected", None)

        if op_name == "add_node":
            parent = op.get("parent", "")
            node_type = op.get("type", "")
            attrs = op.get("attrs") or {}
            label = op.get("label")
            sources = op.get("sources") or []
            confidence = float(op.get("confidence", 1.0))
            msg_id = op.get("msg_id")

            if not parent or not node_type:
                logger.warning(f"add_node: missing parent/type: {op}")
                return ("rejected", None)

            new_id = self.user_profile.add_node(
                parent=parent,
                node_type=node_type,
                attrs=attrs,
                label=label,
                sources=[s for s in sources if s],
                confidence=confidence,
                msg_id=msg_id,
            )
            if not new_id:
                return ("rejected", None)
            fact_text = self._fact_text_for_node(new_id, op.get("type"), attrs, label)
            return ("applied", fact_text)

        if op_name == "update_attr":
            node_id = op.get("node_id", "")
            attr_key = op.get("attr_key", "")
            new_value = op.get("new_value", "")
            if not node_id or not attr_key:
                logger.warning(f"update_attr: missing node_id/attr_key: {op}")
                return ("rejected", None)
            ok = self.user_profile.update_attr(
                node_id=node_id,
                attr_key=attr_key,
                new_value=str(new_value),
                reason=op.get("reason"),
                sources=op.get("sources") or [],
                confidence=op.get("confidence"),
                msg_id=op.get("msg_id"),
            )
            if not ok:
                return ("rejected", None)
            node = self.user_profile.get_node(node_id)
            if not node:
                return ("applied", None)
            fact_text = self._fact_text_for_update(node, attr_key, str(new_value))
            return ("applied", fact_text)

        if op_name == "delete_node":
            node_id = op.get("node_id", "")
            if not node_id:
                logger.warning(f"delete_node: missing node_id: {op}")
                return ("rejected", None)
            ok = self.user_profile.delete_node(
                node_id=node_id,
                reason=op.get("reason"),
                msg_id=op.get("msg_id"),
            )
            return ("applied" if ok else "rejected", None)

        if op_name == "merge_nodes":
            from_id = op.get("from_id", "")
            into_id = op.get("into_id", "")
            if not from_id or not into_id:
                logger.warning(f"merge_nodes: missing from/into: {op}")
                return ("rejected", None)
            ok = self.user_profile.merge_nodes(
                from_id=from_id,
                into_id=into_id,
                reason=op.get("reason"),
                msg_id=op.get("msg_id"),
            )
            return ("applied" if ok else "rejected", None)

        return ("noop", None)

    @staticmethod
    def _fact_text_for_node(
        node_id: str, node_type: str, attrs: dict, label: Optional[str]
    ) -> str:
        """Сгенерировать человекочитаемую строку факта для personal_facts."""
        attrs = attrs or {}
        name = attrs.get("name") or label or node_id
        if node_type == "pet":
            species = attrs.get("species", "питомец")
            extra = ", ".join(
                f"{k}: {v}" for k, v in attrs.items()
                if k not in ("name", "species")
            )
            base = f"питомец ({species}) по имени {name}"
            return f"{base}; {extra}" if extra else base
        if node_type == "person":
            relation = attrs.get("relation", "")
            extra = ", ".join(
                f"{k}: {v}" for k, v in attrs.items()
                if k not in ("name", "relation")
            )
            base = f"{relation} {name}".strip() if relation else name
            return f"{base}; {extra}" if extra else base
        if node_type == "fact":
            return attrs.get("text") or label or node_id
        return label or node_id

    @staticmethod
    def _fact_text_for_update(node: dict, attr_key: str, new_value: str) -> str:
        """Факт-строка для update_attr: текущее состояние атрибута + контекст узла."""
        node_id = node.get("id", "")
        attrs = node.get("attrs") or {}
        label = attrs.get("name") or node.get("label") or node_id
        if attr_key == "name":
            return f"имя '{label}' актуально как '{new_value}'"
        return f"у {label}: {attr_key} = {new_value}"

    # ── парсер операций ─────────────────────────────────────────────

    def _parse_operations(self, response: str) -> List[Dict[str, Any]]:
        """Достать JSON, вытащить operations[]. Невалидные элементы пропускаем."""
        text = (response or "").strip()

        if text.startswith("```"):
            lines = text.split("\n")
            inside = False
            json_lines = []
            for line in lines:
                s = line.strip()
                if s.startswith("```") and not inside:
                    inside = True
                    continue
                elif s == "```" and inside:
                    break
                elif inside:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            logger.warning(f"No JSON object in operations response: {text[:100]}")
            return []

        try:
            parsed = json.loads(text[start:end])
        except json.JSONDecodeError as e:
            logger.error(f"Operations JSON parse error: {e}\nText: {text[:200]}")
            return []

        if not isinstance(parsed, dict):
            return []

        ops_raw = parsed.get("operations", [])
        if not isinstance(ops_raw, list):
            return []

        valid: List[Dict[str, Any]] = []
        for op in ops_raw:
            if not isinstance(op, dict):
                continue
            op_name = op.get("op")
            if op_name not in ("add_node", "update_attr", "delete_node", "merge_nodes"):
                logger.warning(f"Skipping unknown op: {op_name!r}")
                continue
            valid.append(op)
        return valid

    def _save_facts_to_vector(self, fact_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Эмбеддим + сохраняем в personal_facts (с дедупом)."""
        if not fact_records:
            return []

        texts = [r["text"] for r in fact_records]
        try:
            vectors = self.embedder.embed_texts(texts)
        except Exception as e:
            logger.error(f"Batch embed failed: {e}")
            return []

        base_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        saved = []
        for idx, (rec, vector) in enumerate(zip(fact_records, vectors)):
            text = rec["text"]
            if self.vector_store.fact_exists(text, vector, self.dedup_threshold):
                logger.info(f"Skip duplicate fact [{idx}]: {text[:60]}")
                continue
            msg_id = f"fact_{base_ts}_{idx}"
            try:
                self.vector_store.add_fact(
                    msg_id=msg_id,
                    fact_text=text,
                    source_msg_ids=[s for s in (rec.get("sources") or []) if s],
                    confidence=float(rec.get("confidence", 1.0)),
                    vector=vector,
                )
                saved.append({
                    "msg_id": msg_id,
                    "fact_text": text,
                    "confidence": float(rec.get("confidence", 1.0)),
                    "source_msg_ids": rec.get("sources") or [],
                })
                logger.info(f"Saved fact: {text[:80]}")
            except Exception as e:
                logger.error(f"Failed to save fact: {e}")
        return saved

    # ── legacy: плоские факты ───────────────────────────────────────

    def _consolidate_via_facts(self, response: str) -> List[Dict[str, Any]]:
        facts = self._parse_legacy_facts(response)
        return self._save_facts(facts)

    def _parse_legacy_facts(self, response: str) -> List[Dict[str, Any]]:
        text = (response or "").strip()
        if text.startswith("```"):
            lines = text.split("\n")
            inside = False
            json_lines = []
            for line in lines:
                if line.strip().startswith("```") and not inside:
                    inside = True
                    continue
                elif line.strip() == "```" and inside:
                    break
                elif inside:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return []
        try:
            parsed = json.loads(text[start:end])
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, dict):
            return []

        facts_raw = parsed.get("facts", [])
        if not isinstance(facts_raw, list):
            return []

        valid = []
        for f in facts_raw:
            if not isinstance(f, dict):
                continue
            ft = (f.get("text") or "").strip()
            fc = f.get("confidence", 1.0)
            if not ft or not isinstance(fc, (int, float)) or fc < 0.5:
                continue
            valid.append({"text": ft, "confidence": float(fc)})
        return valid

    def _save_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not facts:
            return []

        all_texts = [f["text"] for f in facts]
        try:
            vectors = self.embedder.embed_texts(all_texts)
        except Exception as e:
            logger.error(f"Batch embed failed: {e}")
            return []

        base_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        source_ids = [m.get("msg_id", "") for m in self._buffer if m.get("msg_id")]

        saved = []
        for idx, (fact, vector) in enumerate(zip(facts, vectors)):
            text = fact["text"]
            confidence = fact["confidence"]
            if self.vector_store.fact_exists(text, vector, self.dedup_threshold):
                logger.info(f"Skip duplicate fact [{idx}]: {text[:60]}")
                continue
            msg_id = f"fact_{base_ts}_{idx}"
            try:
                self.vector_store.add_fact(
                    msg_id=msg_id,
                    fact_text=text,
                    source_msg_ids=source_ids,
                    confidence=confidence,
                    vector=vector,
                )
                saved.append({
                    "msg_id": msg_id,
                    "fact_text": text,
                    "confidence": confidence,
                    "source_msg_ids": source_ids,
                })
                logger.info(f"Saved fact: {text[:80]}")
            except Exception as e:
                logger.error(f"Failed to save fact: {e}")
        return saved

    # ── формат сообщений для промпта ────────────────────────────────

    def _format_messages(self, messages: List[Dict]) -> str:
        lines = []
        for msg in messages:
            ts = msg.get("timestamp", "")
            mid = msg.get("msg_id", "")
            mid_str = f" [{mid}]" if mid else ""
            lines.append(f"[{ts}]{mid_str} {msg['content']}")
        return "\n".join(lines)
