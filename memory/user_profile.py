# -*- coding: utf-8 -*-
"""
Профиль пользователя — хранение и управление данными пользователя.

Хранит:
  - toxic_weight (накопленный вес токсичности);
  - graph — строгий граф знаний о пользователе: pets / people / work.
    Видят ТОЛЬКО агенты (search_check, consolidator) — основная модель
    не получает граф в system prompt, чтобы не засорять контекст.
  - audit_log — последние N операций над графом (для отладки и отката).
    Старые операции уезжают в audit_archive.jsonl.
"""
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


GRAPH_VERSION = 1

# Категории графа, фиксированные в MVP. Потом расширяются инкрементально.
ROOT_CATEGORIES = ("pets", "people", "work")

# Защищённые id: их нельзя удалять, и их parent — root.
PROTECTED_NODE_IDS = frozenset({"root", *ROOT_CATEGORIES})

# Максимум операций в audit_log, остальное уходит в архив.
AUDIT_LOG_MAX = 50


def _now_iso() -> str:
    return datetime.now().isoformat()


def _slugify(text: str, max_len: int = 32) -> str:
    """'Кошка Жужа' -> 'koshka-zhuzha'. Транслитерация кириллицы + ascii."""
    if not text:
        return "node"

    out = []
    for ch in text.lower():
        if ch in _CYRILLIC_TO_LATIN:
            out.append(_CYRILLIC_TO_LATIN[ch])
        elif ch.isascii() and (ch.isalnum() or ch == "-"):
            out.append(ch)
    text = "".join(out)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = text[:max_len].rstrip("-")
    return text or "node"


# Базовая транслитерация кириллицы для slug. Полная таблица ГОСТ 7.79B,
# обрезанная до практически нужных букв; редкие Ґ/Є/І/Ї — опущены для MVP.
_CYRILLIC_TO_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


class UserProfile:
    """Управление профилем пользователя + knowledge graph."""

    def __init__(self, profile_id: str, profiles_dir: Path):
        self.profile_id = profile_id
        self.profiles_dir = profiles_dir
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.profiles_dir / f"{profile_id}.json"
        self.audit_archive_path = self.profiles_dir / f"{profile_id}.audit.jsonl"

        self._data = self._load()
        self._ensure_graph()
        self._save()
        logger.info(
            f"Profile '{profile_id}': toxic_weight={self._data['toxic_weight']}, "
            f"graph_nodes={len(self._data['graph']['nodes'])}"
        )

    # ── загрузка / сохранение ──────────────────────────────────────

    def _load(self) -> dict:
        """Загрузить профиль с диска."""
        if self.file_path.exists():
            try:
                data = json.loads(self.file_path.read_text(encoding="utf-8-sig"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Failed to load profile '{self.profile_id}': {e}")

        return {
            "profile_id": self.profile_id,
            "toxic_weight": 0,
            "toxic_history": [],
            "created_at": _now_iso(),
        }

    def _ensure_graph(self):
        """Инициализировать граф root + категориями, если его нет или он
        устарел. Идемпотентно: повторный вызов ничего не ломает."""
        graph = self._data.get("graph")
        if not isinstance(graph, dict) or graph.get("version") != GRAPH_VERSION:
            graph = {"version": GRAPH_VERSION, "nodes": {}}
            self._data["graph"] = graph

        nodes = graph["nodes"]
        ts = _now_iso()

        if "root" not in nodes:
            nodes["root"] = {
                "id": "root",
                "type": "user",
                "label": self.profile_id,
                "parent": None,
                "attrs": {},
                "sources": [],
                "confidence": 1.0,
                "created_at": ts,
                "updated_at": ts,
            }

        for cat_id in ROOT_CATEGORIES:
            if cat_id not in nodes:
                nodes[cat_id] = {
                    "id": cat_id,
                    "type": "category",
                    "label": {
                        "pets": "Питомцы",
                        "people": "Друзья и родные",
                        "work": "Работа",
                    }[cat_id],
                    "parent": "root",
                    "attrs": {},
                    "sources": [],
                    "confidence": 1.0,
                    "created_at": ts,
                    "updated_at": ts,
                }

        if "audit_log" not in self._data:
            self._data["audit_log"] = []

    def _save(self):
        """Сохранить профиль на диск + ротация audit_log в архив."""
        self._rotate_audit_log()
        try:
            self.file_path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"Failed to save profile '{self.profile_id}': {e}")

    def _rotate_audit_log(self):
        """Если audit_log > AUDIT_LOG_MAX, хвост уезжает в JSONL-архив."""
        log = self._data.get("audit_log", [])
        if len(log) <= AUDIT_LOG_MAX:
            return

        overflow = log[:-AUDIT_LOG_MAX]
        self._data["audit_log"] = log[-AUDIT_LOG_MAX:]

        try:
            with self.audit_archive_path.open("a", encoding="utf-8") as f:
                for entry in overflow:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error(f"Failed to write audit archive: {e}")

    def _reload(self):
        """Перечитать профиль с диска (на случай ручного редактирования)."""
        self._data = self._load()
        self._ensure_graph()

    def _append_audit(self, entry: dict):
        """Добавить запись в audit_log + сразу сохранить на диск."""
        entry.setdefault("timestamp", _now_iso())
        self._data.setdefault("audit_log", []).append(entry)
        self._save()

    # ── toxic weight API (без изменений) ───────────────────────────

    @property
    def toxic_weight(self) -> int:
        self._reload()
        return self._data.get("toxic_weight", 0)

    def add_toxic_weight(self, weight: int, message_preview: str = ""):
        self._data["toxic_weight"] = self._data.get("toxic_weight", 0) + weight
        entry = {
            "weight": weight,
            "total": self._data["toxic_weight"],
            "preview": message_preview[:50],
            "timestamp": _now_iso(),
        }
        history = self._data.get("toxic_history", [])
        history.append(entry)
        self._data["toxic_history"] = history[-20:]
        self._save()
        logger.info(
            f"Profile '{self.profile_id}': +{weight} toxic → "
            f"total={self._data['toxic_weight']}"
        )

    def reset_toxic_weight(self):
        old = self._data.get("toxic_weight", 0)
        self._data["toxic_weight"] = 0
        self._save()
        logger.info(f"Profile '{self.profile_id}': toxic reset {old} → 0")

    def is_blocked(self, threshold: int = 40) -> bool:
        return self.toxic_weight >= threshold

    # ── graph API ──────────────────────────────────────────────────

    @property
    def graph(self) -> dict:
        """Сырой dict графа (read-only по смыслу). Копию для мутаций."""
        self._reload()
        return self._data["graph"]

    def get_node(self, node_id: str) -> Optional[dict]:
        """Узел по id или None. Не поднимает I/O на каждом вызове — предполагаем,
        что граф уже синхронизирован (его читает consolidator из того же треда)."""
        return self._data.get("graph", {}).get("nodes", {}).get(node_id)

    def get_children(self, parent_id: str) -> List[dict]:
        """Прямые дети узла (любого типа)."""
        nodes = self._data.get("graph", {}).get("nodes", {})
        return [n for n in nodes.values() if n.get("parent") == parent_id]

    def get_subtree(self, node_id: str) -> Optional[dict]:
        """Узел + вложенные дети. Возвращает дерево-словарь или None."""
        node = self.get_node(node_id)
        if not node:
            return None
        return {
            "id": node["id"],
            "type": node["type"],
            "label": node.get("label"),
            "attrs": dict(node.get("attrs", {})),
            "children": [
                self.get_subtree(c["id"])
                for c in self.get_children(node_id)
            ],
        }

    def find_by_name(self, name: str, node_type: Optional[str] = None) -> List[str]:
        """Поиск узлов по атрибуту name (case-insensitive substring).
        Опционально фильтрует по type ('pet' | 'person' | 'category' | ...).
        Возвращает список node_id, отсорчанный по updated_at desc."""
        if not name:
            return []
        name_lc = name.lower()
        results = []
        for node in self._data.get("graph", {}).get("nodes", {}).values():
            if node_type and node.get("type") != node_type:
                continue
            attrs = node.get("attrs") or {}
            node_name = attrs.get("name") or node.get("label") or ""
            if name_lc in node_name.lower():
                results.append(node)
        results.sort(key=lambda n: n.get("updated_at", ""), reverse=True)
        return [n["id"] for n in results]

    def add_node(
        self,
        parent: str,
        node_type: str,
        attrs: Optional[dict] = None,
        label: Optional[str] = None,
        sources: Optional[List[str]] = None,
        confidence: float = 1.0,
        msg_id: Optional[str] = None,
    ) -> Optional[str]:
        """Создать новый узел под parent. Возвращает id узла или None при
        ошибке валидации (ошибка пишется в logger + audit_log).

        ID генерируется из type + slug(label/name) + счётчик при коллизии.
        """
        attrs = dict(attrs or {})
        sources = list(sources or [])

        if node_type not in ("pet", "person", "category", "fact"):
            logger.warning(
                f"add_node: refused unknown type '{node_type}' (parent={parent})"
            )
            self._append_audit({
                "op": "add_node",
                "result": "rejected",
                "reason": f"unknown type: {node_type}",
                "msg_id": msg_id,
            })
            return None

        if not self.get_node(parent):
            logger.warning(f"add_node: parent '{parent}' not found")
            self._append_audit({
                "op": "add_node",
                "result": "rejected",
                "reason": f"parent not found: {parent}",
                "msg_id": msg_id,
            })
            return None

        # Категории — прямые дети root, иные parent для них запрещён.
        if node_type == "category" and parent != "root":
            logger.warning(
                f"add_node: category '{parent}' must be child of 'root', "
                f"got parent={parent}"
            )
            self._append_audit({
                "op": "add_node",
                "result": "rejected",
                "reason": f"category under non-root parent: {parent}",
                "msg_id": msg_id,
            })
            return None

        if not 0.0 <= confidence <= 1.0:
            logger.warning(f"add_node: confidence out of range: {confidence}")
            self._append_audit({
                "op": "add_node",
                "result": "rejected",
                "reason": f"confidence out of range: {confidence}",
                "msg_id": msg_id,
            })
            return None

        # Генерация id
        base = attrs.get("name") or label or node_type
        slug = _slugify(base)
        candidate = f"{node_type}_{slug}"
        existing_ids = set(self._data["graph"]["nodes"].keys())
        new_id = candidate
        suffix = 1
        while new_id in existing_ids:
            suffix += 1
            new_id = f"{candidate}_{suffix}"

        ts = _now_iso()
        node = {
            "id": new_id,
            "type": node_type,
            "label": label or attrs.get("name") or base,
            "parent": parent,
            "attrs": attrs,
            "sources": sources,
            "confidence": float(confidence),
            "created_at": ts,
            "updated_at": ts,
        }
        self._data["graph"]["nodes"][new_id] = node
        self._append_audit({
            "op": "add_node",
            "result": "ok",
            "target": new_id,
            "details": {
                "parent": parent,
                "type": node_type,
                "attrs": attrs,
                "confidence": float(confidence),
            },
            "sources": sources,
            "msg_id": msg_id,
        })
        logger.info(f"add_node: created '{new_id}' under '{parent}'")
        return new_id

    def update_attr(
        self,
        node_id: str,
        attr_key: str,
        new_value: str,
        reason: Optional[str] = None,
        sources: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        msg_id: Optional[str] = None,
    ) -> bool:
        """Обновить атрибут узла. Возвращает True при успехе, False при ошибке
        валидации (узел не найден, пустой ключ, и т.п.).
        Защита: root и категории можно обновлять только в label."""
        node = self.get_node(node_id)
        if not node:
            logger.warning(f"update_attr: node '{node_id}' not found")
            self._append_audit({
                "op": "update_attr",
                "result": "rejected",
                "target": node_id,
                "reason": "node not found",
                "msg_id": msg_id,
            })
            return False

        if not attr_key or not isinstance(attr_key, str):
            logger.warning(
                f"update_attr: empty/invalid attr_key for '{node_id}'"
            )
            self._append_audit({
                "op": "update_attr",
                "result": "rejected",
                "target": node_id,
                "reason": "empty attr_key",
                "msg_id": msg_id,
            })
            return False

        if node["type"] in ("user", "category") and attr_key != "label":
            logger.warning(
                f"update_attr: protected node '{node_id}' "
                f"(type={node['type']}) only 'label' is mutable"
            )
            self._append_audit({
                "op": "update_attr",
                "result": "rejected",
                "target": node_id,
                "reason": f"protected node attr: {attr_key}",
                "msg_id": msg_id,
            })
            return False

        old_value = node.get("attrs", {}).get(attr_key)
        if attr_key == "label":
            node["label"] = new_value
        else:
            node.setdefault("attrs", {})[attr_key] = new_value
        node["updated_at"] = _now_iso()
        if sources is not None:
            for s in sources:
                if s and s not in node.get("sources", []):
                    node.setdefault("sources", []).append(s)
        if confidence is not None and 0.0 <= confidence <= 1.0:
            node["confidence"] = float(confidence)

        self._append_audit({
            "op": "update_attr",
            "result": "ok",
            "target": node_id,
            "details": {
                "attr": attr_key,
                "old": old_value,
                "new": new_value,
            },
            "reason": reason,
            "sources": sources,
            "msg_id": msg_id,
        })
        logger.info(
            f"update_attr: '{node_id}.{attr_key}' = {str(new_value)[:60]!r}"
        )
        return True

    def delete_node(
        self,
        node_id: str,
        reason: Optional[str] = None,
        msg_id: Optional[str] = None,
    ) -> bool:
        """Удалить узел. Защита: root и категории удалить нельзя.
        Если есть дети — удаление отклоняется (нужно сначала удалить их
        или использовать каскад через merge)."""
        if node_id in PROTECTED_NODE_IDS:
            logger.warning(f"delete_node: refused to delete protected '{node_id}'")
            self._append_audit({
                "op": "delete_node",
                "result": "rejected",
                "target": node_id,
                "reason": f"protected node ({node_id})",
                "msg_id": msg_id,
            })
            return False

        node = self.get_node(node_id)
        if not node:
            logger.warning(f"delete_node: node '{node_id}' not found")
            self._append_audit({
                "op": "delete_node",
                "result": "rejected",
                "target": node_id,
                "reason": "node not found",
                "msg_id": msg_id,
            })
            return False

        children = self.get_children(node_id)
        if children:
            logger.warning(
                f"delete_node: '{node_id}' has {len(children)} children, "
                f"refusing (use merge or delete children first)"
            )
            self._append_audit({
                "op": "delete_node",
                "result": "rejected",
                "target": node_id,
                "reason": f"has {len(children)} children",
                "msg_id": msg_id,
            })
            return False

        snapshot = {
            "id": node["id"],
            "type": node["type"],
            "label": node.get("label"),
            "attrs": dict(node.get("attrs", {})),
            "parent": node.get("parent"),
        }
        del self._data["graph"]["nodes"][node_id]
        self._append_audit({
            "op": "delete_node",
            "result": "ok",
            "target": node_id,
            "details": snapshot,
            "reason": reason,
            "msg_id": msg_id,
        })
        logger.info(f"delete_node: removed '{node_id}'")
        return True

    def merge_nodes(
        self,
        from_id: str,
        into_id: str,
        reason: Optional[str] = None,
        msg_id: Optional[str] = None,
    ) -> bool:
        """Слить два узла: from удаляется, его attrs и дети переезжают в into.
        Используется для: Юля == Юлия, дважды добавленная одна и та же кошка.
        Если attrs конфликтуют — побеждает into, from-значение идёт в audit."""
        if from_id == into_id:
            logger.warning(f"merge_nodes: from == into ('{from_id}')")
            return False

        if from_id in PROTECTED_NODE_IDS or into_id in PROTECTED_NODE_IDS:
            logger.warning(
                f"merge_nodes: refused to merge protected "
                f"('{from_id}' -> '{into_id}')"
            )
            self._append_audit({
                "op": "merge_nodes",
                "result": "rejected",
                "target": f"{from_id}->{into_id}",
                "reason": "protected node involved",
                "msg_id": msg_id,
            })
            return False

        src = self.get_node(from_id)
        dst = self.get_node(into_id)
        if not src or not dst:
            logger.warning(
                f"merge_nodes: missing node(s) '{from_id}' or '{into_id}'"
            )
            self._append_audit({
                "op": "merge_nodes",
                "result": "rejected",
                "target": f"{from_id}->{into_id}",
                "reason": "node not found",
                "msg_id": msg_id,
            })
            return False

        if src["type"] != dst["type"]:
            logger.warning(
                f"merge_nodes: type mismatch "
                f"'{from_id}'={src['type']} vs '{into_id}'={dst['type']}"
            )
            self._append_audit({
                "op": "merge_nodes",
                "result": "rejected",
                "target": f"{from_id}->{into_id}",
                "reason": "type mismatch",
                "msg_id": msg_id,
            })
            return False

        # Ребёнок src переезжает в dst (если у dst уже есть ребёнок с тем же id
        # — оставляем оба, parent будет корректный).
        for child in self.get_children(from_id):
            if child["id"] in self._data["graph"]["nodes"]:
                self._data["graph"]["nodes"][child["id"]]["parent"] = into_id

        # Attrs: dst (into) побеждает, src-значения идут в audit.
        # Новые ключи из src добавляются, конфликтующие — оставляем dst.
        merged_attrs = dict(dst.get("attrs", {}))
        overwritten = {}
        for k, v in (src.get("attrs") or {}).items():
            if k in merged_attrs:
                if merged_attrs[k] != v:
                    overwritten[k] = {
                        "into": merged_attrs[k],
                        "from": v,
                    }
            else:
                merged_attrs[k] = v
        dst["attrs"] = merged_attrs

        # sources объединяются уникально
        src_sources = list(src.get("sources") or [])
        dst_sources = list(dst.get("sources") or [])
        for s in src_sources:
            if s and s not in dst_sources:
                dst_sources.append(s)
        dst["sources"] = dst_sources

        dst["updated_at"] = _now_iso()

        snapshot = {
            "id": src["id"],
            "type": src["type"],
            "label": src.get("label"),
            "attrs": dict(src.get("attrs", {})),
            "parent": src.get("parent"),
        }
        del self._data["graph"]["nodes"][from_id]

        self._append_audit({
            "op": "merge_nodes",
            "result": "ok",
            "target": f"{from_id}->{into_id}",
            "details": {
                "merged_from": snapshot,
                "overwritten_attrs": overwritten,
            },
            "reason": reason,
            "msg_id": msg_id,
        })
        logger.info(f"merge_nodes: '{from_id}' -> '{into_id}'")
        return True

    def to_prompt_text(self, max_chars: int = 3000) -> str:
        """Сериализовать граф в текст для LLM-промпта (search_check).
        Если превышает max_chars — обрезаем + пометка '...'."""
        nodes = self._data.get("graph", {}).get("nodes", {})

        def render_node(node: dict, depth: int) -> List[str]:
            lines: List[str] = []
            indent = "  " * depth
            if node["type"] == "user":
                lines.append(f"[user] {node.get('label') or self.profile_id}")
            elif node["type"] == "category":
                attrs = node.get("attrs") or {}
                facts = attrs.get("_facts") or []
                lines.append(f"- [category] {node['id']}: {node.get('label')}")
                for f in facts:
                    lines.append(f"{indent}  - {f}")
            else:
                attrs_str = ", ".join(
                    f"{k}={v}" for k, v in (node.get("attrs") or {}).items()
                )
                lines.append(
                    f"{indent}- [{node['type']}] {node['id']}: {attrs_str}"
                )
            for child in self.get_children(node["id"]):
                lines.extend(render_node(child, depth + 1))
            return lines

        if "root" not in nodes:
            return ""

        lines = ["=== ИЗВЕСТНО О ПОЛЬЗОВАТЕЛЕ (граф знаний) ==="]
        lines.extend(render_node(nodes["root"], 0))
        lines.append("===")
        text = "\n".join(lines)

        if len(text) > max_chars:
            text = text[: max_chars - 50] + "\n... (граф обрезан)"
        return text
