# -*- coding: utf-8 -*-
"""
EventScheduler — планировщик отложенных триггеров-событий для инициативных
реплик Элеоноры.

Жизненный цикл события:
  1. Пользователь пишет "поставил картошку, через 20 минут будет готова".
  2. SwarmClassifier (time_check) формирует готовый prompt-инжект и delay.
  3. ChatApp вызывает schedule(source_text, prompt, delay) — эмбеддинг
     исходного сообщения сохраняется вместе с записью.
  4. При каждом новом сообщении пользователя ChatApp вызывает
     check_overlap(new_text):
       - если есть семантически близкий активный триггер и новое сообщение
         тоже time-event → replace (старое удаляется, новое пишется)
       - если совпадение есть, но новое сообщение НЕ time-event
         ("мама уже приехала") → cancel (событие исчерпано)
  5. По истечении delay срабатывает fire_callback(event_id, prompt).
     Элеонора генерирует инициативную реплику, запись удаляется.
  6. При закрытии приложения tkinter-таймеры умирают сами.
     При следующем запуске просроченные события (scheduled_at <= now)
     удаляются без триггера — пользователя уже не спросишь "как картошка".

Персистентность: data/waiting_events.json, формат — список dict'ов с полями
  id, scheduled_at (ISO), created_at (ISO), source_text, prompt, embedding.
"""

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class EventScheduler:
    """Планировщик отложенных триггеров с персистом и overlap-детекцией."""

    def __init__(
        self,
        storage_path: Path,
        embedder: Any,
        tk_root: Any,
        fire_callback: Callable[[str, str], None],
        overlap_threshold: float = 0.2,
        grey_zone_threshold: float = 0.5,
    ):
        """
        Args:
            storage_path: путь к JSON-файлу (data/waiting_events.json).
            embedder: Embedder с .embed_texts([str]) -> np.ndarray (N, dim).
            tk_root: корневой tkinter-виджет (для .after / .after_cancel).
            fire_callback: вызывается в main-thread при срабатывании таймера.
                           Сигнатура (event_id: str, prompt: str) -> None.
            overlap_threshold: cosine-distance порог для жёсткого совпадения.
                               Hard hit если cosine_similarity >= 1 - overlap_threshold.
                               По умолчанию 0.2 ⇒ sim >= 0.8.
            grey_zone_threshold: нижняя граница серой зоны (cosine similarity).
                                 Если sim в [grey_zone_threshold; 1 - overlap_threshold)
                                 — возвращаем kind="grey", решение принимает
                                 вызывающий код асинхронно (LLM-арбитр). Ниже
                                 этого порога — no hit.

        Note:
            Арбитр серой зоны вызывается снаружи (в main.py через executor),
            чтобы не блокировать поток обработки основного ответа. check_overlap
            только возвращает кандидата и тип совпадения.
        """
        self.storage_path = Path(storage_path)
        self.embedder = embedder
        self.tk_root = tk_root
        self.fire_callback = fire_callback
        self.overlap_threshold = overlap_threshold
        self.grey_zone_threshold = grey_zone_threshold

        self._events: List[Dict[str, Any]] = []
        self._timer_handles: Dict[str, Any] = {}
        self._lock = threading.RLock()

    # ── персистентность ──

    def _save(self) -> None:
        """Сохранить текущий список событий в JSON (под локом)."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(
                json.dumps(self._events, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"EventScheduler: failed to save: {e}")

    def load_and_cleanup(self) -> int:
        """
        Загрузить события из JSON, выкинуть просроченные (scheduled_at <= now).
        Возвращает количество оставшихся активных событий.
        Таймеры не стартует — это отдельный вызов start_all_timers().
        """
        with self._lock:
            self._events = []
            if not self.storage_path.exists():
                return 0

            try:
                raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"EventScheduler: failed to load: {e}")
                return 0

            if not isinstance(raw, list):
                logger.warning("EventScheduler: JSON root is not a list, ignoring")
                return 0

            now = datetime.now()
            expired = 0
            for evt in raw:
                try:
                    sched_at = datetime.fromisoformat(evt["scheduled_at"])
                except (KeyError, ValueError):
                    expired += 1
                    continue
                if sched_at <= now:
                    expired += 1
                    continue
                self._events.append(evt)

            if expired:
                logger.info(f"EventScheduler: dropped {expired} expired events")
                self._save()

            return len(self._events)

    def start_all_timers(self) -> None:
        """
        Запустить tkinter-таймеры для всех активных событий.
        Вызывается из main-thread после load_and_cleanup().
        """
        with self._lock:
            now = datetime.now()
            for evt in list(self._events):
                sched_at = datetime.fromisoformat(evt["scheduled_at"])
                delay_ms = max(0, int((sched_at - now).total_seconds() * 1000))
                self._arm_timer(evt["id"], delay_ms)

    # ── внутреннее ──

    def _arm_timer(self, event_id: str, delay_ms: int) -> None:
        """Поставить tkinter-таймер на срабатывание через delay_ms."""
        handle = self.tk_root.after(delay_ms, self._on_fire, event_id)
        self._timer_handles[event_id] = handle

    def _on_fire(self, event_id: str) -> None:
        """Вызывается tkinter в main-thread при срабатывании."""
        with self._lock:
            evt = next((e for e in self._events if e["id"] == event_id), None)
            if evt is None:
                return
            prompt = evt["prompt"]
            self._events = [e for e in self._events if e["id"] != event_id]
            self._timer_handles.pop(event_id, None)
            self._save()

        logger.info(f"EventScheduler: firing {event_id}: {prompt[:80]}")
        try:
            self.fire_callback(event_id, prompt)
        except Exception as e:
            logger.error(f"EventScheduler: fire_callback failed: {e}")

    def _cancel_timer(self, event_id: str) -> None:
        """Отменить tkinter-таймер если есть."""
        handle = self._timer_handles.pop(event_id, None)
        if handle is not None:
            try:
                self.tk_root.after_cancel(handle)
            except Exception:
                pass

    # ── публичный API ──

    def schedule(
        self, source_text: str, prompt: str, delay_minutes: int
    ) -> Optional[str]:
        """
        Зарегистрировать новое событие.

        Args:
            source_text: исходное user-сообщение, породившее триггер
                         (используется как ключ для overlap-детекции).
            prompt: готовая фраза-инжект, которую получит Элеонора при срабатывании.
            delay_minutes: через сколько минут триггерить (1 .. 1440).

        Returns:
            event_id или None если не удалось посчитать эмбеддинг.
        """
        if delay_minutes <= 0:
            return None

        try:
            embedding = self.embedder.embed_texts([source_text])[0]
        except Exception as e:
            logger.error(f"EventScheduler: embedding failed: {e}")
            return None

        now = datetime.now()
        scheduled_at = now + _td_minutes(delay_minutes)
        event_id = f"evt_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        record = {
            "id": event_id,
            "created_at": now.isoformat(timespec="seconds"),
            "scheduled_at": scheduled_at.isoformat(timespec="seconds"),
            "source_text": source_text,
            "prompt": prompt,
            "embedding": [float(x) for x in embedding],
        }

        with self._lock:
            self._events.append(record)
            self._save()

        delay_ms = int(delay_minutes * 60 * 1000)
        self.tk_root.after(0, self._arm_timer, event_id, delay_ms)

        logger.info(
            f"EventScheduler: scheduled {event_id}, "
            f"fires at {scheduled_at.strftime('%H:%M')} "
            f"(+{delay_minutes}m): {prompt[:80]}"
        )
        return event_id

    def check_overlap(
        self, text: str
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Найти активный триггер, семантически близкий к переданному тексту.

        Returns:
            (kind, event_id, source_text) где kind ∈ {"hard", "grey", "none"}.
            event_id и source_text == None только для "none".

            - "hard": sim >= 1 - overlap_threshold. Решение уже принято,
                     вызывающий код делает replace/cancel сразу.
            - "grey": grey_zone_threshold <= sim < hard_floor. Вызывающий
                     код ДОЛЖЕН вызвать LLM-арбитра на (source_text, text);
                     арбитр зовётся снаружи асинхронно, чтобы не тормозить
                     основной ответ.
            - "none": sim < grey_zone_threshold или нет активных триггеров.

        Возвращается только лучший кандидат, чтобы арбитра звали один раз
        за сообщение, а не по числу активных событий.
        """
        with self._lock:
            events_snapshot = list(self._events)

        if not events_snapshot:
            logger.info("EventScheduler: overlap check skipped — 0 active events")
            return ("none", None, None)

        try:
            query_vec = self.embedder.embed_texts([text])[0]
        except Exception as e:
            logger.error(f"EventScheduler: overlap embedding failed: {e}")
            return ("none", None, None)

        hard_floor = 1.0 - self.overlap_threshold
        best_evt: Optional[Dict[str, Any]] = None
        best_sim = -1.0
        per_event = []

        for evt in events_snapshot:
            try:
                ev_vec = np.asarray(evt["embedding"], dtype=np.float32)
            except (KeyError, TypeError):
                continue
            sim = _cosine_similarity(query_vec, ev_vec)
            per_event.append((evt["id"], sim))
            if sim > best_sim:
                best_sim = sim
                best_evt = evt

        details = ", ".join(f"{eid}:{s:.3f}" for eid, s in per_event)
        base_log = (
            f"EventScheduler: overlap over {len(events_snapshot)} events "
            f"(hard={hard_floor:.2f}, grey={self.grey_zone_threshold:.2f}, "
            f"best={best_sim:.3f})"
        )

        if best_evt is None:
            logger.info(f"{base_log} -> no hit (no valid embeddings); per-event: {details}")
            return ("none", None, None)

        if best_sim >= hard_floor:
            logger.info(f"{base_log} -> HARD hit {best_evt['id']}; per-event: {details}")
            return ("hard", best_evt["id"], best_evt.get("source_text"))

        if best_sim >= self.grey_zone_threshold:
            logger.info(
                f"{base_log} -> grey zone candidate {best_evt['id']} "
                f"(arbiter decides async); per-event: {details}"
            )
            return ("grey", best_evt["id"], best_evt.get("source_text"))

        logger.info(f"{base_log} -> no hit (below grey); per-event: {details}")
        return ("none", None, None)

    def cancel(self, event_id: str) -> bool:
        """
        Удалить событие, отменить таймер. Возвращает True если было что удалить.
        """
        with self._lock:
            before = len(self._events)
            self._events = [e for e in self._events if e["id"] != event_id]
            removed = before != len(self._events)
            if removed:
                self._save()
        if removed:
            self.tk_root.after(0, self._cancel_timer, event_id)
            logger.info(f"EventScheduler: cancelled {event_id}")
        return removed

    def replace(
        self, old_event_id: str, source_text: str, prompt: str, delay_minutes: int
    ) -> Optional[str]:
        """
        Атомарная замена: удалить старое событие, создать новое с новым delay.
        """
        self.cancel(old_event_id)
        return self.schedule(source_text, prompt, delay_minutes)

    def list_active(self) -> List[Dict[str, Any]]:
        """Снимок активных событий (без эмбеддингов, для отладки/UI)."""
        with self._lock:
            return [
                {k: v for k, v in e.items() if k != "embedding"}
                for e in self._events
            ]


def _td_minutes(minutes: int):
    """datetime.timedelta(minutes=X) без импорта timedelta в main-стек."""
    from datetime import timedelta
    return timedelta(minutes=minutes)
