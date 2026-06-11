"""Helpers for persisted chat history timestamps and LLM context formatting."""

from datetime import datetime
import re
from typing import Optional


_LEADING_HISTORY_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"\[(?:Когда было сказано|Коли було сказано|When said):[^\]]+\]"
    r"|\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}\]?"
    r")\s*",
    re.IGNORECASE,
)


def new_message(role: str, content: str) -> dict:
    """Build a persisted chat message with a timestamp for future turns."""
    return {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(timespec="minutes"),
    }


def parse_message_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def plural_ru(value: int, one: str, few: str, many: str) -> str:
    value_abs = abs(value)
    if value_abs % 100 in (11, 12, 13, 14):
        return many
    if value_abs % 10 == 1:
        return one
    if value_abs % 10 in (2, 3, 4):
        return few
    return many


def human_elapsed(timestamp: Optional[str], now: Optional[datetime] = None) -> str:
    msg_time = parse_message_timestamp(timestamp)
    if not msg_time:
        return "время сообщения неизвестно"

    now = now or datetime.now()
    delta_seconds = int((now - msg_time).total_seconds())
    if delta_seconds < 0:
        return "из будущего"

    minutes = delta_seconds // 60
    if minutes < 5:
        return "только что"

    days = (now.date() - msg_time.date()).days
    if days == 0:
        hours = minutes // 60
        if hours < 1:
            unit = plural_ru(minutes, "минуту", "минуты", "минут")
            return f"{minutes} {unit} назад"
        unit = plural_ru(hours, "час", "часа", "часов")
        return f"сегодня, {hours} {unit} назад"

    if days == 1:
        return "вчера"
    if days < 7:
        unit = plural_ru(days, "день", "дня", "дней")
        return f"{days} {unit} назад"
    if days < 31:
        weeks = max(1, days // 7)
        unit = plural_ru(weeks, "неделю", "недели", "недель")
        return f"{weeks} {unit} назад"
    if days < 365:
        months = max(1, days // 30)
        unit = plural_ru(months, "месяц", "месяца", "месяцев")
        return f"{months} {unit} назад"

    years = max(1, days // 365)
    unit = plural_ru(years, "год", "года", "лет")
    return f"{years} {unit} назад"


def message_timestamp_label(timestamp: Optional[str]) -> Optional[str]:
    msg_time = parse_message_timestamp(timestamp)
    if not msg_time:
        return None
    return msg_time.strftime("%Y-%m-%d %H:%M")


def format_messages_for_llm(
    messages: list[dict],
    now: Optional[datetime] = None,
    annotate_last: bool = False,
) -> list[dict]:
    """Add absolute message timestamps to chat history passed into the LLM."""
    formatted = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role not in {"user", "assistant"}:
            continue
        timestamp = message_timestamp_label(msg.get("timestamp"))
        if not timestamp:
            formatted.append({"role": role, "content": content})
            continue
        formatted.append({
            "role": role,
            "content": f"{timestamp}\n{content}",
        })
    return formatted


def strip_history_markers(text: str) -> str:
    """Remove leaked history age markers from model-visible/output text."""
    return _LEADING_HISTORY_MARKER_RE.sub("", text or "").strip()
