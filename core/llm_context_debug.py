"""Debug dumps for the exact LLM context assembled by the app."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional


def dump_llm_context(
    *,
    output_dir: Path,
    label: str,
    model_id: str,
    system_prompt: str,
    messages: list[dict],
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Write a readable txt dump of the payload sent to the response model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"llm_context_{stamp}_{label}.txt"

    lines = [
        "LLM CONTEXT DUMP",
        f"created_at: {datetime.now().isoformat(timespec='seconds')}",
        f"label: {label}",
        f"model_id: {model_id}",
        f"system_chars: {len(system_prompt)}",
        f"messages_count: {len(messages)}",
        "",
    ]

    if extra:
        lines.extend([
            "=== EXTRA ===",
            json.dumps(extra, ensure_ascii=False, indent=2, default=str),
            "",
        ])

    lines.extend([
        "=== SYSTEM PROMPT ===",
        system_prompt,
        "",
        "=== MESSAGES ===",
    ])

    for index, msg in enumerate(messages, start=1):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content_str = json.dumps(content, ensure_ascii=False, indent=2)
            lines.extend([
                "",
                f"--- MESSAGE {index} / role={role} (multimodal) ---",
                content_str,
            ])
        else:
            lines.extend([
                "",
                f"--- MESSAGE {index} / role={role} / chars={len(content)} ---",
                content,
            ])

    lines.extend([
        "",
        "=== API PAYLOAD SHAPE ===",
        json.dumps(
            {
                "model": model_id,
                "messages": [{"role": "system", "content": system_prompt}] + messages,
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
    ])

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    latest = output_dir / "llm_context_latest.txt"
    latest.write_text(text, encoding="utf-8")
    return path
