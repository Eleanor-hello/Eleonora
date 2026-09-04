# -*- coding: utf-8 -*-
"""Eleonora v3 — конфигурация. Все настройки проекта только здесь."""

import os
from pathlib import Path

# ── Пути ──
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ── LLM сервер (llama.cpp llama-server) ──
LLM_HOST = os.getenv("LLM_HOST", "http://localhost:8080")
RESPONSE_MODEL_ID = os.getenv("RESPONSE_MODEL_ID", "gemma-4-E4B-it-Q8_0.gguf")

RESPONSE_TEMPERATURE = 0.1
RESPONSE_MAX_TOKENS = 32000
LLM_TIMEOUT = 1800          # секунд на один запрос
LLM_MAX_RETRIES = 2        # повторы при ошибках сервера
LLM_RETRY_DELAY = 2.0      # пауза между повторами

# Сколько последних сообщений из истории уходит в контекст модели
HISTORY_MESSAGES = 40

# ── База данных ──
# Сейчас SQLite; для перехода на PostgreSQL достаточно поменять
# DB_URL на "postgresql+psycopg2://user:pass@host/db" и поставить драйвер.
DB_URL = os.getenv("ELEONORA_DB_URL", f"sqlite:///{DATA_DIR / 'eleonora.db'}")

# ── Поиск по памяти (EmbAеddings + LanceDB) ──
# Эмбеддинг-модель: Giga-Embeddings-instruct-480M-0826 (BF16, dim=1024, CPU)
EMBEDDING_MODEL_PATH = DATA_DIR / "models_cache" / "Giga-Embeddings-480M-bf16"
EMBEDDING_INSTRUCTION = "Given a query, retrieve relevant personal facts and conversation memories"
LANCE_DB_PATH = DATA_DIR / "lance_db"
SEARCH_TOP_K = 10
# Порог cosine similarity для би-энкодера (Giga-Embeddings-480M, dim=1024).
# Эта модель даёт релевантное ~0.35-0.46, мусор <0.25 — см. черновик/заметки.
SEARCH_MIN_SCORE = 0.35

# ── Консолидация личных фактов (Этап 4) ──
# Извлекаем строгие личные факты о пользователе из его сообщений и
# сохраняем в LanceDB (таблица personal_facts) для векторного поиска.
CONSOLIDATOR_BATCH_SIZE = 5      # сколько user-сообщений копить перед консолидацией
CONSOLIDATOR_USER_NAME = os.getenv("ELEONORA_USER_NAME", "Сергей")
CONSOLIDATOR_DEDUP_THRESHOLD = 0.15  # порог cosine distance для дедупа фактов
CONSOLIDATOR_BUFFER_FILE = DATA_DIR / "consolidator_buffer.json"

# ── Агенты (рой) ──
# Отдельная модель для служебных агентов; по умолчанию та же что для ответов.
AGENT_MODEL_ID = os.getenv("AGENT_MODEL_ID", RESPONSE_MODEL_ID)
AGENT_TEMPERATURE = 0.1
AGENT_MAX_TOKENS = 300
AGENT_TIMEOUT = 120

# ── TTS (Silero v5.5 + winsound, Windows-only) ──
ENABLE_TTS = os.getenv("ENABLE_TTS", "True").lower() in ("true", "1", "yes")

# Silero понимает '+' перед ударной гласной — это наш механизм ударений.
# Голоса v5_5_ru.pt: женские — xenia, baya, kseniya; мужские — aidar, eugene.
TTS_MODEL_PATH = DATA_DIR / "tts_models" / "v5_5_ru.pt"
TTS_VOICE = os.getenv("TTS_VOICE", "xenia")
TTS_SAMPLE_RATE = 48000
TTS_DEVICE = os.getenv("TTS_DEVICE", "cpu")
TTS_MAX_CHUNK_CHARS = 800  # с запасом до лимита positional encoding (~5000 после фонетизации)
