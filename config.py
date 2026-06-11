# -*- coding: utf-8 -*-
"""Eleonora v2 — конфигурация."""

import os
from pathlib import Path
from typing import Optional

# ============================================================
# Device resolver — dml → privateuseone:0, cpu → cpu, и т.д.
# ============================================================
def resolve_device(device: str) -> str:
    """Преобразовать имя устройства в torch-совместимую строку.

    dml → privateuseone:0  (torch-directml)
    cpu/cuda/hip → as-is
    """
    d = device.lower().strip()
    if d == "dml":
        try:
            import torch_directml
            return str(torch_directml.device())
        except ImportError:
            return "cpu"
    return d


# ============================================================
# Пути
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ============================================================
# LLM SERVER (llama.cpp / llama-server)
# ============================================================
LLM_HOST = os.getenv("LLM_HOST", "http://localhost:8080")

# Модели
RESPONSE_MODEL_ID = "gemma-4-E4B-it-Q4_K_M.gguf"   # Основная модель для ответов

# Генерация
RESPONSE_TEMPERATURE = 0.1
RESPONSE_MAX_TOKENS = 12000
MODEL_TIMEOUT = 900  # секунды

# Повтор при ошибках сервера (400/500)
LLM_MAX_RETRIES = 2
LLM_RETRY_DELAY = 2.0  # секунды между попытками

# История
HISTORY_MESSAGES = 40  # Сколько сообщений хранить в контексте

# Сколько последних сообщений РЕНДЕРИТЬ в UI при старте. Полная история
# (self.messages) не подрезается — LLM-контекст и chat_history.json получают
# всё. Лимит чисто визуальный: 200+ виджетов в CTkScrollableFrame ломают
# layout/scrollregion (bubble'ы не отображаются, скролл уезжает не туда).
UI_MESSAGES_LIMIT = 100

# ============================================================
# SWARM CLASSIFIER
# ============================================================
ENABLE_SWARM_CLASSIFIER = True
SWARM_MODEL_ID = "gemma-4-E4B-it-UD-Q8_K_XL.gguf"
SWARM_TIMEOUT = 900

# ============================================================
# SEARCH (GigaEmbeddings + LanceDB + Reranker)
# ============================================================
ENABLE_SEARCH = True

# GigaEmbeddings
EMBEDDING_MODEL = "ai-sage/Giga-Embeddings-instruct"
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "dml")
MODELS_CACHE_DIR = DATA_DIR / "models_cache"

# LanceDB
LANCE_DB_PATH = DATA_DIR / "lance_db"

# Имена таблиц в LanceDB
TABLE_MESSAGES = "messages"           # Полная история диалога (user + assistant)
TABLE_PERSONAL_FACTS = "personal_facts"  # Строгие личные факты пользователя

# Поиск
SEARCH_TOP_K = 10         # Сколько кандидатов доставать из LanceDB

# Порог cosine distance embedder'а: если выше — факт отсекается до reranker'а.
# Giga-Embeddings (bi-encoder) плохо различает тематически близкие тексты —
# «чёрные дыры», «генетика», «ИИ» все оказываются в 0.45-0.55. Поэтому порог
# embedder'а — это грубый фильтр «очевидный мусор», не точный отбор.
# Настоящий арбитр качества — reranker ниже (RERANKER_MIN_SCORE). 0.60 срезает
# хвост ~0.6+ (тексты уже почти ортогональные семантически).
SEARCH_MIN_SCORE = 0.60

RERANKER_TOP_N = 3        # Сколько лучших после reranker (до MIN_SCORE отсечки)

# Абсолютный порог reranker'а — реальный гейт качества. Jina-v2 даёт sigmoid-
# нормализованный [0, 1]: релевантное 0.6-0.95, пограничное 0.2-0.5, мусор 0.01-0.1.
# ВАЖНО: cross-encoder перекалибровывает скоры не так как bi-encoder. Тематически
# похожий, но НЕ отвечающий на вопрос факт может получить 0.5 — surface form
# ловится (напр. «всегда за мной ходит» ↔ «мониторинг мобильных объектов» → 0.35).
# Понижено до 0.30 — реже режет релевантное.
RERANKER_MIN_SCORE = 0.30

# Ассоциативный поиск (Spreading Activation)
ENABLE_SPREADING_ACTIVATION = True  # Двухступенчатый поиск по якорям
ASSOCIATIVE_DECAY = 0.75            # Множитель скоров для ассоциаций (Stage 2)

# Dedup инъекции: сколько последних user-сообщений помнить,
# в течение которых одно и то же воспоминание не инжектим повторно в промпт.
# Цель — не захламлять контекст Геммы повторами при обсуждении одной темы.
# 9 ≈ половина HISTORY_MESSAGES=20, механизм внимания Gemma 4 прекрасно
# удержит показанный факт в этом окне.
MEMORY_RECENT_WINDOW_USER_MSGS = 9

# Reranker
ENABLE_RERANKER = True
RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "dml")

# ============================================================
# CONSOLIDATOR (извлечение СТРОГИХ личных фактов)
# ============================================================
ENABLE_CONSOLIDATOR = True
CONSOLIDATOR_BATCH_SIZE = 100      # Через сколько user-сообщений запускать (реже, качественнее)
CONSOLIDATOR_USER_NAME = "Сергей"  # Имя пользователя для третьего лица
CONSOLIDATOR_DEDUP_THRESHOLD = 0.15  # Очень строгий порог cosine distance для дубля

# Миграция старой истории в messages таблицу
MIGRATION_BATCH_SIZE = 50          # Размер батча для фоновой миграции chat_history.json

# ============================================================
# ПРОФИЛИ ПОЛЬЗОВАТЕЛЕЙ
# ============================================================
PROFILES_DIR = DATA_DIR / "profiles"
DEFAULT_PROFILE = "seryozha"

# Токсичность
TOXIC_BLOCK_THRESHOLD = 40        # profile_weight >= 40 → полный игнор
TOXIC_ACCUMULATE_MIN = 3          # Минимальный weight для накопления в профиль

# ============================================================
# EVENT SCHEDULER (отложенные триггеры инициативных реплик)
# ============================================================
# Порог cosine distance для overlap-детекции между активными триггерами
# и новым user-сообщением. Совпадение если cos_sim >= 1 - threshold.
# 0.30 ⇒ sim >= 0.70. Замерено на GigaEmbeddings: семантически близкие
# фразы ("мама уезжает через 15 минут" vs "мама уедет через 7 минут,
# у неё автобус скоро прибудет") дают ~0.715 — в нашей зоне. Если
# начнут ложно срабатывать на просто совпадении темы — поднять до 0.25.
EVENT_OVERLAP_THRESHOLD = 0.30

# Нижняя граница "серой зоны" (cosine similarity, не distance).
# Если best_sim в диапазоне [EVENT_OVERLAP_GREY_ZONE; 1-EVENT_OVERLAP_THRESHOLD)
# — зовём LLM-арбитра (event_overlap_judge), он решает same=yes/no.
# Ниже этого значения — сразу no hit, LLM не дёргаем (быстрее + меньше шума).
EVENT_OVERLAP_GREY_ZONE = 0.50

# ============================================================
# TTS (Silero v5.5, Windows-only — через stdlib winsound)
# ============================================================
# Озвучка ответов Элеоноры. Пайплайн: sanitize → yoficate → Silero → winsound.
# Синтез идёт в фоновом потоке, новый ответ прерывает текущее воспроизведение.
ENABLE_TTS = os.getenv("ENABLE_TTS", "True").lower() in ("true", "1", "yes")

# Путь к .pt-файлу Silero v5.5.
# Скачать: https://models.silero.ai/models/tts/ru/v5_5_ru.pt (~139 МБ).
# Держим в ASCII-пути — PyTorch на Windows не открывает не-ASCII (C:\Users\Жужа\...).
TTS_MODEL_PATH = DATA_DIR / "tts_models" / "v5_cis_base_nostress.pt"

# Голос. Женские: baya, kseniya, xenia. Мужские: aidar, eugene.
TTS_VOICE = os.getenv("TTS_VOICE", "ru_zhadyra").strip()

# Частота дискретизации: 8000 / 24000 / 48000. 48k — максимальное качество.
TTS_SAMPLE_RATE = 48000

# Устройство — CPU реал-таймово достаточен на Z1E, ROCm/DirectML на 780M
# экспериментальны и выигрыша не дают.
TTS_DEVICE = os.getenv("TTS_DEVICE", "cpu")

# Максимум символов на один apply_tts(). В Silero v5.5 positional encoding
# имеет хардкод-размер pe[:5000, :] — тексты после фонетизации, раскрывающиеся
# в >5000 символов, валят модель с «tensor (N) must match tensor (5000)».
#
# Эмпирический замер (probe в sandbox/tts_test): на русском Silero надёжно
# тянет 900 chars на вход, падает с 1000+. Ratio chars→символов после
# фонетизации ~5.5x, не 1.2-1.4 как казалось. Sam Silero warning'ит уже с 1000.
# Берём 800 с запасом — это ~50 секунд аудио на чанк, чанкер режет длинные
# ответы по предложениям/запятым (см. tts/chunker.py) и engine стримит.
TTS_MAX_CHUNK_CHARS = 800

# ============================================================
# VISUAL MEMORY (SigLIP 2 + LanceDB + Vision LLM)
# ============================================================
# Визуальная память: запоминание фото с описанием + поиск по тексту/картинке.
# Работает через SigLIP 2 (image+text encoder в одном пространстве 1024d)
# и vision-LLM (Gemma 4 vision через LM Studio OpenAI-compatible API).

ENABLE_VISUAL_MEMORY = True

# SigLIP 2 модель — multilingual image+text encoder.
# Первый запуск скачает ~1.1 GB в MODELS_CACHE_DIR.
SIGLIP_MODEL = "google/siglip2-large-patch16-256"
SIGLIP_DIM = 1024
SIGLIP_DEVICE = os.getenv("SIGLIP_DEVICE", "dml")

# Отдельная LanceDB для визуальных воспоминаний (не смешиваем с текстовой памятью).
VISUAL_DB_PATH = DATA_DIR / "visual_lance_db"

# Папка для копий фото, переименованных в vis_YYYYMMDD_HHMMSS_microsec.<ext>
VISUAL_PHOTOS_DIR = DATA_DIR / "visual_photos"

# Сколько кандидатов возвращать при поиске
VISUAL_SEARCH_TOP_K = 3

# Vision-LLM ( multimodal модель для описания фото).
# Если модель text-only (напр. gemma-4-E4B) — vision отключён.
# Подключи vision-модель (Gemma 4 multi-modal) и укажи её имя здесь.
USE_VISION_LLM = False  # Отключено: текущая модель text-only
VISION_MODEL_ID = None  # Указать имя .gguf когда подключишь vision-модель
VISION_LLM_TIMEOUT = 600

