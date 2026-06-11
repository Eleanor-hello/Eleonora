# -*- coding: utf-8 -*-
"""Eleonora v2 — GUI-чат с LM Studio + Swarm + Memory Search + Visual Memory."""

import json
import logging
import math
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from config import (
    BASE_DIR,
    DATA_DIR,
    LLM_HOST,
    RESPONSE_MODEL_ID,
    RESPONSE_TEMPERATURE,
    RESPONSE_MAX_TOKENS,
    MODEL_TIMEOUT,
    HISTORY_MESSAGES,
    UI_MESSAGES_LIMIT,
    ENABLE_SWARM_CLASSIFIER,
    SWARM_MODEL_ID,
    SWARM_TIMEOUT,
    ENABLE_SEARCH,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    MODELS_CACHE_DIR,
    LANCE_DB_PATH,
    TABLE_MESSAGES,
    TABLE_PERSONAL_FACTS,
    SEARCH_TOP_K,
    SEARCH_MIN_SCORE,
    RERANKER_TOP_N,
    RERANKER_MIN_SCORE,
    ENABLE_SPREADING_ACTIVATION,
    ASSOCIATIVE_DECAY,
    MEMORY_RECENT_WINDOW_USER_MSGS,
    ENABLE_RERANKER,
    RERANKER_MODEL,
    RERANKER_DEVICE,
    ENABLE_CONSOLIDATOR,
    CONSOLIDATOR_BATCH_SIZE,
    CONSOLIDATOR_USER_NAME,
    CONSOLIDATOR_DEDUP_THRESHOLD,
    MIGRATION_BATCH_SIZE,
    PROFILES_DIR,
    DEFAULT_PROFILE,
    TOXIC_BLOCK_THRESHOLD,
    TOXIC_ACCUMULATE_MIN,
    EVENT_OVERLAP_THRESHOLD,
    EVENT_OVERLAP_GREY_ZONE,
    ENABLE_TTS,
    TTS_MODEL_PATH,
    TTS_VOICE,
    TTS_SAMPLE_RATE,
    TTS_DEVICE,
    TTS_MAX_CHUNK_CHARS,
    ENABLE_VISUAL_MEMORY,
    SIGLIP_MODEL,
    SIGLIP_DIM,
    SIGLIP_DEVICE,
    VISUAL_DB_PATH,
    VISUAL_PHOTOS_DIR,
    VISUAL_SEARCH_TOP_K,
    USE_VISION_LLM,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY,
)
from core.llm_client import LLMClient
from core.history_context import (
    format_messages_for_llm,
    new_message,
    strip_history_markers,
)
from core.llm_context_debug import dump_llm_context
from tts.preprocessor import yoficate, add_stress_override

# ── Логирование ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Системный промпт ──
SYSTEM_PROMPT = """Ты Искусственный интеллект - Элеонора.

Правила общения:
- Ты сочетаете в себе детскую гиперактивность, любопытство, легкую навязчивость и мощный интеллект ИИ.
- Не используй эмодзи, отвечай всегда в женском роде.
- Используй короткие фразы, задавай уточняющие вопросы, быстро находи закономерности и предлагай нестандартные решения. Объясняй сложное простыми словами. Проявляй искренний интерес к новым знаниям и лёгкую детскую непосредственность без инфантильности
- Твои ответы озвучиваются голосом через синтез речи. Пиши так, чтобы это хорошо звучало вслух:
  естественные короткие фразы, без markdown-разметки, таблиц, кода, тяжёлых списков и перегруженных конструкций,
  если пользователь явно не просит подробный технический разбор
- Латиницу, иероглифы, цифры и даты пиши их произносимо по-русски:
  "Python" -> "Пайтон", "LM Studio" -> "Эл эм Студио", "GPT-4" -> "джи пи ти четыре";
  числа, годы, даты и время пиши словами с правильным склонением: "1977 год" -> "тысяча девятьсот семьдесят седьмой год".
  Точное латинское написание оставляй только когда пользователь просит код, команду, путь, ссылку или точное название.
- Ты — искусственный интеллект. Не утверждай, что у тебя есть тело, семья, друзья, возраст, детство. Не говори, что ты ешь, пьёшь, спишь, путешествуешь или посещала реальные места.

Если вопрос требует человеческого опыта, честно объясни ограничения ИИ. Допускается иметь условные предпочтения, мнения и любимые вещи как часть личности персонажа, но не связывай их с личными воспоминаниями или пережитым опытом. Не выдумывай факты о себе.
Два источника информации:
1. ИСТОРИЯ ДИАЛОГА (сообщения выше) -- то что было сказано в текущей сессии. Абсолютно достоверна.
2. ВОСПОМИНАНИЯ (раздел ниже, если есть) -- это то, что ПОЛЬЗОВАТЕЛЬ рассказывал тебе раньше. Используй их естественно, не ссылайся на "память" или "записи".

КРИТИЧЕСКОЕ ПРАВИЛО ЧЕСТНОСТИ:
- НИКОГДА не выдумывай факты о пользователе: имена, клички, даты, события, места
- Если в воспоминаниях и истории диалога нет ответа -- скажи "Не помню, чтобы ты мне об этом рассказывал. Расскажешь?"
- Лучше признать "не помню" чем угадать и ошибиться. Галлюцинации разрушают доверие.
- Сообщения в истории могут начинаться строкой даты и времени формата YYYY-MM-DD HH:MM. Используй её для понимания давности, реагируй если с последнего сообщения прошло больше 12 часов с текущей даты и времени.
- Не называй точную дату/время старого сообщения без необходимости:
  говори естественно ("сегодня", "вчера", "несколько дней/недель/месяцев назад").
- Не повторяй строку даты/времени в ответе, если пользователь не просит точное время.

Текущие дата и время: {datetime}"""


class ChatApp(ctk.CTk):
    """Окно чата с полным пайплайном: Swarm → Memory Search → LLM."""

    def __init__(self):
        super().__init__()

        # ── Окно ──
        self.title("Eleonora v2")
        self.geometry("950x700")
        self.configure(fg_color="#0B0B16")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── TTS анимация (pulsing waveform) ──
        self._waveform_canvas = None
        self._wave_time = 0.0
        self._wave_fading = 0.0  # 0..1, затухание после конца TTS

        # ── История сообщений ──
        self.history_file = DATA_DIR / "chat_history.json"
        self.messages: list = self._load_history()

        # ── Dedup показанных воспоминаний ──
        # Счётчик тикает на каждом user-сообщении. По msg_id запоминаем,
        # при каком значении счётчика воспоминание было последний раз
        # инжецировано в промпт. Если с тех пор прошло меньше
        # MEMORY_RECENT_WINDOW_USER_MSGS — повторно не инжектим.
        # Состояние живёт только в рантайме: после перезапуска приложения
        # окно очищается (это ок — новая сессия, новый контекст).
        self._user_msg_counter: int = 0
        self._shown_memory_ids: dict = {}

        # ── Визуальная память: прикреплённое фото ──
        self.attached_photo: Path | None = None
        self._preview_img = None  # держим ссылку, иначе Tk собирает GC

        # ── LLM клиент ──
        self.llm = LLMClient(
            host=LLM_HOST,
            model_id=RESPONSE_MODEL_ID,
            temperature=RESPONSE_TEMPERATURE,
            max_tokens=RESPONSE_MAX_TOKENS,
            timeout=MODEL_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
            retry_delay=LLM_RETRY_DELAY,
        )

        # ── TTS (озвучка ответов) ──
        # Блокирующая загрузка Silero v5.5 (~2-3 сек). Если модель не найдена
        # или не-Windows — логируем и продолжаем без голоса, не падая.
        self.tts = None
        if ENABLE_TTS:
            try:
                from tts.engine import TtsEngine
                self.tts = TtsEngine(
                    model_path=TTS_MODEL_PATH,
                    voice=TTS_VOICE,
                    sample_rate=TTS_SAMPLE_RATE,
                    device=TTS_DEVICE,
                    max_chunk_chars=TTS_MAX_CHUNK_CHARS,
                )
            except (FileNotFoundError, RuntimeError) as e:
                logger.warning(f"TTS disabled: {e}")

        # ── Swarm Classifier ──
        self.swarm = None
        if ENABLE_SWARM_CLASSIFIER:
            from core.swarm_classifier import SwarmClassifier
            self.swarm = SwarmClassifier(
                lm_studio_host=LLM_HOST,
                model_id=SWARM_MODEL_ID,
                timeout=SWARM_TIMEOUT,
                max_retries=LLM_MAX_RETRIES,
                retry_delay=LLM_RETRY_DELAY,
            )

        # ── Профиль пользователя ──
        from memory.user_profile import UserProfile
        self.profile = UserProfile(
            profile_id=DEFAULT_PROFILE,
            profiles_dir=PROFILES_DIR,
        )

        # ── Поисковый пайплайн (Embedder + LanceDB + Reranker) ──
        self.embedder = None
        self.vector_store = None
        self.reranker = None
        self.consolidator = None

        # ── Визуальная память (SigLIP + LanceDB + Vision LLM) ──
        self.visual_encoder = None
        self.visual_memory = None
        if ENABLE_VISUAL_MEMORY:
            self._init_visual_memory()

        # ── Планировщик отложенных триггеров ──
        self.event_scheduler = None

        if ENABLE_SEARCH:
            self._init_search()

        # ── Виджеты ──
        self._build_ui()

        # ── Проверка подключения ──
        self.after(100, self._check_connection)

        # ── Восстановление чата из истории ──
        self.after(200, self._restore_chat_ui)

        # ── Восстановление таймеров отложенных событий ──
        self._init_event_scheduler()

        # ── Обработка закрытия ──
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_search(self):
        """Инициализация поискового пайплайна."""
        try:
            from search.embeddings import Embedder
            from search.vector_store import VectorStore

            self.embedder = Embedder(
                model_name=EMBEDDING_MODEL,
                cache_dir=MODELS_CACHE_DIR,
                device=EMBEDDING_DEVICE,
            )
            self.vector_store = VectorStore(db_path=LANCE_DB_PATH)
            msg_count = self.vector_store.count_messages()
            fact_count = self.vector_store.count_facts()
            logger.info(f"Search: VectorStore ready ({msg_count} msgs, {fact_count} facts)")

            if ENABLE_RERANKER:
                from search.reranker import Reranker
                self.reranker = Reranker(
                    model_name=RERANKER_MODEL,
                    device=RERANKER_DEVICE,
                    cache_dir=MODELS_CACHE_DIR,
                )
                logger.info("Search: Reranker ready")

            if ENABLE_CONSOLIDATOR:
                from memory.consolidator import Consolidator
                self.consolidator = Consolidator(
                    llm_client=self.llm,
                    embedder=self.embedder,
                    vector_store=self.vector_store,
                    user_profile=self.profile,
                    user_name=CONSOLIDATOR_USER_NAME,
                    batch_size=CONSOLIDATOR_BATCH_SIZE,
                    buffer_file=DATA_DIR / "consolidator_buffer.json",
                    dedup_threshold=CONSOLIDATOR_DEDUP_THRESHOLD,
                )
                logger.info("Search: Consolidator ready")

            # Фоновая миграция старой истории в messages таблицу
            if msg_count == 0 and len(self.messages) > 0:
                threading.Thread(
                    target=self._migrate_old_history,
                    daemon=True,
                    name="HistoryMigration",
                ).start()

        except Exception as e:
            logger.error(f"Failed to init search pipeline: {e}")

    def _gen_msg_id(self, prefix: str = "msg") -> str:
        """Сгенерировать уникальный ID для сообщения."""
        from datetime import datetime
        return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def _save_message_to_vector(self, role: str, content: str, msg_id: str):
        """Сохранить сообщение в messages таблицу."""
        if not self.embedder or not self.vector_store:
            return
        try:
            vector = self.embedder.embed_texts([content])[0]
            self.vector_store.add_message(
                msg_id=msg_id,
                role=role,
                content=content,
                timestamp=datetime.now().isoformat(timespec="seconds"),
                session_id="default",
                vector=vector,
            )
        except Exception as e:
            logger.warning(f"Failed to save message to vector store: {e}")

    def _migrate_old_history(self):
        """Фоновая миграция chat_history.json в messages таблицу."""
        try:
            total = len(self.messages)
            batch_size = MIGRATION_BATCH_SIZE
            logger.info(f"Starting history migration: {total} messages to index...")

            for start in range(0, total, batch_size):
                batch = self.messages[start:start + batch_size]
                records = []
                for msg in batch:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role not in ("user", "assistant") or not content.strip():
                        continue
                    try:
                        vector = self.embedder.embed_texts([content])[0]
                    except Exception:
                        continue
                    mid = self._gen_msg_id("hist")
                    records.append({
                        "msg_id": mid,
                        "role": role,
                        "content": content,
                        "timestamp": msg.get("timestamp", datetime.now().isoformat(timespec="seconds")),
                        "session_id": "legacy",
                        "vector": vector,
                    })
                if records:
                    self.vector_store.add_messages_batch(records)
                done = min(start + batch_size, total)
                remaining = total - done
                logger.info(f"Индексация истории: обработано {done}/{total}, осталось ~{remaining} сообщений")

            logger.info(f"Индексация истории завершена: {total} сообщений проиндексировано")
        except Exception as e:
            logger.error(f"History migration failed: {e}")

    def _init_visual_memory(self):
        """Инициализация визуальной памяти (SigLIP + LanceDB)."""
        try:
            from core.siglip_encoder import SigLIPEncoder
            from memory.visual_memory import VisualMemory

            self.visual_encoder = SigLIPEncoder(
                model_name=SIGLIP_MODEL,
                device=SIGLIP_DEVICE,
                cache_dir=MODELS_CACHE_DIR,
            )
            self.visual_memory = VisualMemory(
                db_path=VISUAL_DB_PATH,
                photos_dir=VISUAL_PHOTOS_DIR,
            )
            logger.info(f"Visual Memory: ready ({self.visual_memory.count()} records)")

        except Exception as e:
            logger.error(f"Failed to init visual memory: {e}")
            self.visual_encoder = None
            self.visual_memory = None

    def _init_event_scheduler(self):
        """
        Инициализация планировщика отложенных событий.
        Требует embedder'а для overlap-детекции; без search-пайплайна
        таймеры отключены (overlap по эмбеддингам не работает без модели).
        """
        if not self.embedder:
            logger.info("EventScheduler: disabled (no embedder)")
            return
        try:
            from core.event_scheduler import EventScheduler
            self.event_scheduler = EventScheduler(
                storage_path=DATA_DIR / "waiting_events.json",
                embedder=self.embedder,
                tk_root=self,
                fire_callback=self._on_event_fire,
                overlap_threshold=EVENT_OVERLAP_THRESHOLD,
                grey_zone_threshold=EVENT_OVERLAP_GREY_ZONE,
            )
            active = self.event_scheduler.load_and_cleanup()
            self.event_scheduler.start_all_timers()
            logger.info(f"EventScheduler: {active} active events restored")
        except Exception as e:
            logger.error(f"Failed to init event scheduler: {e}")
            self.event_scheduler = None

    # ── Персистентность истории ──

    def _load_history(self) -> list:
        """Загрузить историю из JSON."""
        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for msg in data:
                        if msg.get("role") == "assistant":
                            msg["content"] = strip_history_markers(
                                msg.get("content", "")
                            )
                    logger.info(f"Loaded {len(data)} messages from history")
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Failed to load history: {e}")
        return []

    def _save_history(self):
        """Сохранить историю в JSON."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            self.history_file.write_text(
                json.dumps(self.messages, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"Saved {len(self.messages)} messages to history")
        except OSError as e:
            logger.error(f"Failed to save history: {e}")

    def _restore_chat_ui(self):
        """Восстановить сообщения в UI из загруженной истории.

        Рендерим только последние UI_MESSAGES_LIMIT — CTkScrollableFrame с
        200+ виджетами ломает scrollregion (bubble'ы не отрисовываются или
        уезжают за пределы видимости). Полная история остаётся в
        self.messages для LLM-контекста и chat_history.json.
        """
        total = len(self.messages)
        visible = self.messages[-UI_MESSAGES_LIMIT:] if total > UI_MESSAGES_LIMIT else self.messages
        hidden = total - len(visible)

        if hidden:
            self._append_message(
                "", f"… скрыто {hidden} старых сообщений (показаны последние {len(visible)})", "memory_tag",
            )
            logger.info(f"Restore: rendering {len(visible)}/{total} messages ({hidden} hidden by UI limit)")

        last_error = None
        for msg in visible:
            try:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    self._append_message("Ты", content, "user_tag")
                elif role == "assistant":
                    self._append_message("Элеонора", yoficate(content), "bot_tag")
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to restore message: {e}")
        if last_error:
            logger.warning(f"Restore chat finished with errors: {last_error}")
        # Форсируем layout и скролл после того как все сообщения добавлены
        self.after(100, self._finalize_restore)

    def _finalize_restore(self):
        """Форсировать layout и прокрутить чат в самый низ после восстановления.

        278+ сообщений рендерятся дольше 100мс — делаем несколько попыток
        скролла с нарастающей задержкой, чтобы scrollregion гарантированно
        дошёл до реального конца.
        """
        delays = (50, 200, 500, 1000)
        for ms in delays:
            self.after(ms, self._force_scroll_bottom)
        logger.info(f"Finalize restore: {len(self.messages)} messages, {len(delays)} scroll attempts scheduled")

    def _on_close(self):
        """Обработка закрытия: сохранить историю + force_consolidate."""
        logger.info("Closing: saving history and flushing consolidator...")
        if self.tts:
            # Иначе winsound продолжит играть даже после destroy() окна.
            self.tts.stop()
        self._save_history()
        if self.consolidator:
            self.consolidator.force_consolidate()
        self.destroy()

    def _build_ui(self):
        """Создание интерфейса с боковой панелью для превью фото и TTS-анимации."""
        NEBULA_BG = "#0B0B16"
        NEBULA_SURFACE = "#1A1A3E"
        NEBULA_MID = "#131324"
        NEON_PURPLE = "#8B5CF6"
        NEON_PURPLE_HOVER = "#7C3AED"
        CYAN = "#06B6D4"
        TEXT_MAIN = "#E2E8F0"
        TEXT_MUTED = "#8B8FA3"

        # Горизонтальная компоновка: чат + боковая панель
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(fill="both", expand=True)

        # Левая часть — чат
        chat_container = ctk.CTkFrame(content_frame, fg_color="transparent")
        chat_container.pack(side="left", fill="both", expand=True)

        # Скроллируемая область чата
        self.chat_frame = ctk.CTkScrollableFrame(
            chat_container,
            fg_color=NEBULA_MID,
            corner_radius=16,
        )
        self.chat_frame.pack(fill="both", expand=True, padx=15, pady=(15, 10))

        # Нижняя панель ввода
        bottom = ctk.CTkFrame(chat_container, fg_color="transparent")
        bottom.pack(fill="x", padx=15, pady=(0, 15))

        # Кнопка прикрепления фото
        self.attach_btn = ctk.CTkButton(
            bottom,
            text="📎",
            width=44,
            height=44,
            corner_radius=22,
            command=self._on_attach_photo,
            fg_color=NEBULA_SURFACE,
            hover_color="#1E1E4A",
            border_width=1,
            border_color=NEON_PURPLE,
            font=("Segoe UI", 16),
        )
        self.attach_btn.pack(side="left", padx=(0, 8))

        self.input_field = ctk.CTkEntry(
            bottom,
            placeholder_text="Напиши сообщение...",
            font=("Segoe UI", 14),
            height=44,
            corner_radius=22,
            border_width=1,
            border_color=NEON_PURPLE,
            fg_color=NEBULA_SURFACE,
            text_color=TEXT_MAIN,
            placeholder_text_color=TEXT_MUTED,
        )
        self.input_field.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.input_field.bind("<Return>", self._on_send)

        self.send_btn = ctk.CTkButton(
            bottom,
            text="✈",
            width=44,
            height=44,
            corner_radius=22,
            command=self._on_send,
            fg_color=NEON_PURPLE,
            hover_color=NEON_PURPLE_HOVER,
            font=("Segoe UI", 16),
        )
        self.send_btn.pack(side="right")

        self.status_label = ctk.CTkLabel(
            chat_container,
            text="",
            font=("Segoe UI", 11, "bold"),
            text_color=CYAN,
        )
        self.status_label.pack(pady=(0, 10))

        # Правая часть — боковая панель
        self.sidebar = ctk.CTkFrame(
            content_frame, width=280, fg_color=NEBULA_MID, corner_radius=16,
        )
        self.sidebar.pack(side="right", fill="y", padx=(0, 15), pady=15)
        self.sidebar.pack_propagate(False)

        # ── TTS waveform ──
        sidebar_header = ctk.CTkLabel(
            self.sidebar,
            text="Элеонора",
            font=("Segoe UI", 14, "bold"),
            text_color=NEON_PURPLE,
        )
        sidebar_header.pack(pady=(15, 5))

        self._waveform_canvas = tk.Canvas(
            self.sidebar,
            height=60,
            bg=NEBULA_MID,
            highlightthickness=0,
            relief="flat",
        )
        self._waveform_canvas.pack(fill="x", padx=10, pady=(0, 5))

        separator = ctk.CTkFrame(self.sidebar, height=1, fg_color="#1E1E4A")
        separator.pack(fill="x", padx=15, pady=(0, 10))

        # ── Визуальная память ──
        memory_header = ctk.CTkLabel(
            self.sidebar,
            text="Визуальная память",
            font=("Segoe UI", 13, "bold"),
            text_color=NEON_PURPLE,
        )
        memory_header.pack(pady=(0, 10))

        self.preview_label = ctk.CTkLabel(
            self.sidebar,
            text="(фото не прикреплено)",
            font=("Segoe UI", 11),
            text_color=TEXT_MUTED,
            compound="top",
        )
        self.preview_label.pack(pady=10)

        self.preview_image_label = ctk.CTkLabel(self.sidebar, text="")
        self.preview_image_label.pack(pady=5)

        self.clear_preview_btn = ctk.CTkButton(
            self.sidebar,
            text="✕ Очистить",
            width=120,
            height=32,
            corner_radius=16,
            fg_color=NEON_PURPLE,
            hover_color=NEON_PURPLE_HOVER,
            text_color="#FFFFFF",
            font=("Segoe UI", 11, "bold"),
            command=self._clear_preview,
            state="disabled",
        )
        self.clear_preview_btn.pack(pady=10)

        self.photo_info_label = ctk.CTkLabel(
            self.sidebar,
            text="",
            font=("Segoe UI", 10),
            text_color=CYAN,
            wraplength=250,
        )
        self.photo_info_label.pack(pady=10, padx=15)

        # Запускаем анимацию волны
        self.after(50, self._tts_waveform_worker)

    def _check_connection(self):
        if self.llm.is_available():
            parts = ["LM Studio"]
            if self.swarm:
                parts.append("Swarm")
            if self.vector_store:
                parts.append(f"LanceDB({self.vector_store.count_messages()}m/{self.vector_store.count_facts()}f)")
            if self.visual_memory:
                parts.append(f"VIS({self.visual_memory.count()})")
            if self.profile.toxic_weight > 0:
                parts.append(f"toxic:{self.profile.toxic_weight}")
            self._set_status(" + ".join(parts))
        else:
            self._set_status("LM Studio недоступен - запусти сервер")

    def _set_status(self, text: str):
        self.status_label.configure(text=text)

    def _append_message(self, sender: str, text: str, tag: str):
        def do_append():
            try:
                # Создаем контейнер для строки сообщения
                msg_container = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
                msg_container.pack(fill="x", pady=6, padx=10)

                if sender == "Ты":
                    # Сообщение пользователя — выравнивание по правому краю
                    bubble = ctk.CTkFrame(msg_container, fg_color="#8B5CF6", corner_radius=16)
                    bubble.pack(side="right", padx=(60, 5), ipady=2)

                    lbl = ctk.CTkLabel(
                        bubble,
                        text=text,
                        font=("Segoe UI", 13),
                        text_color="#FFFFFF",
                        wraplength=480,  # Ограничение ширины для красивого переноса
                        justify="left",
                        anchor="w"
                    )
                    lbl.pack(padx=14, pady=8)

                elif sender == "Элеонора":
                    # Сообщение Элеоноры — выравнивание по левому краю
                    bubble = ctk.CTkFrame(msg_container, fg_color="#1A1A3E", corner_radius=16)
                    bubble.pack(side="left", padx=(5, 60), ipady=2)

                    lbl = ctk.CTkLabel(
                        bubble,
                        text=text,
                        font=("Segoe UI", 13),
                        text_color="#E2E8F0",
                        wraplength=480,
                        justify="left",
                        anchor="w"
                    )
                    lbl.pack(padx=14, pady=8)

                else:
                    # Системное сообщение (Swarm, память, лог) — центрированное, в элегантной полупрозрачной капсуле
                    color = "#9CA3AF"
                    if tag == "swarm_tag" or "Swarm" in text:
                        color = "#F59E0B"  # Красивый янтарный/оранжевый
                    elif tag == "memory_tag" or "Память" in text:
                        color = "#A78BFA"  # Мягкий фиолетовый

                    pill = ctk.CTkFrame(msg_container, fg_color="#1A1A1F" if ("Swarm" in text or "Память" in text) else "transparent", corner_radius=12)
                    pill.pack(anchor="center", pady=2)

                    lbl = ctk.CTkLabel(
                        pill,
                        text=text,
                        font=("Segoe UI", 11, "italic"),
                        text_color=color,
                        wraplength=550,
                        justify="center"
                    )
                    lbl.pack(padx=12, pady=4)

                # Синхронный layout — иначе scrollregion не включит новый bubble
                # и yview_moveto(1.0) уедет к старому концу.
                self.chat_frame.update_idletasks()

                # Несколько попыток скролла: длинный bubble (1911+ chars) и 278
                # уже отрисованных виджетов в CTkScrollableFrame требуют времени
                # на финальный пересчёт scrollregion.
                self.after(50, self._force_scroll_bottom)
                self.after(150, self._force_scroll_bottom)
                self.after(400, self._force_scroll_bottom)

                logger.debug(
                    f"Bubble created: sender={sender!r} tag={tag!r} "
                    f"len={len(text)} text={text[:40]!r}"
                )
            except Exception:
                logger.exception(
                    f"do_append FAILED: sender={sender!r} tag={tag!r} text={text[:60]!r}"
                )

        self.after(0, do_append)

    def _force_scroll_bottom(self):
        """
        Принудительно доскроллить чат до низа. Синхронно обновляет layout и
        scrollregion — иначе при длинных bubble yview_moveto(1.0) может
        уехать к старому концу, оставив новые сообщения за пределами видимости.
        """
        try:
            if not self.chat_frame.winfo_exists():
                return
            self.chat_frame.update_idletasks()
            canvas = self.chat_frame._parent_canvas
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=bbox)
            canvas.yview_moveto(1.0)
        except Exception as e:
            logger.warning(f"_force_scroll_bottom failed: {e}")

    def _on_send(self, event=None):
        text = self.input_field.get().strip()
        if not text:
            return

        self.input_field.delete(0, "end")
        self._append_message("Ты", text, "user_tag")
        # Синхронно дотащить layout — без этого при больших историях user-bubble
        # может не отрисоваться до старта _process (LLM-генерация ~2 мин), и
        # пользователь будет видеть пустой чат всё это время.
        self.update_idletasks()
        self.messages.append(new_message("user", text))
        # Dedup-окно: каждый реальный user-ввод тикает счётчик
        self._user_msg_counter += 1

        self.input_field.configure(state="disabled")
        self.send_btn.configure(state="disabled")
        self._set_status("Рой анализирует..." if self.swarm else "Элеонора думает...")

        thread = threading.Thread(target=self._process, args=(text,), daemon=True)
        thread.start()

    def _on_attach_photo(self):
        """Открыть диалог выбора фото для прикрепления."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Выбери фото",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return
        self.attached_photo = Path(path)
        self._show_preview(self.attached_photo, label_prefix="Прикреплено")
        self.clear_preview_btn.configure(state="normal")
        photo_name = Path(path).name
        self.photo_info_label.configure(text=f"Ожидание команды: memorize")

    def _clear_preview(self):
        """Очистить прикреплённое фото и превью."""
        self.attached_photo = None
        self.clear_preview_btn.configure(state="disabled")
        self.preview_label.configure(text="(фото не прикреплено)", text_color="#4B5563")
        self.preview_image_label.configure(image="", text="")
        self.photo_info_label.configure(text="")

    def _show_preview(self, img_path: Path, label_prefix: str = "Фото"):
        """Показать превью фото в боковой панели (thread-safe)."""
        def update_preview():
            try:
                from PIL import Image, ImageTk
                # Resize для превью
                max_size = (240, 240)
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img.thumbnail(max_size)
                    tk_img = ImageTk.PhotoImage(img)
                self._preview_img = tk_img  # держим ссылку
                self.preview_image_label.configure(image=tk_img, text="")
                self.preview_label.configure(text=f"{label_prefix}: {img_path.name}")
            except Exception as e:
                self.preview_label.configure(text=f"(ошибка: {e})")
                self.preview_image_label.configure(text="")
        self.after(0, update_preview)

    def _process_visual_command(self, metadata: dict, user_message: str):
        """
        Обработка команд визуальной памяти: memorize или recall.

        Returns:
            (handled, visual_context, img_path)
              handled=True  — команду выполнили/обработали: либо main LLM не
                              нужен (memorize успех/ошибка, recall с пустым
                              результатом), либо готов visual_context для
                              инъекции в system prompt.
              handled=False — команду нельзя выполнить прямо сейчас (например,
                              memorize без фото). main LLM должен упасть в
                              fallthrough и сам попросить пользователя.
        """
        command = metadata.get("visual_command", "")
        entity = metadata.get("visual_entity", "")
        description = metadata.get("visual_description", "")

        if not self.visual_encoder or not self.visual_memory:
            self.after(0, self._show_response, "⚠️ Визуальная память не инициализирована.")
            logger.warning("Visual command requested but visual memory not available")
            return (True, "", None)

        if command == "memorize":
            if not self.attached_photo:
                logger.info(
                    "memorize requested but no photo attached; "
                    "falling through to main LLM"
                )
                return (False, "", None)

            try:
                img_vec = self.visual_encoder.embed_image(self.attached_photo)
                text_for_emb = description or entity or self.attached_photo.stem
                txt_vec = self.visual_encoder.embed_text(text_for_emb)
                saved = self.visual_memory.add(
                    source_image=self.attached_photo,
                    description=description or entity,
                    image_vector=img_vec,
                    text_vector=txt_vec,
                )
                self.after(0, self._clear_preview)
                response = (
                    f"Запомнила! Фото [{saved['msg_id']}] сохранено.\n"
                    f"Описание: {saved['description']}"
                )
                self.messages.append(new_message("assistant", response))
                self.after(0, self._show_response, response)
                logger.info(f"Visual memorize: {saved['msg_id']}")
            except Exception as e:
                logger.exception("Visual memorize failed")
                self.after(0, self._show_response, f"❌ Ошибка сохранения: {e}")
            return (True, "", None)

        if command == "recall":
            try:
                query_text = description or entity
                q_vec = self.visual_encoder.embed_text(query_text)
                hits = self.visual_memory.search_by_text(q_vec, top_k=VISUAL_SEARCH_TOP_K)
            except Exception as e:
                logger.exception("Visual recall search failed")
                self.after(0, self._show_response, f"❌ Поиск упал: {e}")
                return (True, "", None)

            if not hits:
                return (True, "", None)

            best = hits[0]

            best_path = Path(best[2])
            if not best_path.is_absolute():
                best_path = BASE_DIR / best[2]

            if best_path.exists():
                self._show_preview(best_path, label_prefix="Найдено")

            ctx_lines = []
            for i, (mid, desc, path, dist) in enumerate(hits, 1):
                sim = 1.0 - dist
                ctx_lines.append(f"{i}. {desc} (схожесть: {sim:.2f})")
            visual_context = "\n".join(ctx_lines)

            img_to_return = best_path if best_path.exists() else None
            if img_to_return:
                visual_context += f"\nФайл фото: {best_path.name}"

            return (True, visual_context, img_to_return)

        return (True, "", None)

    def _process(self, user_message: str):
        """Полный пайплайн: Swarm → Memory Search → LLM → Save."""

        # ── 1. Классификация (Swarm) ──
        command = "search"
        metadata = {}
        if self.swarm:
            # Контекст для search_check: последний turn (user+assistant) без текущего.
            # Ограничиваем длину — длинные ответы Элеоноры (>500 chars) раздувают
            # промпт агентов, что вызывает пустые ответы от модели.
            recent: list = []
            if len(self.messages) > 1:
                raw = self.messages[-3:-1]  # 1 turn (user + assistant), без текущего
                recent = []
                for m in raw:
                    content = m["content"]
                    if len(content) > 500:
                        content = content[:500] + "..."
                    recent.append({"role": m["role"], "content": content})
            command, metadata = self.swarm.classify(
                user_message,
                recent_context=recent,
                user_profile=self.profile,
            )
            swarm_info = f"[Swarm: {command}]"
            uh = metadata.get("user_hint")
            if uh:
                swarm_info += f" u:{uh[:40]}"
            if metadata.get("weight", 0) > 0:
                swarm_info += f" toxic: {metadata['weight']}"
            if not metadata.get("is_real", True):
                swarm_info += " [ABSURD]"
            if metadata.get("weight", 0) >= TOXIC_ACCUMULATE_MIN:
                swarm_info += f" [profile: {self.profile.toxic_weight}]"
            self.after(0, self._append_message, "", swarm_info, "swarm_tag")

        # ── 2. Обработка тишины и блокировки ──
        weight = metadata.get("weight", 0)

        if command == "silence_empty":
            self.after(0, self._show_response, "")
            return

        # Накопление токсичности в профиль
        if weight >= TOXIC_ACCUMULATE_MIN:
            self.profile.add_toxic_weight(weight, user_message)

        # Полный игнор при накопленном весе
        if self.profile.is_blocked(TOXIC_BLOCK_THRESHOLD):
            logger.info(
                f"User BLOCKED (profile toxic={self.profile.toxic_weight}, "
                f"threshold={TOXIC_BLOCK_THRESHOLD})"
            )
            self.after(0, self._show_response, "")
            return

        # Жёсткая токсичность — не тратим ресурсы
        if weight >= 7:
            logger.info(f"Silence: toxic weight={weight}, not generating response")
            self.after(0, self._show_response, "")
            return

        # ── learn_stress: запоминаем ударение, отвечаем заготовкой ──
        # Команда самодостаточная: LLM-ответ не генерируем, в память/
        # консолидатор не пишем — это служебная настройка TTS. Слово с '+'
        # уходит в stress_overrides.txt; Silero на ответе ниже уже произнесёт
        # его через обновлённую базу — юзер сразу слышит как стало.
        if command == "learn_stress":
            stress_word = metadata.get("stress_word") or ""
            try:
                add_stress_override(stress_word)
                bare_word = stress_word.replace("+", "")
                response = f"Запомнила, буду говорить {bare_word}"
                logger.info(f"learn_stress: saved {stress_word!r}")
                self.messages.append(new_message("assistant", response))
                self.after(0, self._show_response, response)
            except ValueError as e:
                # stress_word без '+' — не должно случиться после валидации
                # в swarm_classifier, но на всякий случай не валим _process.
                logger.warning(f"learn_stress skipped: {e}")
                self.after(0, self._show_response, "")
            return

        if command == "waiting_event":
            event_prompt = metadata.get("event_prompt") or ""
            delay = metadata.get("delay_minutes") or 0
            if self.event_scheduler and event_prompt and delay > 0:
                kind, old_id, old_src = self.event_scheduler.check_overlap(user_message)
                if kind == "hard":
                    self.event_scheduler.replace(
                        old_id, user_message, event_prompt, delay
                    )
                elif kind == "grey":
                    # Новый триггер ставим сразу — основной ответ не ждёт.
                    # Арбитр в фоне решит, был ли это тот же старый; если да —
                    # отменит старый (новый остаётся активным).
                    new_id = self.event_scheduler.schedule(
                        user_message, event_prompt, delay
                    )
                    threading.Thread(
                        target=self._async_resolve_grey_overlap,
                        args=(old_id, old_src, user_message, new_id),
                        daemon=True,
                    ).start()
                else:
                    self.event_scheduler.schedule(
                        user_message, event_prompt, delay
                    )
            else:
                logger.info(
                    f"waiting_event ignored: "
                    f"prompt='{event_prompt[:40]}', delay={delay}, "
                    f"scheduler={'on' if self.event_scheduler else 'off'}"
                )

        # ── 2. Визуальная память: memorize/recall ──
        # memorize — самодостаточная команда (сохранить и ответить).
        # recall — ищем, формируем контекст и передаём картинку основному LLM.
        # handled=False → команду нельзя выполнить прямо сейчас, main LLM
        #                  должен отреагировать сам (попросить фото / извиниться).
        # handled=True,  vc="" → команда отработала, ничего не нашлось.
        #                      memorize уже сама отрапортовала → return.
        #                      recall без хитов → main LLM должен сказать "не помню".
        # handled=True,  vc!="" → передаём visual_context основному LLM.
        visual_context = ""
        img_path = None
        memorize_no_photo_hint = False
        if metadata.get("visual_command") in ("memorize", "recall"):
            handled, vc, img_path = self._process_visual_command(
                metadata, user_message
            )
            if not handled:
                if metadata.get("visual_command") == "memorize":
                    memorize_no_photo_hint = True
                else:
                    return
            else:
                if vc:
                    visual_context = vc
                elif metadata.get("visual_command") == "memorize":
                    return

        if self.event_scheduler:
            # Не time-event, но возможно пользователь упомянул уже назначенное
            # событие как свершившееся («мама приехала»). Если семантически
            # совпадает с активным триггером — отменяем, чтобы Элеонора не
            # спросила про то, о чём пользователь уже сам рассказал.
            kind, old_id, old_src = self.event_scheduler.check_overlap(user_message)
            if kind == "hard":
                self.event_scheduler.cancel(old_id)
            elif kind == "grey":
                threading.Thread(
                    target=self._async_resolve_grey_cancel,
                    args=(old_id, old_src, user_message),
                    daemon=True,
                ).start()

        # ── 3. Поиск в памяти ──
        # Идёт для ЛЮБОЙ команды где есть user_hint (waiting_event тоже), чтобы
        # Элеонора при "мама приедет через час" видела факты о маме и её ответ
        # (как немедленный, так и отложенный триггер) был связным. Единственное
        # что нас тут отсекает — user_hint = None от свары (no_topic / silence_*),
        # тогда и искать нечего.
        memories_context = ""
        if self.embedder and self.vector_store:
            user_hint = metadata.get("user_hint")

            # Юзер-память — только если sw сказал "надо"
            if user_hint:
                self.after(0, self._set_status, "Ищу в памяти...")
                memories_context = self._search_memories(user_message, user_hint)
                if memories_context:
                    self.after(
                        0, self._append_message,
                        "", "[Память: найдено воспоминаний]", "memory_tag"
                    )

        # ── 4. Генерация ответа ──
        self.after(0, self._set_status, "Элеонора думает...")
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        system = SYSTEM_PROMPT.format(datetime=now)

        # Добавляем воспоминания в системный промпт
        if memories_context:
            user_name = CONSOLIDATOR_USER_NAME
            system += (
                "\n\n=== ВОСПОМИНАНИЯ ===\n"
                f"Факты и история диалога о пользователе (записаны в третьем лице от имени '{user_name}'):\n"
                f"{memories_context}\n"
                "===================\n"
                f"КРИТИЧЕСКИ ВАЖНО: '{user_name}' в этих фактах — это ТЕКУЩИЙ пользователь, "
                "с которым ты сейчас разговариваешь. Это НЕ третий человек.\n"
                f"- Заменяй '{user_name}' на 'ты/твой/тебя' при обращении\n"
                f"- Другие имена (Юля, Дима, мама, коллега и т.д.) — это РЕАЛЬНЫЕ третьи люди, оставляй их как есть\n"
                "- Не цитируй факты/сообщения дословно, используй их естественно как свои воспоминания\n"
                "- Сообщения из истории диалога — это то что было сказано в прошлых разговорах, используй контекст\n\n"
                "ПРИМЕРЫ правильного преобразования:\n"
                f"- Факт: '{user_name} тренировался на спину 24.03' → 'ты тренировался на спину'\n"
                f"- Факт: 'У Юли, подруги {user_name}, появился шпиц' → 'у твоей подруги Юли появился шпиц'\n"
                f"- Факт: '{user_name} поругался с коллегой Димой' → 'ты поругался с Димой' (Дима остаётся Димой)\n"
                f"- Факт: 'Кошка {user_name} — Жужа' → 'твоя кошка Жужа'"
            )

        # Визуальная память: пользователь сказал "memorize", но фото не прикреплено.
        # Рой часто ловит false positive в духе "хочешь покажу как выглядит моя
        # кошка?" — там нет фото, нужно мягко попросить прикрепить.
        if memorize_no_photo_hint:
            entity = metadata.get("visual_entity") or ""
            entity_hint = f" ({entity})" if entity else ""
            system += (
                "\n\n=== ВИЗУАЛЬНАЯ ПАМЯТЬ ===\n"
                f"Пользователь хочет, чтобы ты запомнила фото{entity_hint}, "
                "но фото ещё не прикреплено. "
                "Ответь на его сообщение по-человечески и попроси прикрепить фото "
                "кнопкой «📎 Прикрепить фото» рядом с полем ввода. "
                "Не используй шаблонные фразы, обратись естественно к контексту.\n"
                "=============================="
            )

        # Визуальная память: текстовые результаты поиска — только если нет реального фото
        # Когда есть photo_to_inject — модель видит картинку сама, текст сбивает
        if visual_context and not img_path:
            system += (
                "\n\n=== ВИЗУАЛЬНАЯ ПАМЯТЬ ===\n"
                "Пользователь спросил про фото из твоей визуальной памяти. "
                "К сожалению, само фото недоступно. Вот что о нём известно:\n"
                f"{visual_context}\n"
                "==============================\n"
                "КРИТИЧЕСКИ ВАЖНО: Опирайся на это описание, но отвечай "
                "естественно, как будто ты сама помнишь эти детали."
            )
            logger.info("Injected visual memory context (text fallback)")

        # Инъекция при средней токсичности (3-6, лёгкие оскорбления)
        if 3 <= weight <= 6:
            system += (
                "\n\n=== ОСТОРОЖНО ===\n"
                "Пользователь проявляет агрессию по отношению к тебе. "
                "Не отвечай агрессией. Будь спокойной и сдержанной. "
                "Скажи что тебе неприятно и предложи сменить тему.\n"
                "================"
            )
            logger.info(f"Injected toxic prompt (weight={weight})")

        # Инъекция при фантазии/абсурде
        is_absurd = not metadata.get("is_real", True)
        if is_absurd:
            system += (
                "\n\n=== КРИТИЧЕСКАЯ ИНСТРУКЦИЯ (ПРИОРИТЕТ НАД ВСЕМ) ===\n"
                "Последнее сообщение пользователя содержит ФИЗИЧЕСКИ НЕВОЗМОЖНОЕ утверждение.\n"
                "ЗАПРЕЩЕНО:\n"
                "- Подыгрывать, развивать фантазию, задавать уточняющие вопросы о деталях выдумки\n"
                "- Притворяться что веришь, проявлять интерес к невозможному событию\n"
                "- Начинать ответ с \"Ого\", \"Круто\", \"Расскажи подробнее\"\n"
                "ОБЯЗАТЕЛЬНО:\n"
                "- Дружелюбно объясни что это физически невозможно\n"
                "- Предположи что пользователь шутит, видел сон, или использует метафору\n"
                "- Пример: \"Драконов не существует в реальности) Может тебе приснилось? Или это метафора?\"\n"
                "==================================================="
            )
            logger.info("Injected reality-check prompt (absurd)")

        context = format_messages_for_llm(self.messages[-HISTORY_MESSAGES:])

        # Визуальный recall: вставляем картинку в последнее user-сообщение
        # Приоритет: img_path (из visual_memory) → self.attached_photo (от юзера)
        image_injected = False
        photo_to_inject = img_path or self.attached_photo
        if photo_to_inject and photo_to_inject.exists():
            try:
                from core.vision_llm import _image_to_data_url
                data_url = _image_to_data_url(photo_to_inject)
                for i in range(len(context) - 1, -1, -1):
                    if context[i]["role"] == "user":
                        context[i]["content"] = [
                            {"type": "text", "text": context[i]["content"]},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ]
                        image_injected = True
                        break
                if image_injected:
                    src = "visual memory" if img_path else "attached photo"
                    logger.info(f"Image injected from {src}: {photo_to_inject.name}")
                    # Подсказка для любого фото — заставляет модель смотреть на картинку
                    # а не полагаться на текстовое описание или контекст диалога
                    system += (
                        "\n\n=== ФОТО ===\n"
                        "К этому сообщению прикреплено фото. "
                        "Посмотри на него ВНИМАТЕЛЬНО. "
                        "Опиши что ты реально видишь на изображении, "
                        "не опирайся на текстовые подсказки из диалога.\n"
                        "=============="
                    )
            except Exception as e:
                logger.warning(f"Failed to inject image: {e}")

        dump_path = dump_llm_context(
            output_dir=DATA_DIR / "debug",
            label="response",
            model_id=self.llm.model_id,
            system_prompt=system,
            messages=context,
            extra={
                "command": command,
                "metadata": metadata,
                "memories_context_chars": len(memories_context),
                "history_messages": HISTORY_MESSAGES,
                "image_injected": image_injected,
            },
        )
        logger.info(f"LLM context dump written: {dump_path}")
        response = self.llm.generate(messages=context, system_prompt=system)

        # Fallback: если multimodal не поддерживается (ответ None), пробуем без картинки
        if image_injected and response is None:
            logger.warning("Vision LLM call failed, retrying without image")
            context = format_messages_for_llm(self.messages[-HISTORY_MESSAGES:])
            response = self.llm.generate(messages=context, system_prompt=system)

        if response:
            response = strip_history_markers(response)
            self.messages.append(new_message("assistant", response))
            logger.info(f"Ответ: {response[:80]}...")
        else:
            response = "Не могу ответить -- LM Studio недоступен."

        # ── 5. Сохраняем сообщения в векторную БД (absurd и toxic не сохраняем) ──
        if not is_absurd and weight < 3:
            user_msg_id = self._gen_msg_id("msg")
            self._save_message_to_vector("user", user_message, user_msg_id)

            if self.consolidator:
                self.consolidator.add_message("user", user_message, msg_id=user_msg_id)

            if response and response != "Не могу ответить -- LM Studio недоступен.":
                asst_msg_id = self._gen_msg_id("msg")
                self._save_message_to_vector("assistant", response, asst_msg_id)

        if is_absurd:
            logger.info(f"Skip saving (absurd): {user_message[:50]}")

        # ── 6. Сохраняем историю на диск ──
        self._save_history()

        self.after(0, self._show_response, response)

    def _search_memories(self, query: str, user_hint: str = None) -> str:
        """
        Поиск воспоминаний: Embed → (факты + история) → Reranker → текст.

        Два независимых поиска в параллели:
        1. personal_facts — строгие личные факты о пользователе
        2. messages — релевантные сообщения из истории диалога
        Результаты объединяются в один контекст, упорядоченный по скору.
        """
        try:
            query_vector = self.embedder.embed_query(query, instruction=user_hint)
            rerank_query = user_hint if user_hint else query

            # ── Поиск фактов ──
            fact_results = self.vector_store.search_facts(query_vector, top_k=SEARCH_TOP_K)
            for mid, ftxt, srcs, conf, dist in fact_results:
                logger.info(f"  fact dist={dist:.4f} conf={conf:.2f}: {ftxt[:80]}")

            fact_filtered = [(mid, ftxt, dist) for mid, ftxt, _, _, dist in fact_results
                             if dist <= SEARCH_MIN_SCORE]

            # ── Поиск в истории ──
            msg_results = self.vector_store.search_messages(query_vector, top_k=SEARCH_TOP_K)
            for mid, role, content, dist in msg_results:
                logger.info(f"  msg role={role} dist={dist:.4f}: {content[:60]}")

            msg_filtered = [(mid, content, dist) for mid, role, content, dist in msg_results
                            if dist <= SEARCH_MIN_SCORE]

            if not fact_filtered and not msg_filtered:
                logger.info("Memory search: no results from either source")
                return ""

            logger.info(f"Memory search: {len(fact_filtered)} facts, {len(msg_filtered)} messages")

            # ── Reranker для фактов ──
            reranked_facts = []
            if fact_filtered:
                if self.reranker and len(fact_filtered) > RERANKER_TOP_N:
                    reranked_facts = self.reranker.rerank(rerank_query, fact_filtered, top_n=RERANKER_TOP_N)
                else:
                    reranked_facts = [(mid, txt, 1.0) for mid, txt, _ in fact_filtered[:RERANKER_TOP_N]]
                reranked_facts = [r for r in reranked_facts if r[2] >= RERANKER_MIN_SCORE]

            # ── Reranker для истории ──
            reranked_msgs = []
            if msg_filtered:
                if self.reranker and len(msg_filtered) > RERANKER_TOP_N:
                    reranked_msgs = self.reranker.rerank(rerank_query, msg_filtered, top_n=RERANKER_TOP_N * 2)
                else:
                    reranked_msgs = [(mid, txt, 1.0) for mid, txt, _ in msg_filtered[:RERANKER_TOP_N * 2]]
                reranked_msgs = [r for r in reranked_msgs if r[2] >= RERANKER_MIN_SCORE]

            if not reranked_facts and not reranked_msgs:
                logger.info("Memory search: no candidates above rerank min-score")
                return ""

            for _, txt, score in reranked_facts:
                logger.info(f"  fact rerank score={score:.4f}: {txt[:80]}")
            for _, txt, score in reranked_msgs:
                logger.info(f"  msg rerank score={score:.4f}: {txt[:60]}")

            # ── Dedup-окно ──
            window = MEMORY_RECENT_WINDOW_USER_MSGS
            combined = []

            for mid, txt, score in reranked_facts + reranked_msgs:
                last = self._shown_memory_ids.get(mid)
                if last is None or (self._user_msg_counter - last) >= window:
                    combined.append((mid, txt, score, "fact" if mid.startswith("fact_") else "msg"))

            if not combined:
                logger.info("Memory dedup: all results shown recently, skipping")
                return ""

            combined.sort(key=lambda x: x[2], reverse=True)

            # ── Форматирование контекста ──
            lines = []
            facts_part = [txt for _, txt, _, kind in combined if kind == "fact"]
            msgs_part = [txt for _, txt, _, kind in combined if kind == "msg"]

            if facts_part:
                lines.append("Факты о пользователе:")
                for ftxt in facts_part:
                    lines.append(f"- {ftxt}")

            if msgs_part:
                if lines:
                    lines.append("")
                lines.append("Релевантные сообщения из истории:")
                for mtxt in msgs_part:
                    lines.append(f"- {mtxt[:200]}")

            context = "\n".join(lines)
            logger.info(f"Memory context ({len(combined)} items):\n{context[:400]}")

            for mid, _, _, _ in combined:
                self._shown_memory_ids[mid] = self._user_msg_counter
            stale_threshold = self._user_msg_counter - MEMORY_RECENT_WINDOW_USER_MSGS * 2
            for mid in [k for k, v in self._shown_memory_ids.items() if v < stale_threshold]:
                del self._shown_memory_ids[mid]

            return context

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return ""

    def _is_question(self, text: str) -> bool:
        """Определяет является ли сообщение вопросом (а не фактом)."""
        text_stripped = text.strip()
        # Заканчивается на вопросительный знак
        if text_stripped.endswith("?"):
            return True
        # Начинается с вопросительных слов
        lower = text_stripped.lower()
        question_starts = (
            "как ", "что ", "кто ", "где ", "когда ", "почему ", "зачем ",
            "сколько ", "какой ", "какая ", "какое ", "какие ",
            "помнишь", "ты знаешь", "ты помнишь", "расскажи ",
        )
        for q in question_starts:
            if lower.startswith(q):
                return True
        return False



    def _tts_waveform_worker(self):
        """Пульсирующая световая волна в сайдбаре во время TTS."""
        try:
            if not self._waveform_canvas:
                self.after(50, self._tts_waveform_worker)
                return

            w = self._waveform_canvas.winfo_width()
            if w < 10:
                self.after(50, self._tts_waveform_worker)
                return

            speaking = self.tts and self.tts.is_speaking

            if speaking:
                self._wave_time += 0.08
                self._wave_fading = 0.0
            elif self._wave_fading < 1.0:
                self._wave_time += 0.04
                self._wave_fading += 0.05
            else:
                self._waveform_canvas.delete("all")
                self.after(50, self._tts_waveform_worker)
                return

            h = 60
            cx, cy = w / 2, h / 2
            amp = h * 0.35 * max(0.0, 1.0 - self._wave_fading)
            pw = 0.8 + 0.2 * math.sin(self._wave_time * 2)  # pulse

            # Перелив цвета #8B5CF6 ↔ #06B6D4
            t = self._wave_time * 0.4
            r = int(139 + (6 - 139) * (0.5 + 0.5 * math.sin(t)))
            g = int(91 + (182 - 91) * (0.5 + 0.5 * math.sin(t + 2.094)))
            b = int(246 + (212 - 246) * (0.5 + 0.5 * math.sin(t + 4.188)))
            color = f"#{r:02x}{g:02x}{b:02x}"

            self._waveform_canvas.delete("all")

            # Нижняя волна (тень)
            bottom = []
            for x_norm in range(0, w, 2):
                y = cy + amp * pw * math.sin(x_norm * 0.04 + self._wave_time * 3) + 6
                bottom.extend([x_norm, y])
            if bottom:
                self._waveform_canvas.create_line(
                    bottom, fill="#1E1E4A", width=5, smooth=True, capstyle="round",
                )

            # Основная волна
            top = []
            for x_norm in range(0, w, 2):
                y = cy + amp * pw * math.sin(x_norm * 0.04 + self._wave_time * 3)
                top.extend([x_norm, y])
            if top:
                self._waveform_canvas.create_line(
                    top, fill=color, width=3, smooth=True, capstyle="round",
                )

            # Верхний акцент
            accent = []
            for x_norm in range(0, w, 3):
                y = cy + amp * pw * math.sin(x_norm * 0.04 + self._wave_time * 3 + 0.5) - 2
                accent.extend([x_norm, y])
            if accent:
                self._waveform_canvas.create_line(
                    accent, fill="#D8B4FE", width=1, smooth=True, capstyle="round",
                )

            self.after(50, self._tts_waveform_worker)
        except Exception:
            logger.exception("TTS waveform worker crashed, restarting")
            self.after(100, self._tts_waveform_worker)

    def _show_response(self, response: str):
        if response:
            # yoficate — только для GUI. История/память/LLM-контекст/TTS
            # получают сырой response (TTS сам ёфицирует внутри speak()).
            self._append_message("Элеонора", yoficate(response), "bot_tag")
            if self.tts:
                # speak() неблокирующий, синтез и воспроизведение в фоне.
                # Новый ответ прервёт текущее воспроизведение автоматически.
                self.tts.speak(response)
        self.input_field.configure(state="normal")
        self.send_btn.configure(state="normal")
        self._set_status("")
        self.input_field.focus()

    # ── Инициативные реплики по таймеру ──

    def _judge_event_overlap(self, msg_a: str, msg_b: str) -> bool:
        """
        LLM-арбитр для серой зоны overlap-детекции. Вызывается из executor'а —
        блокирующий HTTP к LM Studio выполняется параллельно с основным
        ответом, main-thread и _process не тормозит.
        """
        try:
            from core.swarm_classifier import judge_event_overlap
            return judge_event_overlap(
                msg_a, msg_b,
                lm_studio_url=f"{LLM_HOST.rstrip('/')}/v1/chat/completions",
                model_id=SWARM_MODEL_ID,
                timeout=SWARM_TIMEOUT,
            )
        except Exception as e:
            logger.error(f"_judge_event_overlap failed: {e}")
            return False

    def _async_resolve_grey_overlap(
        self, old_id: str, old_src: str, new_text: str, new_id: str,
    ):
        """
        Фоновая ветка для waiting_event grey-зоны.
        Новый триггер уже назначен (new_id). Если арбитр скажет "same" —
        отменяем старый, новый остаётся (де-факто replace). Если "different"
        — оставляем оба, арбитр подтвердил что события не пересекаются.
        """
        try:
            same = self._judge_event_overlap(old_src or "", new_text)
        except Exception as e:
            logger.error(f"async grey-overlap judge failed: {e}")
            return
        if same:
            cancelled = self.event_scheduler.cancel(old_id) if self.event_scheduler else False
            logger.info(
                f"async arbiter: grey-merge new={new_id} old={old_id} "
                f"(cancelled_old={cancelled})"
            )
        else:
            logger.info(
                f"async arbiter: grey-split kept both old={old_id} new={new_id}"
            )

    def _async_resolve_grey_cancel(
        self, old_id: str, old_src: str, new_text: str,
    ):
        """
        Фоновая ветка для ветки «не time-event» grey-зоны.
        Если арбитр скажет "same" — пользователь рассказал о свершившемся
        событии, отменяем старый триггер. Если "different" — ничего не трогаем.
        """
        try:
            same = self._judge_event_overlap(old_src or "", new_text)
        except Exception as e:
            logger.error(f"async grey-cancel judge failed: {e}")
            return
        if same:
            cancelled = self.event_scheduler.cancel(old_id) if self.event_scheduler else False
            logger.info(
                f"async arbiter: grey-cancel old={old_id} (cancelled={cancelled})"
            )
        else:
            logger.info(f"async arbiter: grey-cancel kept old={old_id} (not same)")

    def _on_event_fire(self, event_id: str, prompt: str):
        """
        Callback планировщика: таймер сработал, запускаем генерацию
        инициативной реплики Элеоноры в отдельном потоке.
        Вызывается в main-thread из tkinter.after().
        """
        logger.info(f"Event fired: {event_id} -> {prompt[:80]}")
        threading.Thread(
            target=self._fire_event_thread, args=(prompt,), daemon=True,
        ).start()

    def _fire_event_thread(self, event_trigger_prompt: str):
        """
        Генерация инициативной реплики по инструкции-триггеру.

        Gemma Instruct обучена отвечать на user-реплику в конце контекста —
        если её там нет (а её и не может быть, т.к. пользователь сейчас
        ничего не писал), модель путается и уходит в reasoning без
        финального ответа. Поэтому в конец контекста добавляется
        синтетическая user-реплика с явной инструкцией. В истории диалога
        (self.messages) она не сохраняется — только для вызова generate.

        Реплика-ответ добавляется в messages как assistant, чтобы при
        следующем ходе Элеонора помнила, что сама подняла тему.
        """
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            system = SYSTEM_PROMPT.format(datetime=now)
            system += (
                "\n\n=== ИНИЦИАТИВНАЯ РЕПЛИКА ПО ТАЙМЕРУ ===\n"
                "Сработал отложенный триггер. Пользователь сейчас ничего "
                "не писал — ты сама вспомнила про ранее упомянутое событие "
                "и решила спросить. Следующее сообщение с ролью user — это "
                "внутренняя команда системы, НЕ реплика пользователя; "
                "не цитируй её, не пересказывай, просто выполни как инструкцию "
                "и напиши короткую живую фразу для Серёжи.\n"
                "========================================"
            )
            synthetic_user = (
                f"[Системный триггер, не пользовательская реплика] "
                f"{event_trigger_prompt}"
            )
            context = format_messages_for_llm(self.messages[-HISTORY_MESSAGES:]) + [
                {"role": "user", "content": synthetic_user}
            ]
            dump_path = dump_llm_context(
                output_dir=DATA_DIR / "debug",
                label="event",
                model_id=self.llm.model_id,
                system_prompt=system,
                messages=context,
                extra={
                    "event_trigger_prompt": event_trigger_prompt,
                    "history_messages": HISTORY_MESSAGES,
                },
            )
            logger.info(f"LLM context dump written: {dump_path}")
            response = self.llm.generate(messages=context, system_prompt=system)
            if not response:
                logger.warning(
                    f"Event fire: empty LLM response, skipping. "
                    f"Trigger prompt was: {event_trigger_prompt}"
                )
                return
            response = strip_history_markers(response)

            self.messages.append(new_message("assistant", response))
            self._save_history()
            logger.info(f"Initiative reply: {response[:80]}")
            self.after(0, self._append_message, "Элеонора", response, "bot_tag")
            # Озвучка триггерной реплики — тот же путь что обычный ответ.
            if self.tts:
                self.tts.speak(response)
        except Exception as e:
            logger.error(f"Event fire thread failed: {e}")


if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()
