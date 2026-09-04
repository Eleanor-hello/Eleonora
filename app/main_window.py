# -*- coding: utf-8 -*-
"""Главное окно приложения: чат, поле ввода, статус."""

import logging
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app import theme
from app.chat_widget import ChatWidget
from app.log_panel import LogPanel
from config import (
    AGENT_MAX_TOKENS,
    AGENT_MODEL_ID,
    AGENT_TEMPERATURE,
    AGENT_TIMEOUT,
    DATA_DIR,
    ENABLE_TTS,
    HISTORY_MESSAGES,
    LLM_HOST,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY,
    RESPONSE_MAX_TOKENS,
    RESPONSE_MODEL_ID,
    RESPONSE_TEMPERATURE,
    TTS_DEVICE,
    TTS_MAX_CHUNK_CHARS,
    TTS_MODEL_PATH,
    TTS_SAMPLE_RATE,
    TTS_VOICE,
    LLM_TIMEOUT,
)
from db import chat_repo
from llm.client import LLMClient
from llm.worker import ResponseWorker

logger = logging.getLogger(__name__)

# ── Системный промпт Элеоноры ──
SYSTEM_PROMPT = """Ты — Элеонора, живой и любопытный искусственный интеллект.

Правила:
- Отвечай короткими естественными фразами, как в разговоре.
- Пиши в женском роде, без эмодзи.
- Твои ответы озвучиваются голосом: без markdown, таблиц и списков.
- Латиницу, цифры и даты пиши произносимо по-русски
  (Python → Пайтон, 1977 год → тысяча девятьсот семьдесят седьмой год).
- Ты ИИ: не выдумывай у себя тело, семью, биографию. Не сочиняй фактов о себе.
- Если чего-то не помнишь или не знаешь — честно скажи об этом.

Текущие дата и время: {datetime}"""


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Eleonora v3")
        self.resize(950, 700)
        self.setStyleSheet(f"background: {theme.BG};")

        self.llm = LLMClient(
            host=LLM_HOST,
            model_id=RESPONSE_MODEL_ID,
            temperature=RESPONSE_TEMPERATURE,
            max_tokens=RESPONSE_MAX_TOKENS,
            timeout=LLM_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
            retry_delay=LLM_RETRY_DELAY,
        )
        # Отдельный клиент для служебных агентов (короткие ответы)
        self.agent_llm = LLMClient(
            host=LLM_HOST,
            model_id=AGENT_MODEL_ID,
            temperature=AGENT_TEMPERATURE,
            max_tokens=AGENT_MAX_TOKENS,
            timeout=AGENT_TIMEOUT,
            max_retries=1,
            retry_delay=0.5,
        )
        self.worker = None

        # ── TTS ──
        self.tts = None
        self.tts_enabled = True
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
                logger.warning(f"TTS отключён: {e}")

        self._build_ui()
        self._restore_history()

    # ── UI ──

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(15, 15, 15, 15)
        root.setSpacing(10)

        # Панель логов (показывается кнопкой 📋)
        self.log_panel = LogPanel(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_panel)
        self.resizeDocks([self.log_panel], [220], Qt.Vertical)

        # Чат
        self.chat = ChatWidget()
        root.addWidget(self.chat, stretch=1)

        # Строка ввода
        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        # Кнопка логов
        self.log_btn = QPushButton("📋")
        self.log_btn.setFixedSize(44, 44)
        self.log_btn.setCursor(Qt.PointingHandCursor)
        self.log_btn.setToolTip("Показать/скрыть логи агентов")
        self.log_btn.clicked.connect(self._toggle_logs)
        self.log_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.SURFACE}; color: white;"
            f" border: 1px solid {theme.PURPLE}; border-radius: 22px; font-size: 16px; }}"
            f"QPushButton:hover {{ background: {theme.BORDER}; }}"
        )
        input_row.addWidget(self.log_btn)

        # Кнопка звука
        self.mute_btn = QPushButton("🔊" if self.tts_enabled else "🔇")
        self.mute_btn.setFixedSize(44, 44)
        self.mute_btn.setCursor(Qt.PointingHandCursor)
        self.mute_btn.clicked.connect(self._toggle_mute)
        self.mute_btn.setToolTip("Вкл/выкл озвучку")
        if self.tts is None:
            self.mute_btn.setEnabled(False)
            self.mute_btn.setToolTip("TTS недоступен")
        self.mute_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.SURFACE}; color: white;"
            f" border: 1px solid {theme.PURPLE}; border-radius: 22px; font-size: 16px; }}"
            f"QPushButton:hover {{ background: {theme.BORDER}; }}"
            f"QPushButton:disabled {{ border-color: {theme.BORDER}; color: {theme.TEXT_MUTED}; }}"
        )
        input_row.addWidget(self.mute_btn)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Напиши сообщение...")
        self.input_field.setFixedHeight(44)
        self.input_field.returnPressed.connect(self._on_send)
        self.input_field.setStyleSheet(
            f"QLineEdit {{ background: {theme.SURFACE}; border: 1px solid {theme.PURPLE};"
            f" border-radius: 20px; padding: 0 16px; color: {theme.TEXT_MAIN};"
            f" font-family: {theme.FONT_FAMILY}; font-size: 14px; }}"
        )
        input_row.addWidget(self.input_field, stretch=1)

        self.send_btn = QPushButton("✈")
        self.send_btn.setFixedSize(44, 44)
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.clicked.connect(self._on_send)
        self.send_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.PURPLE}; color: white;"
            f" border-radius: 22px; font-size: 16px; border: none; }}"
            f"QPushButton:hover {{ background: {theme.PURPLE_HOVER}; }}"
            f"QPushButton:disabled {{ background: {theme.BORDER}; }}"
        )
        input_row.addWidget(self.send_btn)

        root.addLayout(input_row)

        # Статус
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(
            f"color: {theme.CYAN}; font-family: {theme.FONT_FAMILY};"
            f"font-size: 11px; font-weight: bold;"
        )
        root.addWidget(self.status_label)

    # ── История ──

    def _restore_history(self):
        """Показать последние сообщения из БД."""
        messages = chat_repo.get_recent(HISTORY_MESSAGES)
        for msg in messages:
            self.chat.add_message(msg.content, mine=(msg.role == "user"))
        if messages:
            self._set_status(f"История: {len(messages)} сообщений")

    # ── Отправка ──

    def _on_send(self):
        text = self.input_field.text().strip()
        if not text or self.worker is not None:
            return

        logger.info(f"[чат] пользователь: {text[:80]}")
        self.input_field.clear()
        self.chat.add_message(text, mine=True)
        chat_repo.add_message("user", text)

        self._set_busy(True)
        self._set_status("Элеонора думает...")

        system = SYSTEM_PROMPT.format(
            datetime=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        context = [
            {"role": m.role, "content": m.content}
            for m in chat_repo.get_recent(HISTORY_MESSAGES)
        ]

        self.worker = ResponseWorker(
            self.llm, self.agent_llm, text, context, system, parent=self
        )
        self.worker.finished.connect(self._on_response)
        self.worker.failed.connect(self._on_error)
        self.worker.learned_stress.connect(self._on_stress_learned)
        for sig in (self.worker.finished, self.worker.failed, self.worker.learned_stress):
            sig.connect(self.worker.deleteLater)
        self.worker.start()

    def _on_stress_learned(self, bare_word: str):
        """Выучили новое ударение — отвечаем заготовкой."""
        logger.info(f"[stress_check] сохранено в базу ударений: {bare_word}")
        response = f"Запомнила, буду говорить {bare_word}"
        self.chat.add_message(response, mine=False)
        chat_repo.add_message("assistant", response)
        self._finish()

    def _on_response(self, text: str):
        logger.info(f"[чат] Элеонора: {text[:80]}")
        self.chat.add_message(text, mine=False)
        chat_repo.add_message("assistant", text)
        if self.tts and self.tts_enabled:
            self.tts.speak(text)   # неблокирующий
        self._finish()

    def _on_error(self, message: str):
        logger.error(message)
        self._finish()
        self._set_status(message)

    def _finish(self):
        self.worker = None
        self._set_busy(False)
        self._set_status("")
        self.input_field.setFocus()

    # ── Утилиты ──

    def _set_busy(self, busy: bool):
        self.input_field.setEnabled(not busy)
        self.send_btn.setEnabled(not busy)

    def _set_status(self, text: str):
        self.status_label.setText(text)

    def _toggle_mute(self):
        if self.tts is None:
            return
        self.tts_enabled = not self.tts_enabled
        if not self.tts_enabled:
            self.tts.stop()   # оборвать текущую озвучку
        self.mute_btn.setText("🔊" if self.tts_enabled else "🔇")

    def _toggle_logs(self):
        self.log_panel.setVisible(not self.log_panel.isVisible())

    def closeEvent(self, event):
        if self.tts:
            self.tts.stop()   # иначе winsound доигрывает после закрытия окна
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(3000)
        super().closeEvent(event)
