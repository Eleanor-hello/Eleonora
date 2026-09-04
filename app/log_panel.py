# -*- coding: utf-8 -*-
"""Панель логов: живой просмотр работы агентов и модулей прямо в GUI.

Все записи идут через стандартный logging: панель ставит свой handler на
корневой логгер, поэтому сюда попадает ВСЁ — llm, tts, агенты (в т.ч.
из фоновых потоков: сигнал Qt переносит запись в GUI-поток безопасно).
"""

import logging

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import QDockWidget, QPlainTextEdit

from app import theme

_MAX_LINES = 2000  # блоков хранит QPlainTextEdit (старые выкидываются)


class _QtLogHandler(logging.Handler, QObject):
    """logging.Handler → Qt-сигнал (работает из любых потоков)."""

    message = Signal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.message.emit(self.format(record))
        except Exception:
            self.handleError(record)


class LogPanel(QDockWidget):
    """Прикрепляемая панель с текстом лога."""

    def __init__(self, parent=None):
        super().__init__("Логи", parent)
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.hide()

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(_MAX_LINES)
        self._text.setStyleSheet(
            f"QPlainTextEdit {{ background: {theme.BG}; color: {theme.TEXT_MUTED};"
            f" border: none; font-family: Consolas; font-size: 12px; }}"
        )
        self.setWidget(self._text)

        self._handler = _QtLogHandler()
        self._handler.message.connect(self._append)
        root = logging.getLogger()
        root.addHandler(self._handler)
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)

    @Slot(str)
    def _append(self, msg: str) -> None:
        self._text.appendPlainText(msg)
