# -*- coding: utf-8 -*-
"""Виджет чата: прокручиваемый список сообщений-пузырей."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from app import theme


class _Bubble(QFrame):
    """Один пузырь сообщения."""

    def __init__(self, text: str, mine: bool, parent=None):
        super().__init__(parent)
        bg = theme.PURPLE if mine else theme.PANEL
        color = "#FFFFFF" if mine else theme.TEXT_MAIN
        self.setObjectName("bubble")
        self.setStyleSheet(
            f"#bubble {{ background: {bg}; border-radius: 14px; padding: 8px; }}"
        )

        lbl = QLabel(text, self)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lbl.setStyleSheet(
            f"background: transparent; color: {color};"
            f"font-family: {theme.FONT_FAMILY}; font-size: 13px;"
        )
        lbl.setMaximumWidth(480)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.addWidget(lbl)

        # Выравнивание: юзер справа, Элеонора слева
        align = Qt.AlignRight if mine else Qt.AlignLeft
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self._align = align

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(min(hint.width(), 520))
        return hint


class ChatWidget(QScrollArea):
    """Область чата с автопрокруткой вниз."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setAlignment(Qt.AlignTop)
        self._layout.setContentsMargins(15, 15, 15, 15)
        self._layout.setSpacing(8)
        self.setWidget(container)

        self.setStyleSheet(
            f"background: {theme.SURFACE};"
            f"border-radius: 14px;"
            f"border: 1px solid {theme.BORDER};"
        )

    def add_message(self, text: str, mine: bool) -> None:
        """Добавить пузырь сообщения и доскроллить вниз."""
        bubble = _Bubble(text, mine, self.widget())
        self._layout.addWidget(bubble)
        if mine:
            self._layout.setAlignment(bubble, Qt.AlignRight)
        else:
            self._layout.setAlignment(bubble, Qt.AlignLeft)

        # Прокрутка вниз после отрисовки
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._scroll_to_bottom)

    def clear_messages(self) -> None:
        """Очистить все сообщения в чате."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
