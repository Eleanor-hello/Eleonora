# -*- coding: utf-8 -*-
"""Модуль db для доступа к хранилищу чатов."""

from db.database import (
    ChatRepo,
    Message,
    chat_repo,
    get_repo,
    init_db,
)

__all__ = [
    "ChatRepo",
    "Message",
    "chat_repo",
    "get_repo",
    "init_db",
]
