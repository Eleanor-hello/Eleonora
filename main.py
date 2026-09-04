# -*- coding: utf-8 -*-
"""Eleonora v3 — точка входа."""

import logging
import sys

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from db.database import init_db


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    init_db()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
