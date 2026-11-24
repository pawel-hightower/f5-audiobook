"""Punkt wejścia aplikacji desktopowej F5-Audiobook (PySide6)."""

from __future__ import annotations

import os
import sys

# Natywny backend Windows Media Foundation (ten sam silnik audio co Windows Media Player):
# brak problemów z Bluetooth A2DP / 24 kHz mono i brak komunikatów FFmpeg na konsoli.
if sys.platform == "win32":
    os.environ.setdefault("QT_MEDIA_BACKEND", "windows")

from PySide6.QtWidgets import QApplication

from f5_audiobook.config import ensure_dirs
from f5_audiobook.gui.main_window import MainWindow


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #181a1f;
    color: #e6e8ec;
    font-family: "Segoe UI", sans-serif;
    font-size: 10pt;
}
QTabWidget::pane {
    border: 1px solid #2c313c;
    border-radius: 6px;
    background-color: #1e2128;
}
QTabBar::tab {
    background-color: #232731;
    color: #a9b1be;
    padding: 8px 18px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}
QTabBar::tab:selected {
    background-color: #2d6cdf;
    color: #ffffff;
    font-weight: bold;
}
QGroupBox {
    border: 1px solid #2e3440;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #8ab4f8;
}
QPushButton {
    background-color: #2b303b;
    color: #e6e8ec;
    border: 1px solid #3c4352;
    border-radius: 5px;
    padding: 6px 14px;
}
QPushButton:hover {
    background-color: #363d4b;
}
QPushButton:disabled {
    background-color: #20232a;
    border-color: #2b303a;
    color: #5d6573;
}
QPushButton#primaryBtn {
    background-color: #2d6cdf;
    border-color: #3b7cf0;
    color: #ffffff;
    font-weight: bold;
}
QPushButton#primaryBtn:hover {
    background-color: #3b7cf0;
}
QPushButton#primaryBtn:disabled {
    background-color: #202632;
    border-color: #2c3445;
    color: #5c6a82;
    font-weight: normal;
}
QPushButton#dangerBtn {
    background-color: #b82e30;
    border-color: #d64547;
    color: #ffffff;
    font-weight: bold;
}
QPushButton#dangerBtn:hover {
    background-color: #d1383a;
}
QPushButton#dangerBtn:disabled {
    background-color: #20232a;
    border-color: #2b303a;
    color: #5d6573;
    font-weight: normal;
}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #13151a;
    border: 1px solid #323846;
    border-radius: 4px;
    padding: 5px 8px;
    color: #e6e8ec;
}
QTableWidget {
    background-color: #13151a;
    alternate-background-color: #1a1d24;
    gridline-color: #262b36;
    border: 1px solid #2c313c;
    selection-background-color: #2d6cdf;
}
QHeaderView::section {
    background-color: #232731;
    color: #cfd5df;
    padding: 5px;
    border: 1px solid #2c313c;
    font-weight: bold;
}
QProgressBar {
    border: 1px solid #323846;
    border-radius: 5px;
    text-align: center;
    background-color: #13151a;
    height: 22px;
}
QProgressBar::chunk {
    background-color: #2d6cdf;
    border-radius: 4px;
}
"""


def main() -> None:
    ensure_dirs()
    app = QApplication(sys.argv)
    app.setApplicationName("F5-Audiobook")
    app.setStyleSheet(DARK_STYLESHEET)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
