"""Punkt wejścia aplikacji desktopowej bAIarz (PySide6)."""

from __future__ import annotations

import ctypes
import os
import sys

# Natywny backend Windows Media Foundation (ten sam silnik audio co Windows Media Player):
# brak problemów z Bluetooth A2DP / 24 kHz mono i brak komunikatów FFmpeg na konsoli.
if sys.platform == "win32":
    os.environ.setdefault("QT_MEDIA_BACKEND", "windows")

from PySide6.QtGui import QIcon
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from f5_audiobook.config import (
    APP_ICON_ICO,
    APP_ICON_PNG,
    APP_NAME,
    APP_USER_MODEL_ID,
    WORK_DIR,
    ensure_dirs,
)
from f5_audiobook.gui.main_window import MainWindow

_SINGLE_INSTANCE_KEY = "bAIarz_SingleInstance_v1"


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
QTableWidget, QTreeWidget {
    background-color: #13151a;
    alternate-background-color: #1a1d24;
    gridline-color: #262b36;
    border: 1px solid #2c313c;
    selection-background-color: #2d6cdf;
}
QTreeWidget::item {
    padding: 3px 4px;
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


def _redirect_null_streams_for_pythonw() -> None:
    """Pod pythonw.exe (bez okna konsoli) sys.stdout/stderr są None — przekierowujemy je do work/baiarz.log,
    aby komunikaty PyTorch/Vocos/F5-TTS nigdy nie wywołały błędu zapisu."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        log_path = WORK_DIR / "baiarz.log"
        # Przytnij stary log, jeśli przekroczył 2 MB
        mode = "w" if (log_path.exists() and log_path.stat().st_size > 2_000_000) else "a"
        stream = open(log_path, mode, encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = stream
        if sys.stderr is None:
            sys.stderr = stream
    except Exception:
        devnull = open(os.devnull, "w", encoding="utf-8")
        if sys.stdout is None:
            sys.stdout = devnull
        if sys.stderr is None:
            sys.stderr = devnull


def _load_app_icon() -> QIcon:
    if APP_ICON_ICO.exists():
        return QIcon(str(APP_ICON_ICO))
    if APP_ICON_PNG.exists():
        return QIcon(str(APP_ICON_PNG))
    return QIcon()


def _notify_existing_instance() -> bool:
    """Jeśli bAIarz jest już uruchomiony, wysyła sygnał przywrócenia okna i zwraca True."""
    sock = QLocalSocket()
    sock.connectToServer(_SINGLE_INSTANCE_KEY)
    if sock.waitForConnected(250):
        sock.write(b"SHOW")
        sock.flush()
        sock.waitForBytesWritten(250)
        sock.disconnectFromServer()
        return True
    return False


def _start_single_instance_server(win: MainWindow) -> QLocalServer:
    QLocalServer.removeServer(_SINGLE_INSTANCE_KEY)
    server = QLocalServer(win)

    def _on_new_connection() -> None:
        while server.hasPendingConnections():
            client = server.nextPendingConnection()
            if client is not None:
                client.waitForReadyRead(100)
                client.readAll()
                client.disconnectFromServer()
                client.deleteLater()
        win.showNormal()
        win.raise_()
        win.activateWindow()

    server.newConnection.connect(_on_new_connection)
    server.listen(_SINGLE_INSTANCE_KEY)
    return server


def main() -> None:
    ensure_dirs()
    _redirect_null_streams_for_pythonw()

    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
        except Exception:
            pass

    app = QApplication(sys.argv)
    if _notify_existing_instance():
        sys.exit(0)

    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    icon = _load_app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    app.setStyleSheet(DARK_STYLESHEET)

    win = MainWindow()
    if not icon.isNull():
        win.setWindowIcon(icon)
    _server = _start_single_instance_server(win)  # utrzymuj referencję przy życiu
    win._instance_server = _server  # type: ignore[attr-defined]
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
