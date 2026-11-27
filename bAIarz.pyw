"""Cichy punkt wejścia aplikacji bAIarz (uruchamiany przez pythonw.exe bez czarnego okna konsoli)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.chdir(ROOT)
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("QT_MEDIA_BACKEND", "windows")

from f5_audiobook.gui.app import main  # noqa: E402

if __name__ == "__main__":
    main()
