"""Zakładka 5: Wbudowany odtwarzacz audiobooków (styl Smart AudioBook Player) z karaoke tekstu na żywo.

Odtwarza płynnie cały rozdział z jednego ciągłego strumienia WAV (bez przerw między zdaniami
i bez ucinania początków zdań przy przyspieszaniu 1.15x–2.0x), z synchronizacją karaoke co do milisekundy.
"""

from __future__ import annotations

import bisect
import json
import shutil
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QCoreApplication, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPixmap
from PySide6.QtMultimedia import QAudioDevice, QAudioOutput, QMediaDevices, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from f5_audiobook.config import WORK_DIR
from f5_audiobook.pipeline.project_db import ChapterRow, ChunkRow, ProjectDB

STATS_FILE = WORK_DIR / "player_stats.json"
CHAPTER_LEAD_IN_MS = 250
AUTO_REWIND_SHORT_MS = 2_000
AUTO_REWIND_LONG_MS = 15_000
AUTO_REWIND_LONG_THRESHOLD_SEC = 3600.0  # <= 1 godzina -> 2 s, > 1 godzina -> 15 s


def compute_auto_rewind_ms(paused_at_epoch: float | None, now_epoch: float | None = None) -> int:
    """Oblicza automatyczne cofnięcie (w ms) przy wznawianiu odsłuchu (styl Smart AudioBook Player).

    - Przerwa <= 1 godzina -> 2 s (2000 ms)
    - Przerwa > 1 godzina  -> 15 s (15000 ms)
    - Brak pauzy (nowa pozycja / ręczny skok) -> 0 ms
    """
    if paused_at_epoch is None:
        return 0
    now = time.time() if now_epoch is None else float(now_epoch)
    elapsed = max(0.0, now - float(paused_at_epoch))
    if elapsed > AUTO_REWIND_LONG_THRESHOLD_SEC:
        return AUTO_REWIND_LONG_MS
    return AUTO_REWIND_SHORT_MS


def _fmt_mmss(ms: int) -> str:
    total_sec = max(0, int(ms // 1000))
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _fmt_duration_human(sec: float) -> str:
    total_sec = max(0, int(round(sec)))
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m} min"
    return f"{s} s"


@dataclass
class _PlayableEntry:
    chunk: ChunkRow
    audio_path: Path
    duration_ms: int
    pause_ms: int
    book_start_ms: int
    book_audio_end_ms: int
    book_end_ms: int
    ch_start_ms: int
    ch_audio_end_ms: int
    ch_end_ms: int


class AudiobookPlayerWidget(QWidget):
    """Wbudowany odtwarzacz audiobooków z ciągłym strumieniem rozdziału, karaoke i statystykami."""

    edit_chunk_requested = Signal(int)  # chunk_id -> przejście do edycji/podglądu Whisper w Zakładce 3
    status_message = Signal(str)

    def __init__(self, parent: QWidget | None = None, stats_path: Path = STATS_FILE) -> None:
        super().__init__(parent)
        self.db: ProjectDB | None = None
        self._stats_path = Path(stats_path)

        # Multimedia (QtMultimedia — zachowuje barwę głosu przy zmianie tempa)
        self._media_devices = QMediaDevices(self)
        self._media_devices.audioOutputsChanged.connect(self._on_audio_outputs_changed)
        self._selected_device_id: bytes = b""  # b"" = automatycznie śledź domyślne urządzenie Windows
        self._audio_out = QAudioOutput(QMediaDevices.defaultAudioOutput(), self)
        self._audio_out.setVolume(1.0)
        self._player = QMediaPlayer(self)
        self._player.setAudioOutput(self._audio_out)
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)
        self._player.errorOccurred.connect(self._on_media_error)

        # Stan globalnej osi czasu
        self._chapters: list[ChapterRow] = []
        self._playable: list[_PlayableEntry] = []
        self._playable_starts_ms: list[int] = []
        self._chunk_id_to_playable_idx: dict[int, int] = {}
        self._chapter_first_playable_idx: dict[int, int] = {}
        self._chapter_playable_indices: dict[int, list[int]] = {}
        self._chapter_duration_ms: dict[int, int] = {}
        self._book_total_ms: int = 0

        # Stan sklejonego ciągłego WAV dla bieżącego rozdziału
        self._stitched_chapter_id: int | None = None
        self._stitched_signature: tuple | None = None
        self._stitched_wav_path: Path | None = None
        self._stitched_starts_ms: list[int] = []  # pozycja każdego chunka w sklejonym WAV (z lead-in)
        self._stitched_playable_indices: list[int] = []
        self._stitched_total_ms: int = 0

        # Stan odtwarzania
        self._current_idx: int = -1
        self._current_chapter_id: int | None = None
        self._current_source_path: Path | None = None
        self._is_playing: bool = False
        self._playback_rate: float = 1.0
        self._muted: bool = False
        self._pending_seek_wav_ms: int | None = None
        self._pending_autoplay: bool = False
        self._paused_ch_ms: int = 0  # dokładna pozycja w rozdziale (bez lead-in), gdy WAV nie jest jeszcze otwarty
        self._slider_dragging: bool = False
        self._karaoke_font_pt: int = 12
        self._karaoke_chunk_to_row: dict[int, int] = {}
        self._karaoke_chapter_title: str = ""
        self._last_highlighted_chunk_id: int | None = None
        self._pending_click_chunk_id: int | None = None

        self._karaoke_click_timer = QTimer(self)
        self._karaoke_click_timer.setSingleShot(True)
        self._karaoke_click_timer.setInterval(260)
        self._karaoke_click_timer.timeout.connect(self._on_karaoke_single_click_timeout)

        # Statystyki słuchania
        self._global_stats: dict = self._load_global_stats()
        self._book_listened_sec: float = 0.0
        self._session_listened_sec: float = 0.0
        self._last_tick_mono: float | None = None
        self._last_persist_mono: float = time.monotonic()
        self._last_paused_epoch: float | None = None

        # Timer odświeżania UI i śledzenia karaoke (100 ms -> płynne podświetlanie)
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(100)
        self._ui_timer.timeout.connect(self._on_ui_tick)
        self._ui_timer.start()

        self._build_ui()
        self._update_controls_enabled()
        self._update_stats_display()

    # =========================================================================
    # INTERFEJS (PIONOWY PODZIAŁ 50/50: STEROWANIE | KARAOKE TEKSTU)
    # =========================================================================
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)

        # ----- LEWA POŁOWA: STEROWANIE AUDIOBOOKIEM -----
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 8, 4)
        left_lay.setSpacing(10)

        header_box = QGroupBox("Teraz słuchasz")
        header_lay = QHBoxLayout(header_box)

        self.lbl_cover = QLabel("Brak\nokładki")
        self.lbl_cover.setFixedSize(92, 130)
        self.lbl_cover.setAlignment(Qt.AlignCenter)
        self.lbl_cover.setStyleSheet("border: 1px solid #323846; background: #13151a; color: #7d8590;")
        header_lay.addWidget(self.lbl_cover)

        title_col = QVBoxLayout()
        title_col.setSpacing(5)
        self.lbl_title = QLabel("Nie wczytano projektu")
        self.lbl_title.setStyleSheet("font-size: 13pt; font-weight: bold; color: #f0f4f8;")
        self.lbl_title.setWordWrap(True)
        self.lbl_author = QLabel("—")
        self.lbl_author.setStyleSheet("font-size: 10pt; color: #9aa3b2;")

        ch_nav_row = QHBoxLayout()
        ch_nav_row.setSpacing(4)
        self.btn_prev_chapter = QPushButton("⟨")
        self.btn_prev_chapter.setFixedWidth(30)
        self.btn_prev_chapter.setToolTip("Poprzedni rozdział")
        self.btn_prev_chapter.clicked.connect(self.prev_chapter)

        self.combo_chapters = QComboBox()
        self.combo_chapters.currentIndexChanged.connect(self._on_chapter_combo_changed)

        self.btn_next_chapter = QPushButton("⟩")
        self.btn_next_chapter.setFixedWidth(30)
        self.btn_next_chapter.setToolTip("Następny rozdział")
        self.btn_next_chapter.clicked.connect(self.next_chapter)

        ch_nav_row.addWidget(self.btn_prev_chapter)
        ch_nav_row.addWidget(self.combo_chapters, 1)
        ch_nav_row.addWidget(self.btn_next_chapter)

        self.lbl_ready_info = QLabel("Wygenerowane fragmenty do odsłuchu: 0")
        self.lbl_ready_info.setStyleSheet("color: #8ab4f8; font-size: 9pt;")

        title_col.addWidget(self.lbl_title)
        title_col.addWidget(self.lbl_author)
        title_col.addLayout(ch_nav_row)
        title_col.addWidget(self.lbl_ready_info)
        title_col.addStretch()
        header_lay.addLayout(title_col, 1)
        left_lay.addWidget(header_box)

        # Postęp rozdziału + książki
        progress_box = QGroupBox("Pozycja i sterowanie")
        prog_lay = QVBoxLayout(progress_box)
        prog_lay.setSpacing(8)

        self.slider_chapter = QSlider(Qt.Horizontal)
        self.slider_chapter.setRange(0, 1000)
        self.slider_chapter.setValue(0)
        self.slider_chapter.sliderPressed.connect(self._on_slider_pressed)
        self.slider_chapter.sliderMoved.connect(self._on_slider_moved)
        self.slider_chapter.sliderReleased.connect(self._on_slider_released)
        prog_lay.addWidget(self.slider_chapter)

        time_row = QHBoxLayout()
        self.lbl_ch_elapsed = QLabel("00:00")
        self.lbl_ch_elapsed.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
        self.lbl_ch_pct = QLabel("Rozdział: 0%")
        self.lbl_ch_pct.setAlignment(Qt.AlignCenter)
        self.lbl_ch_pct.setStyleSheet("color: #9aa3b2; font-size: 9pt;")
        self.lbl_ch_remaining = QLabel("(-00:00)")
        self.lbl_ch_remaining.setStyleSheet("color: #8ab4f8; font-family: Consolas, monospace; font-size: 9.5pt;")
        self.lbl_ch_remaining.setToolTip("Czas pozostały do końca bieżącego rozdziału")
        self.lbl_ch_total = QLabel("00:00")
        self.lbl_ch_total.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
        time_row.addWidget(self.lbl_ch_elapsed)
        time_row.addWidget(self.lbl_ch_pct, 1)
        time_row.addWidget(self.lbl_ch_remaining)
        time_row.addSpacing(6)
        time_row.addWidget(self.lbl_ch_total)
        prog_lay.addLayout(time_row)

        transport_row = QHBoxLayout()
        transport_row.setSpacing(6)

        self.btn_back_60 = QPushButton("⏮ -1m")
        self.btn_back_60.setMinimumHeight(40)
        self.btn_back_60.setToolTip("Cofnij o 1 minutę (Shift+← lub Ctrl+Shift+←)")
        self.btn_back_60.clicked.connect(lambda: self.seek_relative_ms(-60_000))

        self.btn_back_10 = QPushButton("⏪ -10s")
        self.btn_back_10.setMinimumHeight(40)
        self.btn_back_10.setToolTip("Cofnij o 10 sekund (← lub Ctrl+←)")
        self.btn_back_10.clicked.connect(lambda: self.seek_relative_ms(-10_000))

        self.btn_play_pause = QPushButton("▶ Odtwórz")
        self.btn_play_pause.setObjectName("primaryBtn")
        self.btn_play_pause.setMinimumHeight(46)
        self.btn_play_pause.setMinimumWidth(130)
        self.btn_play_pause.setStyleSheet("font-size: 11pt; font-weight: bold;")
        self.btn_play_pause.setToolTip("Odtwórz / Pauza (Spacja w zakładce Odtwarzacz lub Ctrl+Spacja wszędzie)")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)

        self.btn_fwd_10 = QPushButton("⏩ +10s")
        self.btn_fwd_10.setMinimumHeight(40)
        self.btn_fwd_10.setToolTip("Do przodu o 10 sekund (→ lub Ctrl+→)")
        self.btn_fwd_10.clicked.connect(lambda: self.seek_relative_ms(10_000))

        self.btn_fwd_60 = QPushButton("⏭ +1m")
        self.btn_fwd_60.setMinimumHeight(40)
        self.btn_fwd_60.setToolTip("Do przodu o 1 minutę (Shift+→ lub Ctrl+Shift+→)")
        self.btn_fwd_60.clicked.connect(lambda: self.seek_relative_ms(60_000))

        transport_row.addWidget(self.btn_back_60)
        transport_row.addWidget(self.btn_back_10)
        transport_row.addWidget(self.btn_play_pause, 1)
        transport_row.addWidget(self.btn_fwd_10)
        transport_row.addWidget(self.btn_fwd_60)
        prog_lay.addLayout(transport_row)

        book_prog_row = QHBoxLayout()
        book_prog_row.addWidget(QLabel("Książka:"))
        self.bar_book_progress = QProgressBar()
        self.bar_book_progress.setRange(0, 1000)
        self.bar_book_progress.setValue(0)
        self.bar_book_progress.setFixedHeight(14)
        self.bar_book_progress.setFormat("")
        book_prog_row.addWidget(self.bar_book_progress, 1)
        self.lbl_book_time = QLabel("00:00 / 00:00")
        self.lbl_book_time.setStyleSheet("color: #9aa3b2; font-size: 9pt; font-family: Consolas, monospace;")
        book_prog_row.addWidget(self.lbl_book_time)
        prog_lay.addLayout(book_prog_row)

        left_lay.addWidget(progress_box)

        # Tempo, głośność i wyjście audio
        audio_box = QGroupBox("Tempo i głośność")
        audio_lay = QVBoxLayout(audio_box)
        audio_lay.setSpacing(8)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Tempo:"))
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.60, 2.50)
        self.spin_speed.setSingleStep(0.05)
        self.spin_speed.setDecimals(2)
        self.spin_speed.setSuffix("x")
        self.spin_speed.setValue(1.00)
        self.spin_speed.setFixedWidth(82)
        self.spin_speed.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self.spin_speed)

        for preset in (1.0, 1.15, 1.25, 1.5, 1.75):
            btn_p = QPushButton(f"{preset:g}x")
            btn_p.setFixedWidth(48)
            btn_p.setToolTip(f"Ustaw tempo {preset:g}x")
            btn_p.clicked.connect(lambda _chk=False, v=preset: self.set_playback_rate(v))
            speed_row.addWidget(btn_p)
        speed_row.addStretch()
        audio_lay.addLayout(speed_row)

        vol_row = QHBoxLayout()
        self.btn_mute = QPushButton("🔊")
        self.btn_mute.setFixedWidth(36)
        self.btn_mute.setToolTip("Wycisz / Odcisz")
        self.btn_mute.clicked.connect(self._on_toggle_mute)
        self.slider_volume = QSlider(Qt.Horizontal)
        self.slider_volume.setRange(0, 100)
        self.slider_volume.setValue(100)
        self.slider_volume.valueChanged.connect(self._on_volume_changed)
        self.lbl_volume = QLabel("100%")
        self.lbl_volume.setFixedWidth(40)
        self.lbl_volume.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        vol_row.addWidget(QLabel("Głośność:"))
        vol_row.addWidget(self.btn_mute)
        vol_row.addWidget(self.slider_volume, 1)
        vol_row.addWidget(self.lbl_volume)
        audio_lay.addLayout(vol_row)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Wyjście audio:"))
        self.combo_audio_device = QComboBox()
        self.combo_audio_device.setToolTip(
            "Wybierz urządzenie audio (domyślnie automatycznie przełącza się za Windows, np. na słuchawki Bluetooth)"
        )
        self.combo_audio_device.currentIndexChanged.connect(self._on_audio_device_combo_changed)
        dev_row.addWidget(self.combo_audio_device, 1)
        audio_lay.addLayout(dev_row)
        self._refresh_audio_devices()

        left_lay.addWidget(audio_box)

        # Statystyki słuchania
        stats_box = QGroupBox("Statystyki słuchania")
        stats_lay = QVBoxLayout(stats_box)
        stats_lay.setSpacing(4)
        self.lbl_stats_main = QLabel("Dzisiaj: 0 s  •  Ta książka: 0 s  •  Łącznie: 0 s")
        self.lbl_stats_main.setStyleSheet("font-size: 10pt; font-weight: bold; color: #e2e8f0;")
        self.lbl_stats_sub = QLabel("Pozostało do końca gotowego audio: —")
        self.lbl_stats_sub.setStyleSheet("font-size: 9pt; color: #9aa3b2;")
        stats_lay.addWidget(self.lbl_stats_main)
        stats_lay.addWidget(self.lbl_stats_sub)
        left_lay.addWidget(stats_box)

        # Podpowiedź skrótów klawiszowych
        shortcuts_frame = QFrame()
        shortcuts_frame.setStyleSheet(
            "QFrame { background-color: #161922; border: 1px solid #2b3140; border-radius: 6px; padding: 6px; }"
        )
        sc_lay = QVBoxLayout(shortcuts_frame)
        sc_lay.setContentsMargins(8, 6, 8, 6)
        lbl_sc = QLabel(
            "<b>Skróty klawiszowe w aplikacji:</b><br>"
            "• <b>Ctrl+Spacja</b> (lub <b>Spacja</b> tutaj) — Odtwórz / Pauza &nbsp;|&nbsp; "
            "<b>[</b> / <b>]</b> — wolniej / szybciej<br>"
            "• <b>Ctrl+← / Ctrl+→</b> (lub <b>← / →</b> tutaj) — -10s / +10s &nbsp;|&nbsp; "
            "<b>Shift+← / Shift+→</b> — -1 min / +1 min<br>"
            "• <b>Kliknięcie</b> w zdanie po prawej skacze w audio • <b>Dwuklik</b> przenosi do tego zdania w Zakładce 3."
        )
        lbl_sc.setWordWrap(True)
        lbl_sc.setStyleSheet("color: #9aa3b2; font-size: 8.5pt; border: none; background: transparent;")
        sc_lay.addWidget(lbl_sc)
        left_lay.addWidget(shortcuts_frame)

        left_lay.addStretch()

        # ----- PRAWA POŁOWA: KARAOKE / SYNCHRONIZOWANY TEKST -----
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 4, 4, 4)
        right_lay.setSpacing(6)

        text_top = QHBoxLayout()
        self.lbl_karaoke_header = QLabel("Tekst rozdziału (klik — odtwórz | dwuklik — Zakładka 3)")
        self.lbl_karaoke_header.setStyleSheet("font-weight: bold; color: #cfd5df;")
        text_top.addWidget(self.lbl_karaoke_header, 1)

        self.chk_autoscroll = QCheckBox("Śledź tekst")
        self.chk_autoscroll.setChecked(True)
        text_top.addWidget(self.chk_autoscroll)

        btn_font_minus = QPushButton("A-")
        btn_font_minus.setFixedWidth(32)
        btn_font_minus.setToolTip("Zmniejsz czcionkę tekstu")
        btn_font_minus.clicked.connect(lambda: self._change_karaoke_font(-1))
        btn_font_plus = QPushButton("A+")
        btn_font_plus.setFixedWidth(32)
        btn_font_plus.setToolTip("Powiększ czcionkę tekstu")
        btn_font_plus.clicked.connect(lambda: self._change_karaoke_font(+1))
        text_top.addWidget(btn_font_minus)
        text_top.addWidget(btn_font_plus)
        right_lay.addLayout(text_top)

        self.list_karaoke = QListWidget()
        self.list_karaoke.setWordWrap(True)
        self.list_karaoke.setResizeMode(QListView.Adjust)
        self.list_karaoke.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list_karaoke.setSelectionMode(QAbstractItemView.NoSelection)
        self.list_karaoke.setSpacing(3)
        self.list_karaoke.setStyleSheet(
            """
            QListWidget {
                background-color: #13161e;
                border: 1px solid #2b3140;
                border-radius: 6px;
                padding: 8px;
                outline: none;
            }
            QListWidget::item {
                padding: 7px 10px;
                border-radius: 5px;
            }
            QListWidget::item:hover {
                background-color: #1e2433;
            }
            """
        )
        self.list_karaoke.itemClicked.connect(self._on_karaoke_item_clicked)
        self.list_karaoke.itemDoubleClicked.connect(self._on_karaoke_item_double_clicked)
        right_lay.addWidget(self.list_karaoke, 1)

        text_bottom = QHBoxLayout()
        self.btn_refresh_audio = QPushButton("🔄 Odśwież nowe fragmenty")
        self.btn_refresh_audio.setToolTip("Wczytuje świeżo wygenerowane zdania z Zakładki 3 bez przerywania słuchania")
        self.btn_refresh_audio.clicked.connect(self.refresh_new_chunks)

        self.btn_fix_current = QPushButton("🔧 Popraw bieżący fragment (Zakładka 3)")
        self.btn_fix_current.setToolTip(
            "Zatrzymuje odtwarzanie i przechodzi dokładnie do tego zdania w Zakładce 3 (działa też dwuklik w tekście)"
        )
        self.btn_fix_current.clicked.connect(self._on_fix_current_clicked)

        text_bottom.addWidget(self.btn_refresh_audio)
        text_bottom.addStretch()
        text_bottom.addWidget(self.btn_fix_current)
        right_lay.addLayout(text_bottom)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([540, 620])
        root.addWidget(splitter, 1)

    # =========================================================================
    # ZARZĄDZANIE PROJEKTEM, OŚ CZASU I SKLEJANIE ROZDZIAŁU
    # =========================================================================
    def set_project(self, db: ProjectDB | None) -> None:
        self.persist_state()
        self.release_media(clean_tmp=True)
        self.db = db
        self._current_idx = -1
        self._current_chapter_id = None
        self._last_highlighted_chunk_id = None
        self._book_listened_sec = 0.0
        self._session_listened_sec = 0.0
        self._paused_ch_ms = 0
        self._last_paused_epoch = None
        self._karaoke_chunk_to_row.clear()
        self.list_karaoke.clear()

        if db is None:
            self._chapters = []
            self._playable = []
            self._playable_starts_ms = []
            self._chunk_id_to_playable_idx.clear()
            self._chapter_first_playable_idx.clear()
            self._chapter_playable_indices.clear()
            self._chapter_duration_ms.clear()
            self._book_total_ms = 0
            self.lbl_title.setText("Nie wczytano projektu")
            self.lbl_author.setText("—")
            self.lbl_cover.setPixmap(QPixmap())
            self.lbl_cover.setText("Brak\nokładki")
            self.combo_chapters.blockSignals(True)
            self.combo_chapters.clear()
            self.combo_chapters.blockSignals(False)
            self.lbl_ready_info.setText("Wygenerowane fragmenty do odsłuchu: 0")
            self._update_controls_enabled()
            self._update_time_labels()
            self._update_stats_display()
            return

        meta = db.get_all_meta()
        self._refresh_book_header(meta)
        try:
            self._book_listened_sec = float(meta.get("player_listened_sec", "0") or 0.0)
        except ValueError:
            self._book_listened_sec = 0.0
        try:
            saved_speed = float(meta.get("player_speed", "1.0") or 1.0)
            saved_speed = max(0.60, min(2.50, saved_speed))
        except ValueError:
            saved_speed = 1.0

        self._playback_rate = saved_speed
        self.spin_speed.blockSignals(True)
        self.spin_speed.setValue(saved_speed)
        self.spin_speed.blockSignals(False)

        self._rebuild_timeline()

        saved_chunk_id = int(meta["player_chunk_id"]) if meta.get("player_chunk_id", "").isdigit() else None
        saved_offset_ms = int(meta["player_offset_ms"]) if meta.get("player_offset_ms", "").isdigit() else 0

        if self._playable:
            start_idx = 0
            if saved_chunk_id is not None and saved_chunk_id in self._chunk_id_to_playable_idx:
                start_idx = self._chunk_id_to_playable_idx[saved_chunk_id]
            else:
                saved_offset_ms = 0
            self._load_playable_index(start_idx, offset_ms=saved_offset_ms, autoplay=False)

            paused_ts_raw = meta.get("player_paused_at_ts", "").strip()
            if paused_ts_raw:
                try:
                    self._last_paused_epoch = float(paused_ts_raw)
                except ValueError:
                    self._last_paused_epoch = None
            elif start_idx > 0 or saved_offset_ms > 0:
                # Istniejący projekt zapisany przed tą funkcją -> przy pierwszym wznowieniu potraktuj jak dłuższą przerwę (15 s)
                self._last_paused_epoch = 0.0
            else:
                self._last_paused_epoch = None
        else:
            self._last_paused_epoch = None
            self._update_controls_enabled()
            self._update_time_labels()

        self._update_stats_display()

    def refresh_metadata(self) -> None:
        if self.db is None:
            return
        self._refresh_book_header(self.db.get_all_meta())

    def _refresh_book_header(self, meta: dict[str, str]) -> None:
        title = meta.get("title") or (self.db.project_dir.name if self.db else "")
        author = meta.get("author") or "Nieznany autor"
        self.lbl_title.setText(title)
        self.lbl_author.setText(author)

        cover_path = Path(meta["cover_path"]) if meta.get("cover_path") else None
        if cover_path and cover_path.exists():
            pix = QPixmap(str(cover_path)).scaled(92, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_cover.setPixmap(pix)
        else:
            self.lbl_cover.setPixmap(QPixmap())
            self.lbl_cover.setText("Brak\nokładki")

    def refresh_new_chunks(self) -> None:
        """Doładowuje nowo wygenerowane chunki z bazy bez przerywania bieżącego odtwarzania."""
        if self.db is None:
            return
        cur_chunk_id = self._playable[self._current_idx].chunk.id if 0 <= self._current_idx < len(self._playable) else None
        old_count = len(self._playable)

        self._rebuild_timeline()

        if cur_chunk_id is not None and cur_chunk_id in self._chunk_id_to_playable_idx:
            self._current_idx = self._chunk_id_to_playable_idx[cur_chunk_id]
            # Zaktualizuj mapowanie indeksów sklejonego rozdziału na wypadek gdyby doszły wcześniejsze chunki
            if self._stitched_chapter_id is not None and self._stitched_chapter_id == self._current_chapter_id:
                new_indices = self._chapter_playable_indices.get(self._current_chapter_id, [])
                if len(new_indices) == len(self._stitched_playable_indices):
                    self._stitched_playable_indices = list(new_indices)
            if self._current_chapter_id is not None:
                self._populate_karaoke_for_chapter(self._current_chapter_id, highlight_chunk_id=cur_chunk_id)
        elif self._playable and self._current_idx < 0:
            self._load_playable_index(0, offset_ms=0, autoplay=False)

        self._update_controls_enabled()
        self._update_time_labels()
        self._update_stats_display()

        diff = len(self._playable) - old_count
        if diff > 0:
            self.status_message.emit(f"Odtwarzacz: dodano {diff} nowych gotowych fragmentów")

    def _rebuild_timeline(self) -> None:
        if self.db is None:
            return
        self._chapters = self.db.list_chapters()
        all_chunks = self.db.list_chunks(only_selected=False)

        playable: list[_PlayableEntry] = []
        starts_ms: list[int] = []
        id_to_idx: dict[int, int] = {}
        ch_first_idx: dict[int, int] = {}
        ch_indices: dict[int, list[int]] = {}
        ch_dur: dict[int, int] = {}

        book_cursor = 0
        ch_cursors: dict[int, int] = {}

        for c in all_chunks:
            if c.status != "done":
                continue
            ap = self.db.resolve_chunk_audio_path(c)
            if ap is None:
                continue
            dur_ms = max(200, int(round(c.duration_sec * 1000))) if c.duration_sec > 0 else 1000
            pause_ms = max(0, int(c.pause_after_ms))
            ch_start = ch_cursors.get(c.chapter_id, 0)

            entry = _PlayableEntry(
                chunk=c,
                audio_path=ap,
                duration_ms=dur_ms,
                pause_ms=pause_ms,
                book_start_ms=book_cursor,
                book_audio_end_ms=book_cursor + dur_ms,
                book_end_ms=book_cursor + dur_ms + pause_ms,
                ch_start_ms=ch_start,
                ch_audio_end_ms=ch_start + dur_ms,
                ch_end_ms=ch_start + dur_ms + pause_ms,
            )
            idx = len(playable)
            playable.append(entry)
            starts_ms.append(book_cursor)
            id_to_idx[c.id] = idx
            if c.chapter_id not in ch_first_idx:
                ch_first_idx[c.chapter_id] = idx
            ch_indices.setdefault(c.chapter_id, []).append(idx)

            book_cursor += dur_ms + pause_ms
            ch_cursors[c.chapter_id] = ch_start + dur_ms + pause_ms
            ch_dur[c.chapter_id] = ch_cursors[c.chapter_id]

        self._playable = playable
        self._playable_starts_ms = starts_ms
        self._chunk_id_to_playable_idx = id_to_idx
        self._chapter_first_playable_idx = ch_first_idx
        self._chapter_playable_indices = ch_indices
        self._chapter_duration_ms = ch_dur
        self._book_total_ms = book_cursor

        self.combo_chapters.blockSignals(True)
        self.combo_chapters.clear()
        ready_chapters = [ch for ch in self._chapters if ch.id in ch_first_idx]
        for ch in ready_chapters:
            dur_str = _fmt_mmss(ch_dur.get(ch.id, 0))
            label = f"{ch.chapter_index + 1}. {ch.title}  ({ch.done_chunks}/{ch.total_chunks} • {dur_str})"
            self.combo_chapters.addItem(label, ch.id)
        if self._current_chapter_id is not None:
            for i in range(self.combo_chapters.count()):
                if self.combo_chapters.itemData(i) == self._current_chapter_id:
                    self.combo_chapters.setCurrentIndex(i)
                    break
        self.combo_chapters.blockSignals(False)

        self.lbl_ready_info.setText(
            f"Gotowe do słuchania: {len(ready_chapters)} rozdz. • {len(playable)} fragm. "
            f"(łącznie {_fmt_mmss(self._book_total_ms)})"
        )

    def _chapter_signature(self, chapter_id: int) -> tuple:
        indices = self._chapter_playable_indices.get(chapter_id, [])
        sig = []
        for idx in indices:
            e = self._playable[idx]
            try:
                st = e.audio_path.stat()
                sig.append((e.chunk.id, st.st_mtime_ns, st.st_size))
            except OSError:
                sig.append((e.chunk.id, 0, 0))
        return tuple(sig)

    def _ensure_chapter_stitched(self, chapter_id: int) -> Path | None:
        """Skleja gotowe fragmenty rozdziału w jeden ciągły plik WAV (PCM 16-bit) z 250 ms lead-in."""
        if self.db is None:
            return None
        indices = self._chapter_playable_indices.get(chapter_id, [])
        if not indices:
            return None

        sig = self._chapter_signature(chapter_id)
        if (
            self._stitched_chapter_id == chapter_id
            and self._stitched_signature == sig
            and self._stitched_wav_path is not None
            and self._stitched_wav_path.exists()
        ):
            return self._stitched_wav_path

        sr = 24000
        lead_samples = int(sr * CHAPTER_LEAD_IN_MS / 1000)
        parts: list[np.ndarray] = [np.zeros(lead_samples, dtype=np.int16)]
        sample_cursor = lead_samples

        starts_ms: list[int] = []
        playable_indices: list[int] = []
        ch_cursor_no_lead_ms = 0

        for idx in indices:
            e = self._playable[idx]
            if not e.audio_path.exists():
                continue
            w, file_sr = sf.read(str(e.audio_path), dtype="int16")
            sr = file_sr
            if w.ndim == 2:
                w = w.mean(axis=1).astype(np.int16)

            wav_start_ms = int(round(sample_cursor * 1000.0 / sr))
            dur_ms = max(50, int(round(len(w) * 1000.0 / sr)))
            pause_samples = int(round(sr * max(0, e.pause_ms) / 1000.0))
            pause_ms = int(round(pause_samples * 1000.0 / sr))

            parts.append(w)
            sample_cursor += len(w)
            if pause_samples > 0:
                parts.append(np.zeros(pause_samples, dtype=np.int16))
                sample_cursor += pause_samples

            # Zaktualizuj dokładne czasy rozdziałowe co do próbki (idealna synchronizacja karaoke)
            e.ch_start_ms = ch_cursor_no_lead_ms
            e.ch_audio_end_ms = ch_cursor_no_lead_ms + dur_ms
            e.ch_end_ms = ch_cursor_no_lead_ms + dur_ms + pause_ms
            ch_cursor_no_lead_ms = e.ch_end_ms

            starts_ms.append(wav_start_ms)
            playable_indices.append(idx)

        if not playable_indices:
            return None

        # Końcowa pauza rozdziału (300 ms) aby płynnie domknąć ostatnią głoskę
        parts.append(np.zeros(int(sr * 0.30), dtype=np.int16))
        combined = np.concatenate(parts)

        tmp_dir = self.db.project_dir / "_player_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_path = tmp_dir / f"ch_{chapter_id}_{int(time.time() * 1000) % 1_000_000}.wav"
        sf.write(str(out_path), combined, sr, subtype="PCM_16")

        # Usuń stare nieużywane tymczasowe pliki rozdziałów (bez ruszania aktualnego źródła)
        for old_f in tmp_dir.glob("ch_*.wav"):
            if old_f != out_path and old_f != self._current_source_path:
                old_f.unlink(missing_ok=True)

        self._stitched_chapter_id = chapter_id
        self._stitched_signature = sig
        self._stitched_wav_path = out_path
        self._stitched_starts_ms = starts_ms
        self._stitched_playable_indices = playable_indices
        self._stitched_total_ms = ch_cursor_no_lead_ms
        self._chapter_duration_ms[chapter_id] = ch_cursor_no_lead_ms
        return out_path

    def release_media(self, clean_tmp: bool = False) -> None:
        """Zatrzymuje odtwarzanie i zwalnia uchwyt pliku WAV w systemie Windows."""
        self._karaoke_click_timer.stop()
        self._pending_click_chunk_id = None
        self._is_playing = False
        self._pending_autoplay = False
        self._pending_seek_wav_ms = None
        self._last_tick_mono = None
        self._player.stop()
        self._player.setSource(QUrl())
        self._current_source_path = None
        QCoreApplication.processEvents()
        self.btn_play_pause.setText("▶ Odtwórz")

        if clean_tmp and self.db is not None:
            tmp_dir = self.db.project_dir / "_player_tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            self._stitched_chapter_id = None
            self._stitched_signature = None
            self._stitched_wav_path = None
            self._stitched_starts_ms = []
            self._stitched_playable_indices = []

    def pause_playback(self) -> None:
        """Pauzuje odtwarzanie (np. przy podglądzie z Zakładki 3 lub dwukliku w karaoke)."""
        if not self._is_playing:
            return
        self._paused_ch_ms = self.current_chapter_ms()
        self._is_playing = False
        self._pending_autoplay = False
        self._last_tick_mono = None
        self._last_paused_epoch = time.time()
        self._player.pause()
        self.btn_play_pause.setText("▶ Odtwórz")
        self.persist_state()

    # =========================================================================
    # ODTWARZANIE, PRZEWIJANIE I KARAOKE
    # =========================================================================
    def _load_playable_index(self, idx: int, offset_ms: int = 0, autoplay: bool | None = None) -> None:
        if not (0 <= idx < len(self._playable)):
            return
        should_play = self._is_playing if autoplay is None else autoplay
        self._current_idx = idx
        entry = self._playable[idx]

        chapter_changed = entry.chunk.chapter_id != self._current_chapter_id
        if chapter_changed:
            self._current_chapter_id = entry.chunk.chapter_id
            self._sync_chapter_combo(self._current_chapter_id)
            self._populate_karaoke_for_chapter(self._current_chapter_id, highlight_chunk_id=entry.chunk.id)
        else:
            self._highlight_karaoke_chunk(entry.chunk.id)

        clamped_offset = max(0, min(int(offset_ms), max(0, entry.duration_ms - 50)))
        target_ch_ms = entry.ch_start_ms + clamped_offset
        self._paused_ch_ms = target_ch_ms

        self._is_playing = should_play
        self._last_tick_mono = time.monotonic() if should_play else None
        self.btn_play_pause.setText("⏸ Pauza" if should_play else "▶ Odtwórz")

        if not should_play:
            # Jeśli mamy już załadowany ten sam rozdział w playerze, przesuń pozycję bez przeładowania
            if (
                self._stitched_chapter_id == entry.chunk.chapter_id
                and self._stitched_wav_path is not None
                and self._current_source_path == self._stitched_wav_path
            ):
                target_wav_ms = self._ch_ms_to_wav_ms(idx, clamped_offset)
                self._pending_seek_wav_ms = target_wav_ms
                self._pending_autoplay = False
                self._player.pause()
                self._player.setPosition(target_wav_ms)
            else:
                # Leniwe ładowanie: nie otwieramy żadnego pliku WAV dopóki nie padnie Odtwórz
                self._player.stop()
                self._player.setSource(QUrl())
                self._current_source_path = None
                self._pending_seek_wav_ms = None
                self._pending_autoplay = False
        else:
            self._play_chapter_at(entry.chunk.chapter_id, idx, clamped_offset)

        self._update_controls_enabled()
        self._update_time_labels()
        self._update_stats_display()

    def _ch_ms_to_wav_ms(self, playable_idx: int, offset_in_chunk_ms: int) -> int:
        if playable_idx in self._stitched_playable_indices:
            pos_i = self._stitched_playable_indices.index(playable_idx)
            base_ms = self._stitched_starts_ms[pos_i]
            # Jeśli zaczynamy dokładnie od pierwszego zdania rozdziału (offset 0), wystartuj od 0 ms
            # (uwzględniając 250 ms cichego lead-in, aby Bluetooth/WMF nie uciął pierwszego wyrazu!)
            if pos_i == 0 and offset_in_chunk_ms == 0:
                return 0
            return base_ms + max(0, int(offset_in_chunk_ms))
        return CHAPTER_LEAD_IN_MS + self._playable[playable_idx].ch_start_ms + max(0, int(offset_in_chunk_ms))

    def _play_chapter_at(self, chapter_id: int, playable_idx: int, offset_in_chunk_ms: int) -> None:
        wav_path = self._ensure_chapter_stitched(chapter_id)
        if wav_path is None:
            self._is_playing = False
            self.btn_play_pause.setText("▶ Odtwórz")
            return

        target_wav_ms = self._ch_ms_to_wav_ms(playable_idx, offset_in_chunk_ms)
        self._paused_ch_ms = max(0, target_wav_ms - CHAPTER_LEAD_IN_MS)

        if self._current_source_path == wav_path and self._player.mediaStatus() in (
            QMediaPlayer.LoadedMedia,
            QMediaPlayer.BufferedMedia,
            QMediaPlayer.EndOfMedia,
        ):
            # Ten sam ciągły plik rozdziału -> błyskawiczny seek bez przeładowania potoku audio!
            self._pending_seek_wav_ms = None
            self._pending_autoplay = False
            if abs(self._player.playbackRate() - self._playback_rate) > 1e-3:
                self._player.setPlaybackRate(self._playback_rate)
            self._player.setPosition(target_wav_ms)
            if self._player.playbackState() != QMediaPlayer.PlayingState:
                self._player.play()
        else:
            # Nowy rozdział -> ładujemy raz na cały rozdział; rate ustawiamy RAZ przed play()
            self._current_source_path = wav_path
            self._pending_seek_wav_ms = target_wav_ms
            self._pending_autoplay = True
            self._player.setSource(QUrl.fromLocalFile(str(wav_path)))

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.LoadedMedia:
            # Ustawiamy playbackRate RAZ przed startem (unikamy wielokrotnego SetRate w WMF, które ucina początek!)
            if abs(self._player.playbackRate() - self._playback_rate) > 1e-3:
                self._player.setPlaybackRate(self._playback_rate)
            if self._pending_seek_wav_ms is not None:
                pos = self._pending_seek_wav_ms
                self._pending_seek_wav_ms = None
                if pos > 0:
                    self._player.setPosition(pos)
            if self._pending_autoplay:
                self._pending_autoplay = False
                if self._is_playing:
                    self._player.play()
        elif status == QMediaPlayer.EndOfMedia:
            if not self._is_playing:
                return
            self._advance_after_chapter_end()

    def _on_media_error(self, _error, error_string: str) -> None:
        if error_string:
            self.status_message.emit(f"Błąd odtwarzacza: {error_string}")

    def _advance_after_chapter_end(self) -> None:
        if self._current_chapter_id is None:
            return
        prev_ch_id = self._current_chapter_id
        prev_stitched_count = len(self._stitched_playable_indices)

        # Sprawdź czy w tle dogenerowały się nowe fragmenty (w tym rozdziale lub w kolejnych)
        self.refresh_new_chunks()

        # 1. Jeśli w bieżącym rozdziale pojawiły się nowe fragmenty -> kontynuuj od pierwszego nowego
        new_ch_indices = self._chapter_playable_indices.get(prev_ch_id, [])
        if len(new_ch_indices) > prev_stitched_count:
            next_idx = new_ch_indices[prev_stitched_count]
            self._load_playable_index(next_idx, offset_ms=0, autoplay=True)
            return

        # 2. Przejdź do kolejnego gotowego rozdziału
        cur_combo_i = self.combo_chapters.currentIndex()
        if 0 <= cur_combo_i < self.combo_chapters.count() - 1:
            next_ch_id = self.combo_chapters.itemData(cur_combo_i + 1)
            if next_ch_id in self._chapter_first_playable_idx:
                self._load_playable_index(self._chapter_first_playable_idx[next_ch_id], offset_ms=0, autoplay=True)
                return

        # 3. Koniec wygenerowanego audio
        self._is_playing = False
        self._last_tick_mono = None
        self.btn_play_pause.setText("▶ Odtwórz")
        self.persist_state()
        self.status_message.emit("Dotarto do końca wygenerowanych fragmentów.")

    def toggle_play_pause(self) -> None:
        if not self._playable:
            self.refresh_new_chunks()
            if not self._playable:
                return
        if self._is_playing:
            self.pause_playback()
            return

        self._sync_audio_output_device()

        if self._current_idx < 0:
            self._last_paused_epoch = None
            self._load_playable_index(0, offset_ms=0, autoplay=True)
            return

        # Automatyczne cofnięcie po przerwie (styl Smart AudioBook Player: 2 s <= 1 h, 15 s > 1 h)
        rewind_ms = compute_auto_rewind_ms(self._last_paused_epoch)
        self._last_paused_epoch = None
        if rewind_ms > 0:
            cur_book_ms = self.current_book_ms()
            if cur_book_ms > 0:
                target_ms = max(0, cur_book_ms - rewind_ms)
                idx = bisect.bisect_right(self._playable_starts_ms, target_ms) - 1
                idx = max(0, min(idx, len(self._playable) - 1))
                target_entry = self._playable[idx]
                offset_ms = max(0, min(target_ms - target_entry.book_start_ms, max(0, target_entry.duration_ms - 60)))
                self._load_playable_index(idx, offset_ms=offset_ms, autoplay=True)
                kind = "dłuższej przerwie" if rewind_ms >= AUTO_REWIND_LONG_MS else "krótkiej przerwie"
                self.status_message.emit(f"⏪ Auto-cofnięcie o {rewind_ms // 1000} s po {kind}")
                return

        entry = self._playable[self._current_idx]
        if (
            self._current_source_path is not None
            and self._current_source_path == self._stitched_wav_path
            and self._stitched_chapter_id == entry.chunk.chapter_id
            and self._player.mediaStatus() != QMediaPlayer.EndOfMedia
        ):
            self._is_playing = True
            self._last_tick_mono = time.monotonic()
            self.btn_play_pause.setText("⏸ Pauza")
            if abs(self._player.playbackRate() - self._playback_rate) > 1e-3:
                self._player.setPlaybackRate(self._playback_rate)
            self._player.play()
        else:
            offset_in_chunk = max(0, self._paused_ch_ms - entry.ch_start_ms)
            self._load_playable_index(self._current_idx, offset_ms=offset_in_chunk, autoplay=True)

    def current_chapter_ms(self) -> int:
        if not (0 <= self._current_idx < len(self._playable)):
            return 0
        entry = self._playable[self._current_idx]
        if self._pending_seek_wav_ms is not None:
            return max(0, self._pending_seek_wav_ms - CHAPTER_LEAD_IN_MS)
        if (
            self._current_source_path is not None
            and self._current_source_path == self._stitched_wav_path
            and self._stitched_chapter_id == entry.chunk.chapter_id
            and self._player.mediaStatus() in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia, QMediaPlayer.EndOfMedia)
        ):
            return max(0, int(self._player.position()) - CHAPTER_LEAD_IN_MS)
        return max(0, self._paused_ch_ms)

    def current_book_ms(self) -> int:
        if not (0 <= self._current_idx < len(self._playable)):
            return 0
        entry = self._playable[self._current_idx]
        first_idx = self._chapter_first_playable_idx.get(entry.chunk.chapter_id, self._current_idx)
        ch_book_base = self._playable[first_idx].book_start_ms
        return ch_book_base + self.current_chapter_ms()

    def seek_relative_ms(self, delta_ms: int) -> None:
        if not self._playable:
            return
        if self._current_idx < 0:
            self._load_playable_index(0, offset_ms=0, autoplay=False)
            return
        self.seek_to_book_ms(self.current_book_ms() + int(delta_ms))

    def seek_to_book_ms(self, target_ms: int) -> None:
        if not self._playable:
            return
        self._last_paused_epoch = None
        clamped = max(0, min(int(target_ms), max(0, self._book_total_ms - 80)))
        idx = bisect.bisect_right(self._playable_starts_ms, clamped) - 1
        idx = max(0, min(idx, len(self._playable) - 1))
        entry = self._playable[idx]
        offset_ms = max(0, min(clamped - entry.book_start_ms, max(0, entry.duration_ms - 60)))
        self._load_playable_index(idx, offset_ms=offset_ms, autoplay=self._is_playing)
        self.persist_state()

    def seek_to_chapter_ms(self, target_ch_ms: int) -> None:
        if not self._playable or self._current_chapter_id is None:
            return
        self._last_paused_epoch = None
        first_idx = self._chapter_first_playable_idx.get(self._current_chapter_id)
        if first_idx is None:
            return
        ch_dur = self._chapter_duration_ms.get(self._current_chapter_id, 0)
        clamped_ch = max(0, min(int(target_ch_ms), max(0, ch_dur - 80)))

        # Jeśli rozdział jest już sklejony, znajdź dokładnie pasujący chunk z tablicy czasów
        if self._stitched_chapter_id == self._current_chapter_id and self._stitched_starts_ms:
            target_wav_ms = CHAPTER_LEAD_IN_MS + clamped_ch
            pos_i = max(0, bisect.bisect_right(self._stitched_starts_ms, target_wav_ms) - 1)
            idx = self._stitched_playable_indices[pos_i]
            offset_in_chunk = max(0, target_wav_ms - self._stitched_starts_ms[pos_i])
            self._load_playable_index(idx, offset_ms=offset_in_chunk, autoplay=self._is_playing)
            self.persist_state()
            return

        ch_book_base = self._playable[first_idx].book_start_ms
        self.seek_to_book_ms(ch_book_base + clamped_ch)

    def prev_chapter(self) -> None:
        idx = self.combo_chapters.currentIndex()
        if idx > 0:
            self.combo_chapters.setCurrentIndex(idx - 1)

    def next_chapter(self) -> None:
        idx = self.combo_chapters.currentIndex()
        if 0 <= idx < self.combo_chapters.count() - 1:
            self.combo_chapters.setCurrentIndex(idx + 1)

    def _sync_chapter_combo(self, chapter_id: int) -> None:
        self.combo_chapters.blockSignals(True)
        for i in range(self.combo_chapters.count()):
            if self.combo_chapters.itemData(i) == chapter_id:
                self.combo_chapters.setCurrentIndex(i)
                break
        self.combo_chapters.blockSignals(False)

    def _on_chapter_combo_changed(self, combo_idx: int) -> None:
        if combo_idx < 0:
            return
        ch_id = self.combo_chapters.itemData(combo_idx)
        if ch_id is None or ch_id not in self._chapter_first_playable_idx:
            return
        target_idx = self._chapter_first_playable_idx[ch_id]
        self._last_paused_epoch = None
        self._load_playable_index(target_idx, offset_ms=0, autoplay=self._is_playing)
        self.persist_state()

    # =========================================================================
    # TEMPO, GŁOŚNOŚĆ I WYJŚCIE AUDIO
    # =========================================================================
    def set_playback_rate(self, rate: float) -> None:
        clamped = round(max(0.60, min(2.50, float(rate))), 2)
        self.spin_speed.setValue(clamped)

    def adjust_playback_rate(self, delta: float) -> None:
        self.set_playback_rate(self._playback_rate + delta)

    def _on_speed_changed(self, val: float) -> None:
        self._playback_rate = round(float(val), 2)
        if self._current_source_path is not None and abs(self._player.playbackRate() - self._playback_rate) > 1e-3:
            self._player.setPlaybackRate(self._playback_rate)
        self._update_stats_display()
        self.persist_state()

    def _on_volume_changed(self, val: int) -> None:
        self._audio_out.setVolume(max(0.0, min(1.0, val / 100.0)))
        self.lbl_volume.setText(f"{val}%")
        if val > 0 and self._muted:
            self._muted = False
            self._audio_out.setMuted(False)
            self.btn_mute.setText("🔊")

    def _on_toggle_mute(self) -> None:
        self._muted = not self._muted
        self._audio_out.setMuted(self._muted)
        self.btn_mute.setText("🔇" if self._muted else "🔊")

    def _find_audio_device_by_id(self, dev_id: bytes) -> QAudioDevice:
        if dev_id:
            for d in QMediaDevices.audioOutputs():
                if bytes(d.id()) == dev_id:
                    return d
        return QMediaDevices.defaultAudioOutput()

    def _apply_audio_device(self, dev: QAudioDevice) -> None:
        if dev.isNull():
            return
        if bytes(self._audio_out.device().id()) == bytes(dev.id()):
            return
        was_playing = self._is_playing
        cur_pos_ms = self.current_book_ms() if was_playing else None
        self._audio_out.setDevice(dev)
        self._audio_out.setVolume(max(0.0, min(1.0, self.slider_volume.value() / 100.0)))
        self._audio_out.setMuted(self._muted)
        if was_playing and cur_pos_ms is not None and 0 <= self._current_idx < len(self._playable):
            self._player.stop()
            self._current_source_path = None
            self.seek_to_book_ms(cur_pos_ms)

    def _sync_audio_output_device(self) -> None:
        target = self._find_audio_device_by_id(self._selected_device_id)
        self._apply_audio_device(target)
        if self.combo_audio_device.count() > 0:
            def_name = QMediaDevices.defaultAudioOutput().description() or "brak"
            self.combo_audio_device.setItemText(0, f"Domyślne systemowe ({def_name})")

    def _refresh_audio_devices(self) -> None:
        self.combo_audio_device.blockSignals(True)
        self.combo_audio_device.clear()
        default_dev = QMediaDevices.defaultAudioOutput()
        def_name = default_dev.description() or "brak"
        self.combo_audio_device.addItem(f"Domyślne systemowe ({def_name})", b"")

        selected_idx = 0
        for d in QMediaDevices.audioOutputs():
            d_id = bytes(d.id())
            self.combo_audio_device.addItem(d.description(), d_id)
            if self._selected_device_id and d_id == self._selected_device_id:
                selected_idx = self.combo_audio_device.count() - 1

        self.combo_audio_device.setCurrentIndex(selected_idx)
        self.combo_audio_device.blockSignals(False)

    def _on_audio_outputs_changed(self) -> None:
        self._refresh_audio_devices()
        self._sync_audio_output_device()

    def _on_audio_device_combo_changed(self, idx: int) -> None:
        if idx < 0:
            return
        data = self.combo_audio_device.itemData(idx)
        self._selected_device_id = bytes(data) if data else b""
        self._sync_audio_output_device()

    # =========================================================================
    # SUWAK ROZDZIAŁU I TICK UI (SYNCHRONIZACJA KARAOKE)
    # =========================================================================
    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True

    def _on_slider_moved(self, val: int) -> None:
        if self._current_chapter_id is None:
            return
        ch_dur = self._chapter_duration_ms.get(self._current_chapter_id, 0)
        preview_ms = int(round(ch_dur * (val / 1000.0)))
        rem_ms = max(0, ch_dur - preview_ms)
        self.lbl_ch_elapsed.setText(_fmt_mmss(preview_ms))
        self.lbl_ch_remaining.setText(f"(-{_fmt_mmss(rem_ms)})")
        self.lbl_ch_pct.setText(f"Rozdział: {val / 10.0:.0f}%")

    def _on_slider_released(self) -> None:
        self._slider_dragging = False
        if self._current_chapter_id is None:
            return
        ch_dur = self._chapter_duration_ms.get(self._current_chapter_id, 0)
        target_ms = int(round(ch_dur * (self.slider_chapter.value() / 1000.0)))
        self.seek_to_chapter_ms(target_ms)

    def _sync_active_chunk_from_position(self) -> None:
        if (
            self._pending_seek_wav_ms is not None
            or not self._stitched_starts_ms
            or self._current_source_path != self._stitched_wav_path
            or self._player.mediaStatus() not in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia)
        ):
            return
        pos_wav_ms = int(self._player.position())
        pos_i = max(0, bisect.bisect_right(self._stitched_starts_ms, pos_wav_ms) - 1)
        if 0 <= pos_i < len(self._stitched_playable_indices):
            new_idx = self._stitched_playable_indices[pos_i]
            if new_idx != self._current_idx and 0 <= new_idx < len(self._playable):
                self._current_idx = new_idx
                self._highlight_karaoke_chunk(self._playable[new_idx].chunk.id)

    def _on_ui_tick(self) -> None:
        now = time.monotonic()
        if self._is_playing:
            if self._last_tick_mono is not None:
                dt = max(0.0, min(1.0, now - self._last_tick_mono))
                self._accumulate_listening_time(dt)
            self._last_tick_mono = now
            self._sync_active_chunk_from_position()
            if now - self._last_persist_mono >= 10.0:
                self.persist_state()
                self._last_persist_mono = now
            self._update_time_labels()
            self._update_stats_display()
        else:
            self._last_tick_mono = None

    def _update_time_labels(self) -> None:
        if not self._playable or not (0 <= self._current_idx < len(self._playable)):
            self.lbl_ch_elapsed.setText("00:00")
            self.lbl_ch_remaining.setText("(-00:00)")
            self.lbl_ch_total.setText("00:00")
            self.lbl_ch_pct.setText("Rozdział: 0%")
            self.lbl_book_time.setText("00:00 / 00:00")
            self.bar_book_progress.setValue(0)
            if not self._slider_dragging:
                self.slider_chapter.setValue(0)
            return

        ch_id = self._playable[self._current_idx].chunk.chapter_id
        ch_dur = max(1, self._chapter_duration_ms.get(ch_id, 1))
        ch_pos = min(ch_dur, self.current_chapter_ms())
        ch_rem = max(0, ch_dur - ch_pos)

        if not self._slider_dragging:
            self.slider_chapter.setValue(int(round(1000.0 * ch_pos / ch_dur)))
            self.lbl_ch_elapsed.setText(_fmt_mmss(ch_pos))
            self.lbl_ch_remaining.setText(f"(-{_fmt_mmss(ch_rem)})")
            self.lbl_ch_pct.setText(f"Rozdział: {100.0 * ch_pos / ch_dur:.0f}%")
        self.lbl_ch_total.setText(_fmt_mmss(ch_dur))

        book_dur = max(1, self._book_total_ms)
        book_pos = min(book_dur, self.current_book_ms())
        self.bar_book_progress.setValue(int(round(1000.0 * book_pos / book_dur)))
        self.lbl_book_time.setText(f"{_fmt_mmss(book_pos)} / {_fmt_mmss(book_dur)}")

    def _update_controls_enabled(self) -> None:
        has_audio = len(self._playable) > 0
        for w in (
            self.btn_play_pause,
            self.btn_back_10,
            self.btn_back_60,
            self.btn_fwd_10,
            self.btn_fwd_60,
            self.slider_chapter,
            self.combo_chapters,
            self.btn_prev_chapter,
            self.btn_next_chapter,
            self.btn_fix_current,
        ):
            w.setEnabled(has_audio)
        self.btn_refresh_audio.setEnabled(self.db is not None)

    # =========================================================================
    # KARAOKE TEKSTU (KLIK = ODTWÓRZ, DWUKLIK = SKOCZ DO ZAKŁADKI 3)
    # =========================================================================
    def _populate_karaoke_for_chapter(self, chapter_id: int, highlight_chunk_id: int | None = None) -> None:
        if self.db is None:
            return
        chunks = self.db.list_chunks(chapter_id=chapter_id)
        self.list_karaoke.clear()
        self._karaoke_chunk_to_row.clear()
        self._last_highlighted_chunk_id = None

        ch_title = chunks[0].chapter_title if chunks else ""
        self._karaoke_chapter_title = ch_title
        self.lbl_karaoke_header.setText(f"Tekst: {ch_title}  (klik — skocz | dwuklik — Zakładka 3)")

        base_font = QFont()
        base_font.setPointSize(self._karaoke_font_pt)

        for row_idx, c in enumerate(chunks):
            txt = (c.raw_text or c.normalized_text).strip()
            is_ready = c.id in self._chunk_id_to_playable_idx
            role_ico = " 👨" if c.voice_role == "male" else (" 👩" if c.voice_role == "female" else "")
            item = QListWidgetItem(f"#{c.chunk_index + 1}{role_ico}  {txt}")
            item.setData(Qt.UserRole, c.id)
            item.setData(Qt.UserRole + 1, is_ready)
            item.setData(Qt.UserRole + 2, c.chunk_index + 1)
            item.setFont(base_font)
            if is_ready:
                item.setForeground(QBrush(QColor("#cfd6e2")))
                qg_str = f" | QG={c.qg_score:.2f}" if c.qg_score is not None else ""
                item.setToolTip(
                    f"Fragment #{c.chunk_index + 1} ({c.duration_sec:.1f}s{qg_str})\n"
                    f"• Kliknij — odtwórz stąd\n• Dwuklik — pokaż w Zakładce 3 (Whisper i edycja)"
                )
            else:
                item.setForeground(QBrush(QColor("#596273")))
                f_it = QFont(base_font)
                f_it.setItalic(True)
                item.setFont(f_it)
                item.setToolTip(
                    f"Fragment #{c.chunk_index + 1} — jeszcze nie wygenerowano\n"
                    f"• Dwuklik — pokaż w Zakładce 3"
                )
            self.list_karaoke.addItem(item)
            self._karaoke_chunk_to_row[c.id] = row_idx

        if highlight_chunk_id is not None:
            self._highlight_karaoke_chunk(highlight_chunk_id)

    def _highlight_karaoke_chunk(self, active_chunk_id: int) -> None:
        if self._last_highlighted_chunk_id == active_chunk_id:
            return
        active_row = self._karaoke_chunk_to_row.get(active_chunk_id)
        if active_row is None:
            return
        self._last_highlighted_chunk_id = active_chunk_id

        base_font = QFont()
        base_font.setPointSize(self._karaoke_font_pt)
        active_font = QFont(base_font)
        active_font.setBold(True)
        pending_font = QFont(base_font)
        pending_font.setItalic(True)

        for row in range(self.list_karaoke.count()):
            item = self.list_karaoke.item(row)
            if item is None:
                continue
            is_ready = bool(item.data(Qt.UserRole + 1))
            if row == active_row:
                item.setBackground(QBrush(QColor("#1f3b66")))
                item.setForeground(QBrush(QColor("#ffffff")))
                item.setFont(active_font)
            elif not is_ready:
                item.setBackground(QBrush(Qt.transparent))
                item.setForeground(QBrush(QColor("#596273")))
                item.setFont(pending_font)
            elif row < active_row:
                item.setBackground(QBrush(Qt.transparent))
                item.setForeground(QBrush(QColor("#808a9b")))
                item.setFont(base_font)
            else:
                item.setBackground(QBrush(Qt.transparent))
                item.setForeground(QBrush(QColor("#dce2ec")))
                item.setFont(base_font)

        active_item = self.list_karaoke.item(active_row)
        if active_item is not None:
            chunk_num = active_item.data(Qt.UserRole + 2)
            num_part = f" • #{chunk_num}" if chunk_num is not None else ""
            self.lbl_karaoke_header.setText(
                f"Tekst: {self._karaoke_chapter_title}{num_part}  (klik — skocz | dwuklik — Zakładka 3)"
            )
            if self.chk_autoscroll.isChecked():
                self.list_karaoke.scrollToItem(active_item, QAbstractItemView.PositionAtCenter)

    def _on_karaoke_item_clicked(self, item: QListWidgetItem) -> None:
        """Odracza reakcję na pojedynczy klik o 260 ms, aby rozróżnić go od dwukliku bez przesuwania listy."""
        chunk_id = item.data(Qt.UserRole)
        if chunk_id is None:
            return
        self._pending_click_chunk_id = int(chunk_id)
        self._karaoke_click_timer.start()

    def _on_karaoke_single_click_timeout(self) -> None:
        chunk_id = self._pending_click_chunk_id
        self._pending_click_chunk_id = None
        if chunk_id is None:
            return
        if chunk_id not in self._chunk_id_to_playable_idx:
            self.refresh_new_chunks()
        if chunk_id not in self._chunk_id_to_playable_idx:
            self.status_message.emit("Ten fragment nie został jeszcze wygenerowany.")
            return
        target_idx = self._chunk_id_to_playable_idx[chunk_id]
        self._last_paused_epoch = None
        self._load_playable_index(target_idx, offset_ms=0, autoplay=True)
        self.persist_state()

    def _on_karaoke_item_double_clicked(self, item: QListWidgetItem) -> None:
        """Dwuklik anuluje oczekujący pojedynczy klik i od razu przenosi do tej sekcji w Zakładce 3."""
        self._karaoke_click_timer.stop()
        self._pending_click_chunk_id = None
        chunk_id = item.data(Qt.UserRole)
        if chunk_id is None:
            return
        self.pause_playback()
        self.edit_chunk_requested.emit(int(chunk_id))

    def _change_karaoke_font(self, delta: int) -> None:
        self._karaoke_font_pt = max(9, min(22, self._karaoke_font_pt + delta))
        if self._current_chapter_id is not None:
            cur_id = self._playable[self._current_idx].chunk.id if 0 <= self._current_idx < len(self._playable) else None
            self._populate_karaoke_for_chapter(self._current_chapter_id, highlight_chunk_id=cur_id)

    def _on_fix_current_clicked(self) -> None:
        if not (0 <= self._current_idx < len(self._playable)):
            return
        chunk_id = self._playable[self._current_idx].chunk.id
        self.pause_playback()
        self.edit_chunk_requested.emit(chunk_id)

    # =========================================================================
    # STATYSTYKI SŁUCHANIA I ZAPIS STANU
    # =========================================================================
    def _load_global_stats(self) -> dict:
        if self._stats_path.exists():
            try:
                data = json.loads(self._stats_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("total_listened_sec", 0.0)
                    data.setdefault("daily_sec", {})
                    return data
            except Exception:
                pass
        return {"total_listened_sec": 0.0, "daily_sec": {}}

    def _accumulate_listening_time(self, dt_sec: float) -> None:
        if dt_sec <= 0:
            return
        today_key = date.today().isoformat()
        self._session_listened_sec += dt_sec
        self._book_listened_sec += dt_sec
        self._global_stats["total_listened_sec"] = float(self._global_stats.get("total_listened_sec", 0.0)) + dt_sec
        daily = self._global_stats.setdefault("daily_sec", {})
        daily[today_key] = float(daily.get(today_key, 0.0)) + dt_sec

    def _update_stats_display(self) -> None:
        today_key = date.today().isoformat()
        today_sec = float(self._global_stats.get("daily_sec", {}).get(today_key, 0.0))
        total_sec = float(self._global_stats.get("total_listened_sec", 0.0))

        self.lbl_stats_main.setText(
            f"Dzisiaj: {_fmt_duration_human(today_sec)}  •  "
            f"Ta książka: {_fmt_duration_human(self._book_listened_sec)}  •  "
            f"Łącznie: {_fmt_duration_human(total_sec)}"
        )

        if self._book_total_ms > 0 and 0 <= self._current_idx < len(self._playable):
            pos_ms = self.current_book_ms()
            pct = 100.0 * pos_ms / max(1, self._book_total_ms)
            rem_audio_sec = max(0.0, (self._book_total_ms - pos_ms) / 1000.0)
            rem_real_sec = rem_audio_sec / max(0.5, self._playback_rate)
            self.lbl_stats_sub.setText(
                f"Odsłuchano: {pct:.1f}%  •  Pozostało do końca gotowego audio: "
                f"~{_fmt_duration_human(rem_real_sec)} (przy {self._playback_rate:.2f}x)"
            )
        else:
            self.lbl_stats_sub.setText("Pozostało do końca gotowego audio: —")

    def persist_state(self) -> None:
        try:
            self._stats_path.parent.mkdir(parents=True, exist_ok=True)
            self._global_stats["last_updated"] = datetime.now().isoformat(timespec="seconds")
            self._stats_path.write_text(
                json.dumps(self._global_stats, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        if self.db is not None and self.db.db_path.exists():
            try:
                self.db.set_meta("player_listened_sec", f"{self._book_listened_sec:.1f}")
                self.db.set_meta("player_speed", f"{self._playback_rate:.2f}")
                if 0 <= self._current_idx < len(self._playable):
                    entry = self._playable[self._current_idx]
                    offset_ms = max(0, self.current_chapter_ms() - entry.ch_start_ms)
                    self.db.set_meta("player_chunk_id", str(entry.chunk.id))
                    self.db.set_meta("player_offset_ms", str(min(offset_ms, entry.duration_ms)))
                if self._is_playing:
                    self.db.set_meta("player_paused_at_ts", f"{time.time():.2f}")
                elif self._last_paused_epoch is not None:
                    self.db.set_meta("player_paused_at_ts", f"{self._last_paused_epoch:.2f}")
                else:
                    self.db.set_meta("player_paused_at_ts", "")
            except Exception:
                pass
