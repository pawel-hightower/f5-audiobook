"""Główne okno aplikacji desktopowej F5-Audiobook (PySide6)."""

from __future__ import annotations

import os
import sys
import time
import winsound
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from f5_audiobook.audio.exporter import ExportResult
from f5_audiobook.config import (
    OUTPUT_DIR,
    VOICES_DIR,
    WORK_DIR,
    load_pronunciation_dict,
    save_pronunciation_dict,
)
from f5_audiobook.engine.f5_ipa import F5IpaEngine
from f5_audiobook.gui.workers import (
    BatchWorker,
    ExportWorker,
    RegenerateChunkWorker,
    SingleTestWorker,
)
from f5_audiobook.pipeline.batch_runner import BatchOptions, BatchProgress
from f5_audiobook.pipeline.project_db import (
    ChunkRow,
    ProjectDB,
    list_saved_projects,
)
from f5_audiobook.text.epub_parser import parse_book
from f5_audiobook.text.normalizer_pl import normalize_polish_text
from f5_audiobook.voices import (
    VoiceProfile,
    cut_reference_sample,
    list_voice_profiles,
    save_voice_profile,
    slugify_voice_name,
)


def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    sec = int(seconds)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _play_wav_async(path: Path) -> None:
    if not path.exists():
        return
    if sys.platform == "win32":
        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)


def _stop_wav_playback() -> None:
    if sys.platform == "win32":
        winsound.PlaySound(None, winsound.SND_PURGE)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("F5-Audiobook — Generator Polskich Audiobooków AI")
        self.resize(1220, 820)

        self.engine = F5IpaEngine()
        self.db: ProjectDB | None = None
        self.batch_worker: BatchWorker | None = None
        self.test_worker: SingleTestWorker | None = None
        self.regen_worker: RegenerateChunkWorker | None = None
        self.export_worker: ExportWorker | None = None
        self._chunk_rows_cache: list[ChunkRow] = []

        # Licznik czasu od startu generowania
        self._elapsed_base_sec: float = 0.0
        self._run_segment_start: float | None = None
        self._last_progress: BatchProgress | None = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._on_elapsed_tick)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._build_book_tab()
        self._build_voice_tab()
        self._build_generation_tab()
        self._build_pronunciation_tab()

        self._refresh_saved_projects()
        self._refresh_voices()
        self._load_pronunciation_table()

    # =========================================================================
    # ZAKŁADKA 1: KSIĄŻKA I ROZDZIAŁY
    # =========================================================================
    def _build_book_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Pasek wczytywania
        top_bar = QHBoxLayout()
        self.btn_open_book = QPushButton("📖 Otwórz EPUB / TXT...")
        self.btn_open_book.setObjectName("primaryBtn")
        self.btn_open_book.clicked.connect(self._on_open_book)

        top_bar.addWidget(self.btn_open_book)
        top_bar.addSpacing(12)
        top_bar.addWidget(QLabel("Lub wznów projekt:"))
        self.combo_projects = QComboBox()
        self.combo_projects.setMinimumWidth(260)
        top_bar.addWidget(self.combo_projects)

        self.btn_load_project = QPushButton("📂 Wczytaj projekt")
        self.btn_load_project.clicked.connect(self._on_load_saved_project)
        top_bar.addWidget(self.btn_load_project)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Metadane i okładka
        meta_group = QGroupBox("Informacje o książce")
        meta_layout = QHBoxLayout(meta_group)

        self.lbl_cover = QLabel("Brak okładki")
        self.lbl_cover.setFixedSize(96, 136)
        self.lbl_cover.setAlignment(Qt.AlignCenter)
        self.lbl_cover.setStyleSheet("border: 1px solid #323846; background: #13151a;")
        meta_layout.addWidget(self.lbl_cover)

        info_col = QVBoxLayout()
        self.lbl_book_title = QLabel("Nie wczytano książki")
        self.lbl_book_title.setStyleSheet("font-size: 13pt; font-weight: bold;")
        self.lbl_book_author = QLabel("Autor: —")
        self.lbl_book_stats = QLabel("Rozdziały: 0 | Fragmenty: 0 | Szacowany czas: —")
        self.lbl_book_path = QLabel("")
        self.lbl_book_path.setStyleSheet("color: #8b949e; font-size: 9pt;")
        info_col.addWidget(self.lbl_book_title)
        info_col.addWidget(self.lbl_book_author)
        info_col.addWidget(self.lbl_book_stats)
        info_col.addWidget(self.lbl_book_path)
        info_col.addStretch()
        meta_layout.addLayout(info_col, 1)
        layout.addWidget(meta_group)

        # Splitter: tabela rozdziałów | podgląd tekstu po normalizacji
        splitter = QSplitter(Qt.Horizontal)

        left_box = QWidget()
        left_lay = QVBoxLayout(left_box)
        left_lay.setContentsMargins(0, 0, 0, 0)

        sel_bar = QHBoxLayout()
        btn_sel_all = QPushButton("Zaznacz wszystkie")
        btn_sel_all.clicked.connect(lambda: self._set_all_chapters(True))
        btn_sel_none = QPushButton("Odznacz wszystkie")
        btn_sel_none.clicked.connect(lambda: self._set_all_chapters(False))
        sel_bar.addWidget(btn_sel_all)
        sel_bar.addWidget(btn_sel_none)
        sel_bar.addStretch()
        left_lay.addLayout(sel_bar)

        self.tbl_chapters = QTableWidget(0, 4)
        self.tbl_chapters.setHorizontalHeaderLabels(["Użyj", "Rozdział", "Fragmenty", "Postęp"])
        self.tbl_chapters.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_chapters.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_chapters.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tbl_chapters.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tbl_chapters.setAlternatingRowColors(True)
        self.tbl_chapters.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_chapters.itemChanged.connect(self._on_chapter_checkbox_changed)
        self.tbl_chapters.itemSelectionChanged.connect(self._on_chapter_selected_preview)
        left_lay.addWidget(self.tbl_chapters)
        splitter.addWidget(left_box)

        right_box = QWidget()
        right_lay = QVBoxLayout(right_box)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.addWidget(QLabel("Podgląd tekstu po normalizacji (podział na zdania/chunki):"))
        self.txt_chapter_preview = QPlainTextEdit()
        self.txt_chapter_preview.setReadOnly(True)
        right_lay.addWidget(self.txt_chapter_preview)
        splitter.addWidget(right_box)

        splitter.setSizes([560, 620])
        layout.addWidget(splitter, 1)

        self.tabs.addTab(tab, "1. Książka i Rozdziały")

    # =========================================================================
    # ZAKŁADKA 2: LEKTOR I PRÓBKA GŁOSU
    # =========================================================================
    def _build_voice_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Wybór profilu
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Profil lektora:"))
        self.combo_voices = QComboBox()
        self.combo_voices.setMinimumWidth(240)
        self.combo_voices.currentIndexChanged.connect(self._on_voice_selected)
        top_bar.addWidget(self.combo_voices)

        self.btn_new_voice = QPushButton("➕ Nowy profil")
        self.btn_new_voice.clicked.connect(self._on_new_voice_profile)
        top_bar.addWidget(self.btn_new_voice)

        self.btn_save_voice = QPushButton("💾 Zapisz profil")
        self.btn_save_voice.setObjectName("primaryBtn")
        self.btn_save_voice.clicked.connect(self._on_save_voice_profile)
        top_bar.addWidget(self.btn_save_voice)

        self.btn_reset_voice_defaults = QPushButton("↩ Domyślne")
        self.btn_reset_voice_defaults.setToolTip(
            "Przywraca wypracowane domyślne parametry syntezy (speed=1.15, CFG=2.00, ODE=32, Seed=101)"
        )
        self.btn_reset_voice_defaults.clicked.connect(self._on_reset_voice_defaults)
        top_bar.addWidget(self.btn_reset_voice_defaults)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Parametry profilu
        form_group = QGroupBox("Ustawienia głosu i silnika F5-TTS IPA")
        form = QFormLayout(form_group)

        self.ed_voice_display = QLineEdit()
        form.addRow("Nazwa wyświetlana:", self.ed_voice_display)

        ref_row = QHBoxLayout()
        self.ed_ref_audio = QLineEdit()
        self.btn_browse_ref = QPushButton("Wybierz WAV...")
        self.btn_browse_ref.clicked.connect(self._on_browse_ref_wav)
        btn_play_ref = QPushButton("🔊 Odsłuchaj próbkę")
        btn_play_ref.clicked.connect(self._on_play_ref_wav)
        ref_row.addWidget(self.ed_ref_audio, 1)
        ref_row.addWidget(self.btn_browse_ref)
        ref_row.addWidget(btn_play_ref)
        form.addRow("Plik próbki (6–10 s WAV):", ref_row)

        self.ed_ref_text = QPlainTextEdit()
        self.ed_ref_text.setMaximumHeight(72)
        self.ed_ref_text.setPlaceholderText("Dokładna transkrypcja słowo w słowo tego, co mówi lektor na próbce WAV...")
        form.addRow("Transkrypcja próbki:", self.ed_ref_text)

        params_row = QHBoxLayout()
        self.spn_speed = QDoubleSpinBox()
        self.spn_speed.setRange(0.75, 1.50)
        self.spn_speed.setSingleStep(0.02)
        self.spn_speed.setValue(1.15)

        self.spn_cfg = QDoubleSpinBox()
        self.spn_cfg.setRange(1.0, 3.5)
        self.spn_cfg.setSingleStep(0.1)
        self.spn_cfg.setValue(2.0)

        self.spn_nfe = QSpinBox()
        self.spn_nfe.setRange(16, 48)
        self.spn_nfe.setSingleStep(4)
        self.spn_nfe.setValue(32)

        self.spn_seed = QSpinBox()
        self.spn_seed.setRange(1, 9999999)
        self.spn_seed.setValue(101)

        params_row.addWidget(QLabel("Tempo (speed):"))
        params_row.addWidget(self.spn_speed)
        params_row.addSpacing(14)
        params_row.addWidget(QLabel("CFG:"))
        params_row.addWidget(self.spn_cfg)
        params_row.addSpacing(14)
        params_row.addWidget(QLabel("Kroki ODE:"))
        params_row.addWidget(self.spn_nfe)
        params_row.addSpacing(14)
        params_row.addWidget(QLabel("Seed:"))
        params_row.addWidget(self.spn_seed)
        params_row.addStretch()
        form.addRow("Parametry syntezy:", params_row)
        layout.addWidget(form_group)

        # Wbudowana wycinarka próbki z audiobooka
        self.cut_group = QGroupBox("Wycinarka próbki referencyjnej (FFmpeg — 24 kHz mono WAV)")
        cut_lay = QHBoxLayout(self.cut_group)
        self.ed_cut_src = QLineEdit()
        self.ed_cut_src.setPlaceholderText("Ścieżka do pliku audiobooka (MP3 / M4B / WAV)...")
        btn_cut_browse = QPushButton("Wskaż plik...")
        btn_cut_browse.clicked.connect(self._on_browse_cut_source)
        self.ed_cut_start = QLineEdit("00:00:00")
        self.ed_cut_start.setFixedWidth(85)
        self.spn_cut_dur = QDoubleSpinBox()
        self.spn_cut_dur.setRange(4.0, 12.0)
        self.spn_cut_dur.setSingleStep(0.5)
        self.spn_cut_dur.setValue(7.5)
        self.spn_cut_dur.setSuffix(" s")
        btn_cut_run = QPushButton("✂ Wytnij do profilu")
        btn_cut_run.clicked.connect(self._on_cut_sample)

        cut_lay.addWidget(self.ed_cut_src, 1)
        cut_lay.addWidget(btn_cut_browse)
        cut_lay.addWidget(QLabel("Start:"))
        cut_lay.addWidget(self.ed_cut_start)
        cut_lay.addWidget(QLabel("Długość:"))
        cut_lay.addWidget(self.spn_cut_dur)
        cut_lay.addWidget(btn_cut_run)
        layout.addWidget(self.cut_group)

        # Szybki test 1 zdania
        test_group = QGroupBox("Szybki test 1 zdania (z bieżącymi ustawieniami i słownikiem wymowy)")
        test_lay = QVBoxLayout(test_group)
        self.ed_test_sentence = QLineEdit(
            "Wieczór zapadał nad miastem powoli, a chłodny wiatr niósł zapach zbliżającego się deszczu."
        )
        test_lay.addWidget(self.ed_test_sentence)

        test_btns = QHBoxLayout()
        self.btn_run_test = QPushButton("▶ Generuj i odsłuchaj test")
        self.btn_run_test.setObjectName("primaryBtn")
        self.btn_run_test.clicked.connect(self._on_run_single_test)
        btn_stop_snd = QPushButton("⏹ Zatrzymaj odsłuch")
        btn_stop_snd.clicked.connect(_stop_wav_playback)
        self.lbl_test_status = QLabel("")
        test_btns.addWidget(self.btn_run_test)
        test_btns.addWidget(btn_stop_snd)
        test_btns.addWidget(self.lbl_test_status, 1)
        test_lay.addLayout(test_btns)
        layout.addWidget(test_group)

        layout.addStretch()

        # Blokada edycji na dole zakładki (ochrona przed przypadkowym zepsuciem lektora)
        lock_bar = QHBoxLayout()
        self.chk_edit_voice = QCheckBox("🔒 Edytuj ustawienia lektora")
        self.chk_edit_voice.setChecked(False)
        self.chk_edit_voice.toggled.connect(self._on_toggle_voice_edit_lock)
        lock_hint = QLabel("Domyślnie zablokowane, aby nie zmienić przypadkiem wypracowanych parametrów głosu.")
        lock_hint.setStyleSheet("color: #8b949e; font-size: 9pt;")
        lock_bar.addWidget(self.chk_edit_voice)
        lock_bar.addSpacing(12)
        lock_bar.addWidget(lock_hint)
        lock_bar.addStretch()
        layout.addLayout(lock_bar)

        self._voice_editable_widgets = [
            self.btn_new_voice,
            self.btn_save_voice,
            self.btn_reset_voice_defaults,
            self.ed_voice_display,
            self.ed_ref_audio,
            self.btn_browse_ref,
            self.ed_ref_text,
            self.spn_speed,
            self.spn_cfg,
            self.spn_nfe,
            self.spn_seed,
            self.cut_group,
        ]
        self._set_voice_edit_enabled(False)

        self.tabs.addTab(tab, "2. Lektor i Próbka")

    # =========================================================================
    # ZAKŁADKA 3: GENEROWANIE, KOREKTA I EKSPORT
    # =========================================================================
    def _build_generation_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Opcje batcha i Quality Guard
        ctrl_group = QGroupBox("Sterowanie generowaniem wsadowym")
        ctrl_lay = QVBoxLayout(ctrl_group)

        opts_row = QHBoxLayout()
        self.chk_qg = QCheckBox("Quality Guard (auto-weryfikacja Whisper CPU + retry zająknięć)")
        self.chk_qg.setChecked(True)
        opts_row.addWidget(self.chk_qg)
        opts_row.addSpacing(12)
        opts_row.addWidget(QLabel("Max retry:"))
        self.spn_qg_retries = QSpinBox()
        self.spn_qg_retries.setRange(0, 4)
        self.spn_qg_retries.setValue(2)
        opts_row.addWidget(self.spn_qg_retries)

        opts_row.addSpacing(12)
        opts_row.addWidget(QLabel("Próg QG:"))
        self.spn_qg_thresh = QDoubleSpinBox()
        self.spn_qg_thresh.setRange(0.55, 0.95)
        self.spn_qg_thresh.setSingleStep(0.02)
        self.spn_qg_thresh.setValue(0.78)
        opts_row.addWidget(self.spn_qg_thresh)

        opts_row.addSpacing(12)
        opts_row.addWidget(QLabel("Cooldown (s):"))
        self.spn_cooldown = QDoubleSpinBox()
        self.spn_cooldown.setRange(0.0, 5.0)
        self.spn_cooldown.setSingleStep(0.5)
        self.spn_cooldown.setValue(0.0)
        self.spn_cooldown.setToolTip("Opcjonalna przerwa między zdaniami dla chłodzenia laptopa w nocy")
        opts_row.addWidget(self.spn_cooldown)
        opts_row.addStretch()
        ctrl_lay.addLayout(opts_row)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start / Wznów generowanie")
        self.btn_start.setObjectName("primaryBtn")
        self.btn_start.clicked.connect(self._on_start_batch)

        self.btn_pause = QPushButton("⏸ Pauza")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._on_toggle_pause)

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setObjectName("dangerBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_batch)

        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_stop)
        btn_row.addSpacing(16)
        self.lbl_progress = QLabel("Postęp: 0 / 0 | Upłynęło: 0s | ETA: — | VRAM: —")
        btn_row.addWidget(self.lbl_progress, 1)
        ctrl_lay.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        ctrl_lay.addWidget(self.progress_bar)
        layout.addWidget(ctrl_group)

        # Tabela chunków + korekta pojedynczych zdań
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filtruj fragmenty:"))
        self.combo_chunk_filter = QComboBox()
        self.combo_chunk_filter.addItems([
            "Wszystkie (zaznaczone rozdziały)",
            "Tylko gotowe",
            "Tylko oczekujące",
            "Podejrzane (ocena QG < 0.85)",
            "Tylko błędy",
        ])
        self.combo_chunk_filter.currentIndexChanged.connect(self._refresh_chunks_table)
        filter_row.addWidget(self.combo_chunk_filter)

        btn_refresh_chunks = QPushButton("🔄 Odśwież listę")
        btn_refresh_chunks.clicked.connect(self._refresh_chunks_table)
        filter_row.addWidget(btn_refresh_chunks)
        filter_row.addStretch()

        self.btn_play_chunk = QPushButton("🔊 Odsłuchaj zaznaczony")
        self.btn_play_chunk.clicked.connect(self._on_play_selected_chunk)
        btn_stop_chunk = QPushButton("⏹ Cisza")
        btn_stop_chunk.clicked.connect(_stop_wav_playback)
        self.btn_regen_chunk = QPushButton("🔄 Regeneruj (nowy seed)")
        self.btn_regen_chunk.clicked.connect(lambda: self._on_regenerate_selected_chunk(edit_text=False))
        self.btn_edit_regen_chunk = QPushButton("✏ Edytuj tekst i regeneruj")
        self.btn_edit_regen_chunk.clicked.connect(lambda: self._on_regenerate_selected_chunk(edit_text=True))

        filter_row.addWidget(self.btn_play_chunk)
        filter_row.addWidget(btn_stop_chunk)
        filter_row.addWidget(self.btn_regen_chunk)
        filter_row.addWidget(self.btn_edit_regen_chunk)
        layout.addLayout(filter_row)

        self.tbl_chunks = QTableWidget(0, 7)
        self.tbl_chunks.setHorizontalHeaderLabels(
            ["Rozdział", "#", "Status", "Czas", "QG", "Tekst (do przeczytania)", "Usłyszano (Whisper)"]
        )
        self.tbl_chunks.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_chunks.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tbl_chunks.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tbl_chunks.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tbl_chunks.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.tbl_chunks.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.tbl_chunks.horizontalHeader().setSectionResizeMode(6, QHeaderView.Interactive)
        self.tbl_chunks.setColumnWidth(6, 240)
        self.tbl_chunks.setAlternatingRowColors(True)
        self.tbl_chunks.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_chunks.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.tbl_chunks, 2)

        # Panel eksportu
        exp_group = QGroupBox("Mastering i Eksport (FFmpeg EBU R128)")
        exp_lay = QHBoxLayout(exp_group)
        self.chk_exp_m4b = QCheckBox("M4B (spis rozdziałów + okładka)")
        self.chk_exp_m4b.setChecked(True)
        self.chk_exp_mp3 = QCheckBox("MP3 (per rozdział)")
        self.chk_exp_mp3.setChecked(False)
        self.chk_exp_norm = QCheckBox("Normalizacja EBU R128 (-16 LUFS)")
        self.chk_exp_norm.setChecked(True)

        self.combo_bitrate = QComboBox()
        for label, val in [
            ("48 kbps (mały plik)", "48k"),
            ("64 kbps (zalecane)", "64k"),
            ("96 kbps (wysokie)", "96k"),
            ("128 kbps (studyjne)", "128k"),
            ("160 kbps (maks.)", "160k"),
        ]:
            self.combo_bitrate.addItem(label, val)
        self.combo_bitrate.setCurrentIndex(1)  # domyślnie 64k

        self.combo_pause_scale = QComboBox()
        for label, val in [
            ("0.8x (krótsze)", 0.8),
            ("1.0x (domyślne 0.6s)", 1.0),
            ("1.3x (spokojniejsze)", 1.3),
            ("1.6x (długie)", 1.6),
        ]:
            self.combo_pause_scale.addItem(label, val)
        self.combo_pause_scale.setCurrentIndex(1)
        self.combo_pause_scale.setToolTip(
            "Skaluje długość przerw między zdaniami i akapitami podczas składania audiobooka (bez ponownego generowania GPU)"
        )

        self.btn_export = QPushButton("📦 Eksportuj Audiobooka")
        self.btn_export.setObjectName("primaryBtn")
        self.btn_export.clicked.connect(self._on_export)
        btn_open_out = QPushButton("📂 Otwórz output/")
        btn_open_out.clicked.connect(lambda: os.startfile(str(OUTPUT_DIR.resolve())))

        exp_lay.addWidget(self.chk_exp_m4b)
        exp_lay.addWidget(self.chk_exp_mp3)
        exp_lay.addWidget(self.chk_exp_norm)
        exp_lay.addSpacing(8)
        exp_lay.addWidget(QLabel("Bitrate:"))
        exp_lay.addWidget(self.combo_bitrate)
        exp_lay.addSpacing(8)
        exp_lay.addWidget(QLabel("Pauzy:"))
        exp_lay.addWidget(self.combo_pause_scale)
        exp_lay.addStretch()
        exp_lay.addWidget(self.btn_export)
        exp_lay.addWidget(btn_open_out)
        layout.addWidget(exp_group)

        # Log na dole
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(130)
        layout.addWidget(self.txt_log)

        self.tabs.addTab(tab, "3. Generowanie i Eksport")

    # =========================================================================
    # ZAKŁADKA 4: SŁOWNIK WYMOWY
    # =========================================================================
    def _build_pronunciation_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        info = QLabel(
            "Słownik wymowy (pronunciation.json) zamienia obce słowa, skróty i nazwy własne "
            "na zapis fonetyczny przed syntezą (np. audiobooka → audiobuka, NVIDIA → Enwidia). "
            "Zmiany działają dla nowych projektów oraz w teście 1 zdania."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.tbl_pron = QTableWidget(0, 2)
        self.tbl_pron.setHorizontalHeaderLabels(["Słowo w tekście książki", "Jak lektor ma je przeczytać"])
        self.tbl_pron.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_pron.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_pron.setAlternatingRowColors(True)
        layout.addWidget(self.tbl_pron, 1)

        btn_bar = QHBoxLayout()
        btn_add = QPushButton("➕ Dodaj wiersz")
        btn_add.clicked.connect(self._on_pron_add_row)
        btn_del = QPushButton("🗑 Usuń zaznaczony")
        btn_del.clicked.connect(self._on_pron_delete_row)
        btn_save = QPushButton("💾 Zapisz słownik")
        btn_save.setObjectName("primaryBtn")
        btn_save.clicked.connect(self._on_pron_save)
        btn_bar.addWidget(btn_add)
        btn_bar.addWidget(btn_del)
        btn_bar.addStretch()
        btn_bar.addWidget(btn_save)
        layout.addLayout(btn_bar)

        self.tabs.addTab(tab, "4. Słownik Wymowy")

    # =========================================================================
    # LOGIKA: KSIĄŻKI I PROJEKTY
    # =========================================================================
    def _log(self, msg: str) -> None:
        self.txt_log.appendPlainText(msg)

    def _refresh_saved_projects(self) -> None:
        self.combo_projects.clear()
        for proj_dir in list_saved_projects(WORK_DIR):
            self.combo_projects.addItem(proj_dir.name, str(proj_dir))

    def _on_open_book(self) -> None:
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz ebooka EPUB lub TXT",
            "",
            "E-booki (*.epub *.txt);;Wszystkie pliki (*.*)",
        )
        if not fpath:
            return
        try:
            book = parse_book(Path(fpath))
            pron = load_pronunciation_dict()
            voice = self._current_voice_from_ui()
            base_seed = voice.seed if voice else 101
            self.db = ProjectDB.create_from_book(
                book,
                pronunciation_dict=pron,
                work_root=WORK_DIR,
                base_seed=base_seed,
            )
            self._refresh_saved_projects()
            self._load_project_into_ui()
            self._log(f"Wczytano książkę: {book.title} ({len(book.chapters)} rozdziałów).")
        except Exception as e:
            QMessageBox.critical(self, "Błąd wczytywania książki", str(e))

    def _on_load_saved_project(self) -> None:
        proj_str = self.combo_projects.currentData()
        if not proj_str:
            return
        try:
            self.db = ProjectDB(Path(proj_str))
            self._load_project_into_ui()
            self._log(f"Wznowiono projekt: {self.db.project_dir.name}")
        except Exception as e:
            QMessageBox.critical(self, "Błąd otwierania projektu", str(e))

    def _load_project_into_ui(self) -> None:
        if self.db is None:
            return
        meta = self.db.get_all_meta()
        self.lbl_book_title.setText(meta.get("title", self.db.project_dir.name))
        self.lbl_book_author.setText(f"Autor: {meta.get('author', '—')}")
        self.lbl_book_path.setText(f"Projekt SQLite: {self.db.project_dir}")

        cover_path = Path(meta["cover_path"]) if meta.get("cover_path") else None
        if cover_path and cover_path.exists():
            pix = QPixmap(str(cover_path)).scaled(96, 136, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_cover.setPixmap(pix)
        else:
            self.lbl_cover.setPixmap(QPixmap())
            self.lbl_cover.setText("Brak okładki")

        self._refresh_chapters_table()
        self._refresh_chunks_table()
        self._update_progress_widgets()

    def _refresh_chapters_table(self) -> None:
        if self.db is None:
            return
        chapters = self.db.list_chapters()
        self.tbl_chapters.blockSignals(True)
        self.tbl_chapters.setRowCount(len(chapters))
        sel_ch = 0
        sel_chunks = 0
        for row_idx, ch in enumerate(chapters):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk.setCheckState(Qt.Checked if ch.selected else Qt.Unchecked)
            chk.setData(Qt.UserRole, ch.id)

            title_item = QTableWidgetItem(ch.title)
            title_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

            cnt_item = QTableWidgetItem(str(ch.total_chunks))
            cnt_item.setTextAlignment(Qt.AlignCenter)
            cnt_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

            prog_item = QTableWidgetItem(f"{ch.done_chunks}/{ch.total_chunks}")
            prog_item.setTextAlignment(Qt.AlignCenter)
            prog_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

            self.tbl_chapters.setItem(row_idx, 0, chk)
            self.tbl_chapters.setItem(row_idx, 1, title_item)
            self.tbl_chapters.setItem(row_idx, 2, cnt_item)
            self.tbl_chapters.setItem(row_idx, 3, prog_item)

            if ch.selected:
                sel_ch += 1
                sel_chunks += ch.total_chunks

        self.tbl_chapters.blockSignals(False)
        est_min = round(sel_chunks * 6.0 / 60.0)
        self.lbl_book_stats.setText(
            f"Zaznaczone rozdziały: {sel_ch}/{len(chapters)} | "
            f"Fragmenty: {sel_chunks} | Szacowana długość audiobooka: ~{est_min} min"
        )

    def _set_all_chapters(self, selected: bool) -> None:
        if self.db is None:
            return
        for ch in self.db.list_chapters():
            self.db.set_chapter_selected(ch.id, selected)
        self._refresh_chapters_table()
        self._refresh_chunks_table()
        self._update_progress_widgets()

    def _on_chapter_checkbox_changed(self, item: QTableWidgetItem) -> None:
        if self.db is None or item.column() != 0:
            return
        ch_id = item.data(Qt.UserRole)
        if ch_id is None:
            return
        self.db.set_chapter_selected(int(ch_id), item.checkState() == Qt.Checked)
        self._refresh_chapters_table()
        self._refresh_chunks_table()
        self._update_progress_widgets()

    def _on_chapter_selected_preview(self) -> None:
        if self.db is None:
            return
        rows = self.tbl_chapters.selectionModel().selectedRows()
        if not rows:
            return
        item0 = self.tbl_chapters.item(rows[0].row(), 0)
        if item0 is None:
            return
        ch_id = int(item0.data(Qt.UserRole))
        chunks = self.db.list_chunks(chapter_id=ch_id)
        lines = [f"[{c.chunk_index + 1:03d}] ({c.pause_after_ms}ms)  {c.normalized_text}" for c in chunks]
        self.txt_chapter_preview.setPlainText("\n".join(lines))

    # =========================================================================
    # LOGIKA: PROFILE LEKTORÓW I WYCINARKA
    # =========================================================================
    def _refresh_voices(self, select_name: str | None = None) -> None:
        profiles = list_voice_profiles(VOICES_DIR)
        self.combo_voices.blockSignals(True)
        self.combo_voices.clear()
        sel_idx = 0
        for i, p in enumerate(profiles):
            self.combo_voices.addItem(f"{p.display_name} ({p.name})", p)
            if select_name and p.name == select_name:
                sel_idx = i
        self.combo_voices.blockSignals(False)
        if profiles:
            self.combo_voices.setCurrentIndex(sel_idx)
            self._populate_voice_fields(profiles[sel_idx])

    def _populate_voice_fields(self, p: VoiceProfile) -> None:
        self.ed_voice_display.setText(p.display_name)
        self.ed_ref_audio.setText(str(p.ref_audio))
        self.ed_ref_text.setPlainText(p.ref_text)
        self.spn_speed.setValue(p.speed)
        self.spn_cfg.setValue(p.cfg_strength)
        self.spn_nfe.setValue(p.nfe_step)
        self.spn_seed.setValue(p.seed)

    def _on_voice_selected(self, idx: int) -> None:
        p = self.combo_voices.itemData(idx)
        if isinstance(p, VoiceProfile):
            self._populate_voice_fields(p)

    def _current_voice_from_ui(self) -> VoiceProfile | None:
        p = self.combo_voices.currentData()
        if not isinstance(p, VoiceProfile):
            return None
        return VoiceProfile(
            name=p.name,
            display_name=self.ed_voice_display.text().strip() or p.name,
            ref_audio=self.ed_ref_audio.text().strip(),
            ref_text=self.ed_ref_text.toPlainText().strip(),
            speed=float(self.spn_speed.value()),
            cfg_strength=float(self.spn_cfg.value()),
            nfe_step=int(self.spn_nfe.value()),
            seed=int(self.spn_seed.value()),
        )

    def _on_new_voice_profile(self) -> None:
        name, ok = QInputDialog.getText(self, "Nowy profil lektora", "Podaj nazwę nowego lektora:")
        if not ok or not name.strip():
            return
        slug = slugify_voice_name(name)
        profile = VoiceProfile(
            name=slug,
            display_name=name.strip(),
            ref_audio="ref.wav",
            ref_text="",
        )
        save_voice_profile(profile, VOICES_DIR)
        self._refresh_voices(select_name=slug)

    def _set_voice_edit_enabled(self, enabled: bool) -> None:
        for w in self._voice_editable_widgets:
            w.setEnabled(enabled)
        self.chk_edit_voice.setText("🔓 Edytuj ustawienia lektora (odblokowane)" if enabled else "🔒 Edytuj ustawienia lektora")

    def _on_toggle_voice_edit_lock(self, checked: bool) -> None:
        if checked:
            reply = QMessageBox.warning(
                self,
                "Uwaga — edycja lektora",
                "Możesz zepsuć swojego ulubionego lektora! Czy na pewno chcesz odblokować edycję parametrów?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self.chk_edit_voice.blockSignals(True)
                self.chk_edit_voice.setChecked(False)
                self.chk_edit_voice.blockSignals(False)
                self._set_voice_edit_enabled(False)
                return
            self._set_voice_edit_enabled(True)
        else:
            self._set_voice_edit_enabled(False)

    def _on_reset_voice_defaults(self) -> None:
        profile = self._current_voice_from_ui()
        if profile is None:
            return
        reply = QMessageBox.question(
            self,
            "Przywrócić domyślne?",
            "Przywrócić wypracowane domyślne parametry syntezy (speed=1.15, CFG=2.00, ODE=32, Seed=101)"
            + (" oraz oryginalną próbkę ref_7s.wav?" if profile.name == "moj_lektor" else "?"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.spn_speed.setValue(1.15)
        self.spn_cfg.setValue(2.0)
        self.spn_nfe.setValue(32)
        self.spn_seed.setValue(101)
        if profile.name == "moj_lektor":
            self.ed_voice_display.setText("Mój Lektor (domyślny)")
            self.ed_ref_audio.setText("ref_7s.wav")
            self.ed_ref_text.setPlainText(
                "Smród panujący w chlewni zdawał się mieć niemal płynną formę."
            )
        save_voice_profile(self._current_voice_from_ui(), VOICES_DIR)
        self._refresh_voices(select_name=profile.name)
        QMessageBox.information(self, "Przywrócono", "Przywrócono i zapisano domyślne ustawienia lektora.")

    def _on_save_voice_profile(self) -> None:
        profile = self._current_voice_from_ui()
        if profile is None:
            return
        save_voice_profile(profile, VOICES_DIR)
        self._refresh_voices(select_name=profile.name)
        QMessageBox.information(self, "Zapisano", f"Zapisano profil lektora: {profile.display_name}")

    def _on_browse_ref_wav(self) -> None:
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Wybierz plik WAV próbki głosu", str(VOICES_DIR), "Pliki audio (*.wav)"
        )
        if fpath:
            self.ed_ref_audio.setText(fpath)

    def _on_play_ref_wav(self) -> None:
        profile = self._current_voice_from_ui()
        if profile and profile.ref_audio_path.exists():
            _play_wav_async(profile.ref_audio_path)
        else:
            QMessageBox.warning(self, "Brak pliku", "Nie znaleziono pliku próbki referencyjnej WAV.")

    def _on_browse_cut_source(self) -> None:
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz plik źródłowy audiobooka",
            "",
            "Pliki audio (*.mp3 *.m4b *.m4a *.wav *.flac *.ogg);;Wszystkie (*.*)",
        )
        if fpath:
            self.ed_cut_src.setText(fpath)

    def _on_cut_sample(self) -> None:
        profile = self._current_voice_from_ui()
        if profile is None:
            QMessageBox.warning(self, "Brak profilu", "Najpierw wybierz lub utwórz profil lektora.")
            return
        src = Path(self.ed_cut_src.text().strip())
        if not src.exists():
            QMessageBox.warning(self, "Brak pliku", "Wskaż istniejący plik audio do wycięcia.")
            return
        out_wav = profile.profile_dir / "ref.wav"
        try:
            cut_reference_sample(
                src,
                out_wav,
                start=self.ed_cut_start.text().strip(),
                duration=float(self.spn_cut_dur.value()),
            )
            self.ed_ref_audio.setText("ref.wav")
            save_voice_profile(self._current_voice_from_ui(), VOICES_DIR)
            _play_wav_async(out_wav)
            QMessageBox.information(
                self,
                "Wycięto próbkę",
                f"Zapisano {out_wav.name} i uruchomiono odsłuch.\n"
                "Upewnij się, że wpisałeś dokładną transkrypcję w polu powyżej i zapisz profil!",
            )
        except Exception as e:
            QMessageBox.critical(self, "Błąd FFmpeg", str(e))

    def _on_run_single_test(self) -> None:
        if self.batch_worker and self.batch_worker.isRunning():
            QMessageBox.warning(self, "Zajęte", "Najpierw zatrzymaj generowanie wsadowe.")
            return
        profile = self._current_voice_from_ui()
        if profile is None or not profile.ref_audio_path.exists() or not profile.ref_text:
            QMessageBox.warning(
                self,
                "Niepełny profil lektora",
                "Uzupełnij istniejący plik próbki WAV oraz jego dokładną transkrypcję.",
            )
            return

        raw_text = self.ed_test_sentence.text().strip()
        if not raw_text:
            return
        norm_text = normalize_polish_text(raw_text, load_pronunciation_dict())

        self.btn_run_test.setEnabled(False)
        self.lbl_test_status.setText("Generuję próbkę na GPU...")
        out_wav = OUTPUT_DIR / "_voice_test.wav"
        self.test_worker = SingleTestWorker(self.engine, profile, norm_text, out_wav)
        self.test_worker.finished_ok.connect(self._on_single_test_done)
        self.test_worker.failed.connect(self._on_single_test_failed)
        self.test_worker.start()

    def _on_single_test_done(self, wav_path: str, dur: float) -> None:
        self.btn_run_test.setEnabled(True)
        self.lbl_test_status.setText(f"Gotowe ({dur:.2f}s) — odtwarzam...")
        _play_wav_async(Path(wav_path))

    def _on_single_test_failed(self, err: str) -> None:
        self.btn_run_test.setEnabled(True)
        self.lbl_test_status.setText("Błąd!")
        QMessageBox.critical(self, "Błąd syntezy testowej", err)

    # =========================================================================
    # LOGIKA: GENEROWANIE WSADOWE, TABELA CHUNKÓW I EKSPORT
    # =========================================================================
    def _current_batch_options(self) -> BatchOptions:
        return BatchOptions(
            use_quality_guard=self.chk_qg.isChecked(),
            qg_max_retries=int(self.spn_qg_retries.value()),
            qg_min_similarity=float(self.spn_qg_thresh.value()),
            cooldown_sec=float(self.spn_cooldown.value()),
        )

    def _current_elapsed_sec(self) -> float:
        total = self._elapsed_base_sec
        if self._run_segment_start is not None:
            total += max(0.0, time.monotonic() - self._run_segment_start)
        return total

    def _on_elapsed_tick(self) -> None:
        self._update_progress_widgets(self._last_progress)

    def _update_progress_widgets(self, prog: BatchProgress | None = None) -> None:
        if prog is not None:
            self._last_progress = prog
            done, err, total = prog.done, prog.errors, prog.total
            eta_s = _format_eta(prog.eta_sec)
            vram_s = f"{prog.vram_mb:.0f} MiB" if prog.vram_mb > 0 else "—"
        elif self.db is not None:
            done, err, total = self.db.get_progress()
            eta_s = "—"
            vram_s = "—"
        else:
            done, err, total = 0, 0, 0
            eta_s = "—"
            vram_s = "—"

        elapsed_s = _format_eta(self._current_elapsed_sec())
        pct = int((done / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.lbl_progress.setText(
            f"Postęp: {done} / {total} ({pct}%) | Błędów: {err} | Upłynęło: {elapsed_s} | ETA: {eta_s} | VRAM: {vram_s}"
        )

    def _refresh_chunks_table(self) -> None:
        if self.db is None:
            self.tbl_chunks.setRowCount(0)
            self._chunk_rows_cache = []
            return

        all_chunks = self.db.list_chunks(only_selected=True)
        mode = self.combo_chunk_filter.currentIndex()
        if mode == 1:
            rows = [c for c in all_chunks if c.status == "done"]
        elif mode == 2:
            rows = [c for c in all_chunks if c.status == "pending"]
        elif mode == 3:
            rows = [c for c in all_chunks if c.status == "done" and c.qg_score is not None and c.qg_score < 0.85]
        elif mode == 4:
            rows = [c for c in all_chunks if c.status == "error"]
        else:
            rows = all_chunks

        self._chunk_rows_cache = rows
        self.tbl_chunks.setRowCount(len(rows))
        for i, c in enumerate(rows):
            self.tbl_chunks.setItem(i, 0, QTableWidgetItem(c.chapter_title[:28]))
            self.tbl_chunks.setItem(i, 1, QTableWidgetItem(str(c.chunk_index + 1)))

            st_item = QTableWidgetItem(c.status)
            if c.status == "done":
                st_item.setForeground(QColor("#57d9a3"))
            elif c.status == "error":
                st_item.setForeground(QColor("#ff6b6b"))
            self.tbl_chunks.setItem(i, 2, st_item)

            dur_s = f"{c.duration_sec:.1f}s" if c.duration_sec > 0 else "—"
            self.tbl_chunks.setItem(i, 3, QTableWidgetItem(dur_s))

            qg_s = f"{c.qg_score:.2f}" if c.qg_score is not None else "—"
            qg_item = QTableWidgetItem(qg_s)
            if c.qg_score is not None and c.qg_score < 0.80:
                qg_item.setForeground(QColor("#f5a623"))
            self.tbl_chunks.setItem(i, 4, qg_item)

            self.tbl_chunks.setItem(i, 5, QTableWidgetItem(c.normalized_text))
            self.tbl_chunks.setItem(i, 6, QTableWidgetItem(c.heard_text or c.error_msg))

    def _on_start_batch(self) -> None:
        if self.db is None:
            QMessageBox.warning(self, "Brak projektu", "Najpierw wczytaj książkę w zakładce 1.")
            return
        voice = self._current_voice_from_ui()
        if voice is None or not voice.ref_audio_path.exists() or not voice.ref_text:
            QMessageBox.warning(self, "Brak profilu lektora", "Sprawdź ustawienia lektora w zakładce 2.")
            return

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸ Pauza")
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)

        self._elapsed_base_sec = 0.0
        self._run_segment_start = time.monotonic()
        self._last_progress = None
        self._elapsed_timer.start()
        self._update_progress_widgets()

        self.batch_worker = BatchWorker(self.engine, self.db, voice, self._current_batch_options())
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.log.connect(self._log)
        self.batch_worker.finished_ok.connect(self._on_batch_finished)
        self.batch_worker.failed.connect(self._on_batch_failed)
        self.batch_worker.start()

    def _on_toggle_pause(self) -> None:
        if not self.batch_worker:
            return
        paused = not self.batch_worker.pause_event.is_set()
        self.batch_worker.set_paused(paused)
        if paused:
            if self._run_segment_start is not None:
                self._elapsed_base_sec += max(0.0, time.monotonic() - self._run_segment_start)
                self._run_segment_start = None
        else:
            self._run_segment_start = time.monotonic()
        self.btn_pause.setText("▶ Wznów" if paused else "⏸ Pauza")
        self._log("Wstrzymano (pauza) — dokończę bieżące zdanie." if paused else "Wznowiono generowanie.")

    def _on_stop_batch(self) -> None:
        if self.batch_worker:
            self.batch_worker.request_stop()
            self._log("Zatrzymuję po bieżącym zdaniu...")

    def _on_batch_progress(self, prog: BatchProgress) -> None:
        self._update_progress_widgets(prog)
        if prog.done % 5 == 0 or prog.done == prog.total:
            self._refresh_chapters_table()
            self._refresh_chunks_table()

    def _on_batch_finished(self, prog: BatchProgress) -> None:
        if self._run_segment_start is not None:
            self._elapsed_base_sec += max(0.0, time.monotonic() - self._run_segment_start)
            self._run_segment_start = None
        self._elapsed_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(True)
        self._update_progress_widgets(prog)
        self._refresh_chapters_table()
        self._refresh_chunks_table()
        self._log(prog.message)

    def _on_batch_failed(self, err: str) -> None:
        if self._run_segment_start is not None:
            self._elapsed_base_sec += max(0.0, time.monotonic() - self._run_segment_start)
            self._run_segment_start = None
        self._elapsed_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(True)
        self._log(f"[BŁĄD KRYTYCZNY] {err}")
        QMessageBox.critical(self, "Błąd generowania", err)

    def _selected_chunk_row(self) -> ChunkRow | None:
        rows = self.tbl_chunks.selectionModel().selectedRows()
        if not rows:
            return None
        idx = rows[0].row()
        if 0 <= idx < len(self._chunk_rows_cache):
            return self._chunk_rows_cache[idx]
        return None

    def _on_play_selected_chunk(self) -> None:
        c = self._selected_chunk_row()
        if c is None or not c.audio_path or not Path(c.audio_path).exists():
            QMessageBox.information(self, "Brak audio", "Zaznaczony fragment nie ma jeszcze pliku WAV.")
            return
        _play_wav_async(Path(c.audio_path))

    def _on_regenerate_selected_chunk(self, edit_text: bool) -> None:
        if self.batch_worker and self.batch_worker.isRunning():
            QMessageBox.warning(self, "Zajęte", "Najpierw zatrzymaj generowanie wsadowe.")
            return
        c = self._selected_chunk_row()
        if c is None or self.db is None:
            QMessageBox.information(self, "Wybierz wiersz", "Zaznacz w tabeli fragment do regeneracji.")
            return
        voice = self._current_voice_from_ui()
        if voice is None:
            return

        new_text: str | None = None
        if edit_text:
            txt, ok = QInputDialog.getText(
                self,
                "Edytuj tekst fragmentu",
                "Możesz poprawić zapis fonetyczny lub interpunkcję:",
                text=c.normalized_text,
            )
            if not ok or not txt.strip():
                return
            new_text = txt.strip()

        self.btn_regen_chunk.setEnabled(False)
        self.btn_edit_regen_chunk.setEnabled(False)
        self._log(f"Regeneruję fragment #{c.chunk_index + 1} ({c.chapter_title})...")

        self.regen_worker = RegenerateChunkWorker(
            self.engine,
            self.db,
            c.id,
            voice,
            self._current_batch_options(),
            new_text=new_text,
        )
        self.regen_worker.log.connect(self._log)
        self.regen_worker.finished_ok.connect(self._on_regen_done)
        self.regen_worker.failed.connect(self._on_regen_failed)
        self.regen_worker.start()

    def _on_regen_done(self, updated: ChunkRow) -> None:
        self.btn_regen_chunk.setEnabled(True)
        self.btn_edit_regen_chunk.setEnabled(True)
        self._refresh_chunks_table()
        self._update_progress_widgets()
        self._log(f"Zregenerowano #{updated.chunk_index + 1} (seed={updated.seed}, {updated.duration_sec:.1f}s)")
        if updated.audio_path:
            _play_wav_async(Path(updated.audio_path))

    def _on_regen_failed(self, err: str) -> None:
        self.btn_regen_chunk.setEnabled(True)
        self.btn_edit_regen_chunk.setEnabled(True)
        QMessageBox.critical(self, "Błąd regeneracji", err)

    def _on_export(self) -> None:
        if self.db is None:
            QMessageBox.warning(self, "Brak projektu", "Najpierw wczytaj projekt.")
            return
        if not self.chk_exp_m4b.isChecked() and not self.chk_exp_mp3.isChecked():
            QMessageBox.warning(self, "Wybierz format", "Zaznacz przynajmniej M4B lub MP3.")
            return

        self.btn_export.setEnabled(False)
        self.btn_export.setText("⏳ Eksportuję...")
        self.progress_bar.setRange(0, 0)  # animowany pasek w trakcie masteringu FFmpeg
        bitrate = str(self.combo_bitrate.currentData() or "64k")
        pause_scale = float(self.combo_pause_scale.currentData() or 1.0)
        self.lbl_progress.setText(f"⏳ Trwa mastering EBU R128 i eksport ({bitrate})...")
        self._log(f"Rozpoczynam mastering i eksport audiobooka (bitrate: {bitrate}, pauzy: {pause_scale:.1f}x)...")
        self.export_worker = ExportWorker(
            db=self.db,
            export_mp3=self.chk_exp_mp3.isChecked(),
            export_m4b=self.chk_exp_m4b.isChecked(),
            normalize_audio=self.chk_exp_norm.isChecked(),
            bitrate=bitrate,
            pause_scale=pause_scale,
            output_root=OUTPUT_DIR,
        )
        self.export_worker.log.connect(self._log)
        self.export_worker.finished_ok.connect(self._on_export_done)
        self.export_worker.failed.connect(self._on_export_failed)
        self.export_worker.start()

    def _on_export_done(self, res: ExportResult) -> None:
        self.btn_export.setEnabled(True)
        self.btn_export.setText("📦 Eksportuj Audiobooka")
        self.progress_bar.setRange(0, 100)
        self._update_progress_widgets(self._last_progress)
        mins = res.total_duration_sec / 60.0
        msg = (
            f"Audiobook gotowy ({mins:.1f} min)!\n\n"
            f"Folder: {res.output_dir}\n"
            f"M4B: {res.m4b_file.name if res.m4b_file else 'nie wybrano'}\n"
            f"Plików MP3: {len(res.mp3_files)}"
        )
        QMessageBox.information(self, "Eksport zakończony", msg)

    def _on_export_failed(self, err: str) -> None:
        self.btn_export.setEnabled(True)
        self.btn_export.setText("📦 Eksportuj Audiobooka")
        self.progress_bar.setRange(0, 100)
        self._update_progress_widgets(self._last_progress)
        self._log(f"[BŁĄD EKSPORTU] {err}")
        QMessageBox.critical(self, "Błąd eksportu", err)

    # =========================================================================
    # LOGIKA: SŁOWNIK WYMOWY
    # =========================================================================
    def _load_pronunciation_table(self) -> None:
        mapping = load_pronunciation_dict()
        self.tbl_pron.setRowCount(len(mapping))
        for i, (k, v) in enumerate(sorted(mapping.items(), key=lambda kv: kv[0].lower())):
            self.tbl_pron.setItem(i, 0, QTableWidgetItem(k))
            self.tbl_pron.setItem(i, 1, QTableWidgetItem(v))

    def _on_pron_add_row(self) -> None:
        r = self.tbl_pron.rowCount()
        self.tbl_pron.insertRow(r)
        self.tbl_pron.setItem(r, 0, QTableWidgetItem(""))
        self.tbl_pron.setItem(r, 1, QTableWidgetItem(""))
        self.tbl_pron.editItem(self.tbl_pron.item(r, 0))

    def _on_pron_delete_row(self) -> None:
        rows = sorted({idx.row() for idx in self.tbl_pron.selectedIndexes()}, reverse=True)
        for r in rows:
            self.tbl_pron.removeRow(r)

    def _on_pron_save(self) -> None:
        mapping: dict[str, str] = {}
        for r in range(self.tbl_pron.rowCount()):
            k_item = self.tbl_pron.item(r, 0)
            v_item = self.tbl_pron.item(r, 1)
            k = k_item.text().strip() if k_item else ""
            v = v_item.text().strip() if v_item else ""
            if k and v:
                mapping[k] = v
        save_pronunciation_dict(mapping)
        QMessageBox.information(self, "Zapisano", f"Zapisano słownik wymowy ({len(mapping)} wpisów).")

    def closeEvent(self, event) -> None:
        _stop_wav_playback()
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.request_stop()
            self.batch_worker.wait(5000)
        self.engine.unload()
        super().closeEvent(event)
