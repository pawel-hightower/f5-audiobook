"""Główne okno aplikacji desktopowej F5-Audiobook (PySide6)."""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
import winsound
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
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
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from f5_audiobook.audio.exporter import ExportResult, _safe_filename
from f5_audiobook.config import (
    APP_TITLE,
    OUTPUT_DIR,
    VOICES_DIR,
    WORK_DIR,
    load_pronunciation_dict,
    save_pronunciation_dict,
)
from f5_audiobook.engine.bielik_tagger import is_bielik_available
from f5_audiobook.engine.f5_ipa import F5IpaEngine
from f5_audiobook.gui.player_widget import AudiobookPlayerWidget
from f5_audiobook.gui.voice_panel import VoiceRolePanel
from f5_audiobook.gui.workers import (
    AiPreloadWorker,
    BatchWorker,
    BielikTaggerWorker,
    ExportWorker,
    RegenerateChunkWorker,
    SingleTestWorker,
)
from f5_audiobook.pipeline.batch_runner import BatchOptions, BatchProgress
from f5_audiobook.pipeline.project_db import (
    ChapterRow,
    ChunkRow,
    ProjectDB,
    list_saved_projects,
    slugify_project_name,
)
from f5_audiobook.text.epub_parser import parse_book
from f5_audiobook.text.normalizer_pl import normalize_polish_text
from f5_audiobook.voices import (
    VoiceProfile,
    list_voice_profiles,
    load_voice_roles,
    save_voice_roles,
)

# Indeksy kolumn tabeli fragmentów w Zakładce 3
_COL_PLAY = 0
_COL_VOICE = 1
_COL_CHAPTER = 2
_COL_NUM = 3
_COL_STATUS = 4
_COL_DUR = 5
_COL_QG = 6
_COL_TEXT = 7
_COL_HEARD = 8
_COL_COUNT = 9

_VOICE_ROLE_DISPLAY: dict[str, tuple[str, str, str]] = {
    "narrator": ("🎙 N", "#8b949e", "Narrator — kliknij, aby przełączyć rolę (N → M → K)"),
    "male": ("👨 M", "#6cb6ff", "Dialog męski — kliknij, aby przełączyć rolę (N → M → K)"),
    "female": ("👩 K", "#f778ba", "Dialog żeński — kliknij, aby przełączyć rolę (N → M → K)"),
    "dialogue": ("💬 ?", "#f0b429", "Dialog nieprzypisany — uruchom Bielika AI lub kliknij, aby wybrać ręcznie"),
}
_VOICE_ROLE_CYCLE: dict[str, str] = {
    "narrator": "male",
    "male": "female",
    "female": "narrator",
    "dialogue": "male",
}


_PART_HEADING_RE = re.compile(
    r"^\s*(część|czesc|księga|ksiega|tom|akt|part|book|volume)\b",
    re.IGNORECASE,
)


def _is_part_heading(title: str) -> bool:
    return bool(_PART_HEADING_RE.match(title or ""))


def _group_chapters_into_parts(chapters: list[ChapterRow]) -> list[tuple[ChapterRow, list[ChapterRow]]]:
    """Grupuje rozdziały w pary (nagłówek_części_lub_samodzielny_rozdział, lista_podrozdziałów)."""
    groups: list[tuple[ChapterRow, list[ChapterRow]]] = []
    current_part_idx: int | None = None
    for ch in chapters:
        if _is_part_heading(ch.title):
            groups.append((ch, []))
            current_part_idx = len(groups) - 1
        elif current_part_idx is not None:
            groups[current_part_idx][1].append(ch)
        else:
            groups.append((ch, []))
    return groups


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
        self.setWindowTitle(APP_TITLE)
        self.resize(1220, 820)

        self.engine = F5IpaEngine()
        self.db: ProjectDB | None = None
        self.batch_worker: BatchWorker | None = None
        self.test_worker: SingleTestWorker | None = None
        self.regen_worker: RegenerateChunkWorker | None = None
        self.export_worker: ExportWorker | None = None
        self.bielik_worker: BielikTaggerWorker | None = None
        self.preload_worker: AiPreloadWorker | None = None
        self._ai_libs_ready: bool = False
        self._ai_device_info: str = ""
        self._active_test_role: str = "narrator"
        self._chunk_rows_cache: list[ChunkRow] = []
        self._tbl_row_meta: list[tuple[str, object]] = []  # ("group", group_id) | ("chunk", ChunkRow)
        self._chunk_id_to_row: dict[int, int] = {}
        self._group_id_to_row: dict[int, int] = {}
        self._chapter_to_part_id: dict[int, int] = {}
        self._part_titles: dict[int, str] = {}
        self._chapters_cache: list[ChapterRow] = []
        self._tab1_collapsed_part_ids: set[int] = set()
        self._tab3_collapsed_groups: set[int] = set()
        self._tab3_active_group_ids: list[int] = []
        self._last_chapter_id_seen: int | None = None
        self._chunks_sort_col: int = _COL_NUM
        self._chunks_sort_order: Qt.SortOrder = Qt.AscendingOrder

        # Licznik czasu od startu generowania
        self._elapsed_base_sec: float = 0.0
        self._run_segment_start: float | None = None
        self._last_progress: BatchProgress | None = None
        self._batch_state: str = "idle"  # idle | running | paused | stopping | stopped | done | error
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._on_elapsed_tick)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._voice_help_map: dict[int, tuple[str, str]] = {}
        self._saved_book_meta: tuple[str, str, str] = ("", "", "")

        self._build_book_tab()
        self._build_voice_tab()
        self._build_generation_tab()
        self._build_pronunciation_tab()
        self._build_player_tab()
        self._build_global_status_bar()
        self._setup_player_shortcuts()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._refresh_saved_projects()
        self._refresh_voices()
        self._load_pronunciation_table()
        self._set_ai_generation_buttons_enabled(False)
        self.lbl_global_status.setText(
            "⏳ Wczytuję biblioteki AI (PyTorch / CUDA) w tle — możesz już otwierać projekty i słuchać..."
        )
        self.status_progress_bar.setRange(0, 0)
        self.status_progress_bar.setVisible(True)
        QTimer.singleShot(80, self._start_ai_preload)

    def _set_ai_generation_buttons_enabled(self, ready: bool) -> None:
        self.btn_start.setEnabled(ready)
        self.btn_regen_chunk.setEnabled(ready)
        tip_wait = "Trwa wczytywanie bibliotek AI (PyTorch / CUDA) w tle — za kilka sekund będzie gotowe."
        self.btn_start.setToolTip("" if ready else tip_wait)
        for panel in self._voice_panels.values():
            panel.btn_run_test.setEnabled(ready)
            panel.btn_run_test.setToolTip("" if ready else tip_wait)

    def _start_ai_preload(self) -> None:
        if self._ai_libs_ready or (self.preload_worker and self.preload_worker.isRunning()):
            return
        self.preload_worker = AiPreloadWorker()
        self.preload_worker.finished_ok.connect(self._on_ai_preload_done)
        self.preload_worker.failed.connect(self._on_ai_preload_failed)
        self.preload_worker.start()

    def _on_ai_preload_done(self, dev_info: str) -> None:
        self._ai_libs_ready = True
        self._ai_device_info = dev_info
        busy = any(
            w is not None and w.isRunning()
            for w in (self.batch_worker, self.bielik_worker, self.regen_worker, self.test_worker)
        )
        if not busy:
            self._set_ai_generation_buttons_enabled(True)
        self.status_progress_bar.setRange(0, 100)
        self._update_progress_widgets(self._last_progress)
        self.lbl_global_status.setText(f"✅ Biblioteki AI gotowe ({dev_info})")
        self._log(f"Wczytano biblioteki AI w tle ({dev_info}).")

    def _on_ai_preload_failed(self, err: str) -> None:
        self._ai_libs_ready = False
        self.status_progress_bar.setRange(0, 100)
        self.status_progress_bar.setVisible(False)
        self.lbl_global_status.setText(f"⚠️ Błąd ładowania bibliotek AI: {err}")
        self._log(f"[BŁĄD ŁADOWANIA AI] {err}")

    def _build_global_status_bar(self) -> None:
        sb = self.statusBar()
        sb.setStyleSheet(
            "QStatusBar { background-color: #13151a; border-top: 1px solid #2c313c; padding: 2px 8px; }"
        )
        self.lbl_global_status = QLabel("Gotowy")
        self.lbl_global_status.setStyleSheet("color: #cfd5df; font-size: 9pt;")
        sb.addWidget(self.lbl_global_status, 1)

        self.status_progress_bar = QProgressBar()
        self.status_progress_bar.setRange(0, 100)
        self.status_progress_bar.setValue(0)
        self.status_progress_bar.setFixedSize(210, 16)
        self.status_progress_bar.setFormat("%p%")
        self.status_progress_bar.setVisible(False)
        sb.addPermanentWidget(self.status_progress_bar)

    def _register_voice_help(self, widget: QWidget, title: str, description: str) -> None:
        self._voice_help_map[id(widget)] = (title, description)
        widget.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Enter and id(obj) in self._voice_help_map:
            title, desc = self._voice_help_map[id(obj)]
            self.lbl_voice_help_title.setText(f"💡 {title}")
            self.lbl_voice_help_body.setText(desc)
        return super().eventFilter(obj, event)

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

        self.chk_multi_voice_open = QCheckBox("🎭 Tryb 3 głosów")
        self.chk_multi_voice_open.setToolTip(
            "Mini-superprodukcja: przy wczytywaniu nowej książki rozdziela dialogi (—) od didaskaliów narratora "
            "pod 3 głosy (Narrator + M + K) i analizę Bielik AI. Wyłączone = klasyczny 1 lektor."
        )

        top_bar.addWidget(self.btn_open_book)
        top_bar.addWidget(self.chk_multi_voice_open)
        top_bar.addSpacing(12)
        top_bar.addWidget(QLabel("Lub wznów projekt:"))
        self.combo_projects = QComboBox()
        self.combo_projects.setMinimumWidth(260)
        top_bar.addWidget(self.combo_projects)

        self.btn_load_project = QPushButton("📂 Wczytaj projekt")
        self.btn_load_project.clicked.connect(self._on_load_saved_project)
        top_bar.addWidget(self.btn_load_project)

        self.btn_delete_project = QPushButton("🗑 Usuń projekt")
        self.btn_delete_project.setToolTip(
            "Usuwa wybrany projekt z listy (folder work/ z plikami WAV oraz powiązane MP3/M4B z output/)"
        )
        self.btn_delete_project.clicked.connect(self._on_delete_saved_project)
        top_bar.addWidget(self.btn_delete_project)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Metadane i okładka (w pełni edytowalne: tytuł, autor, nazwa projektu, okładka)
        meta_group = QGroupBox("Informacje o książce (używane jako metadane i nazwa pliku M4B / MP3)")
        meta_layout = QHBoxLayout(meta_group)

        cover_col = QVBoxLayout()
        cover_col.setSpacing(4)
        self.lbl_cover = QLabel("Brak okładki")
        self.lbl_cover.setFixedSize(96, 126)
        self.lbl_cover.setAlignment(Qt.AlignCenter)
        self.lbl_cover.setStyleSheet("border: 1px solid #323846; background: #13151a;")
        cover_col.addWidget(self.lbl_cover)

        self.btn_change_cover = QPushButton("🖼 Okładka")
        self.btn_change_cover.setFixedWidth(96)
        self.btn_change_cover.setToolTip("Wybierz lub podmień obraz okładki (JPG/PNG) dołączany do M4B i MP3")
        self.btn_change_cover.setEnabled(False)
        self.btn_change_cover.clicked.connect(self._on_change_cover)
        cover_col.addWidget(self.btn_change_cover)
        cover_col.addStretch()
        meta_layout.addLayout(cover_col)

        info_col = QVBoxLayout()
        info_col.setSpacing(6)

        meta_grid = QGridLayout()
        meta_grid.setHorizontalSpacing(8)
        meta_grid.setVerticalSpacing(5)
        meta_grid.setColumnStretch(1, 1)

        self.ed_book_title = QLineEdit()
        self.ed_book_title.setPlaceholderText("Nie wczytano książki")
        self.ed_book_title.setStyleSheet("font-size: 11pt; font-weight: bold;")
        self.ed_book_title.setToolTip("Tytuł książki — trafia do tagów M4B/MP3 i określa nazwę pliku wynikowego .m4b")
        self.ed_book_title.textChanged.connect(self._update_book_meta_preview)
        self.ed_book_title.returnPressed.connect(self._on_save_book_meta)

        self.ed_book_author = QLineEdit()
        self.ed_book_author.setPlaceholderText("—")
        self.ed_book_author.setToolTip("Autor książki — trafia do tagów Artist / Album Artist w pliku M4B i MP3")
        self.ed_book_author.textChanged.connect(self._update_book_meta_preview)
        self.ed_book_author.returnPressed.connect(self._on_save_book_meta)

        self.ed_project_name = QLineEdit()
        self.ed_project_name.setPlaceholderText("—")
        self.ed_project_name.setToolTip("Nazwa projektu na liście rozwijanej i nazwa folderu roboczego w work/")
        self.ed_project_name.textChanged.connect(self._update_book_meta_preview)
        self.ed_project_name.returnPressed.connect(self._on_save_book_meta)

        self.btn_swap_title_author = QPushButton("⇄ Zamień Tytuł ↔ Autor")
        self.btn_swap_title_author.setToolTip(
            "Zamienia miejscami Tytuł i Autora (przydatne, gdy ebook EPUB ma odwrócone metadane)"
        )
        self.btn_swap_title_author.clicked.connect(self._on_swap_title_author)

        self.btn_save_book_meta = QPushButton("💾 Zapisz dane")
        self.btn_save_book_meta.setToolTip(
            "Zapisuje tytuł, autora i (jeśli zmieniono) bezpiecznie zmienia nazwę folderu projektu w work/"
        )
        self.btn_save_book_meta.clicked.connect(self._on_save_book_meta)

        meta_grid.addWidget(QLabel("Tytuł:"), 0, 0)
        meta_grid.addWidget(self.ed_book_title, 0, 1)
        meta_grid.addWidget(self.btn_swap_title_author, 0, 2)

        meta_grid.addWidget(QLabel("Autor:"), 1, 0)
        meta_grid.addWidget(self.ed_book_author, 1, 1)

        meta_grid.addWidget(QLabel("Projekt:"), 2, 0)
        meta_grid.addWidget(self.ed_project_name, 2, 1)
        meta_grid.addWidget(self.btn_save_book_meta, 2, 2)

        info_col.addLayout(meta_grid)

        self.lbl_book_stats = QLabel("Rozdziały: 0 | Fragmenty: 0 | Szacowany czas: —")
        self.lbl_book_path = QLabel("")
        self.lbl_book_path.setStyleSheet("color: #8b949e; font-size: 9pt;")
        info_col.addWidget(self.lbl_book_stats)
        info_col.addWidget(self.lbl_book_path)
        info_col.addStretch()
        meta_layout.addLayout(info_col, 1)
        layout.addWidget(meta_group)

        self._set_book_meta_controls_enabled(False)

        # Splitter: tabela rozdziałów | podgląd tekstu po normalizacji
        splitter = QSplitter(Qt.Horizontal)

        left_box = QWidget()
        left_lay = QVBoxLayout(left_box)
        left_lay.setContentsMargins(0, 0, 0, 0)

        sel_bar = QHBoxLayout()
        self.btn_toggle_all_chapters = QPushButton("☑ Zaznacz wszystkie")
        self.btn_toggle_all_chapters.setToolTip("Przełącza zaznaczenie wszystkich rozdziałów (zaznacz wszystko / odznacz wszystko)")
        self.btn_toggle_all_chapters.clicked.connect(self._on_toggle_all_chapters)

        self.btn_toggle_highlighted_chapters = QPushButton("☑ Zaznacz podświetlone")
        self.btn_toggle_highlighted_chapters.setToolTip(
            "Przełącza ptaszek na wierszach podświetlonych myszką (z Shift lub Ctrl)"
        )
        self.btn_toggle_highlighted_chapters.setEnabled(False)
        self.btn_toggle_highlighted_chapters.clicked.connect(self._on_toggle_highlighted_chapters)

        self.btn_toggle_tree_collapse = QPushButton("⊟ Zwiń części")
        self.btn_toggle_tree_collapse.setToolTip("Zwija lub rozwija wszystkie części książki w drzewku")
        self.btn_toggle_tree_collapse.setEnabled(False)
        self.btn_toggle_tree_collapse.clicked.connect(self._on_toggle_tab1_collapse)

        sel_bar.addWidget(self.btn_toggle_all_chapters)
        sel_bar.addSpacing(6)
        sel_bar.addWidget(self.btn_toggle_highlighted_chapters)
        sel_bar.addSpacing(6)
        sel_bar.addWidget(self.btn_toggle_tree_collapse)
        sel_bar.addStretch()
        left_lay.addLayout(sel_bar)

        mv_bar = QHBoxLayout()
        self.btn_convert_multi_voice = QPushButton("🎭 Przelicz zaznaczone na 3 głosy")
        self.btn_convert_multi_voice.setToolTip(
            "Dzieli zaznaczone rozdziały na nowo z pliku EPUB/TXT na dialogi (M/K) i narrację (N). "
            "Rozdziały z już wygenerowanym audio są bezpiecznie pomijane."
        )
        self.btn_convert_multi_voice.setEnabled(False)
        self.btn_convert_multi_voice.clicked.connect(self._on_convert_selected_to_multi_voice)
        mv_bar.addWidget(self.btn_convert_multi_voice)
        mv_bar.addSpacing(10)
        self.lbl_highlighted_info = QLabel("")
        self.lbl_highlighted_info.setStyleSheet("color: #8ab4f8; font-size: 9pt;")
        mv_bar.addWidget(self.lbl_highlighted_info, 1)
        left_lay.addLayout(mv_bar)

        self.tree_chapters = QTreeWidget()
        self.tree_chapters.setColumnCount(4)
        self.tree_chapters.setHeaderLabels(["Część / Rozdział", "Fragmenty", "Postęp", "Tryb"])
        hdr_ch = self.tree_chapters.header()
        hdr_ch.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr_ch.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr_ch.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr_ch.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tree_chapters.setAlternatingRowColors(True)
        self.tree_chapters.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tree_chapters.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree_chapters.setIndentation(22)
        self.tree_chapters.setAnimated(True)
        self.tree_chapters.itemChanged.connect(self._on_chapter_checkbox_changed)
        self.tree_chapters.itemSelectionChanged.connect(self._on_chapter_selected_preview)
        self.tree_chapters.itemCollapsed.connect(self._on_tab1_item_collapsed)
        self.tree_chapters.itemExpanded.connect(self._on_tab1_item_expanded)
        left_lay.addWidget(self.tree_chapters)
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
    # ZAKŁADKA 2: LEKTORZY I PRÓBKI GŁOSU (3 PODZAKŁADKI: NARRATOR / M / K)
    # =========================================================================
    def _build_voice_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.voice_tabs = QTabWidget()
        self._voice_panels: dict[str, VoiceRolePanel] = {}
        for role_key, tab_title, role_label in [
            ("narrator", "🎙 Narrator", "Narrator"),
            ("male", "👨 Dialogi męskie", "Dialogi męskie"),
            ("female", "👩 Dialogi żeńskie", "Dialogi żeńskie"),
        ]:
            panel = VoiceRolePanel(
                role_key=role_key,
                role_label=role_label,
                register_help_cb=self._register_voice_help,
                parent=self.voice_tabs,
            )
            panel.profiles_changed.connect(self._on_voice_profiles_changed)
            panel.role_profile_assigned.connect(self._on_role_profile_assigned)
            panel.run_test_requested.connect(self._on_run_single_test_for_role)
            panel.play_wav_requested.connect(lambda p: _play_wav_async(p))
            panel.stop_wav_requested.connect(_stop_wav_playback)
            self._voice_panels[role_key] = panel
            self.voice_tabs.addTab(panel, tab_title)

        layout.addWidget(self.voice_tabs)

        # Wspólny panel kontekstowej pomocy (aktualizuje się po najechaniu myszką w każdej podzakładce)
        help_group = QGroupBox("Pomoc kontekstowa")
        help_group.setMaximumHeight(115)
        help_group.setStyleSheet(
            "QGroupBox { border: 1px solid #2d6cdf; background-color: #141821; border-radius: 6px; margin-top: 8px; padding: 8px; }"
            "QGroupBox::title { color: #8ab4f8; }"
        )
        help_lay = QVBoxLayout(help_group)
        help_lay.setContentsMargins(6, 4, 6, 4)
        self.lbl_voice_help_title = QLabel("💡 Najedź myszką na dowolny parametr powyżej")
        self.lbl_voice_help_title.setStyleSheet(
            "font-weight: bold; font-size: 10pt; color: #8ab4f8; background: transparent;"
        )
        self.lbl_voice_help_body = QLabel(
            "Każda z 3 podzakładek (Narrator, Dialogi męskie, Dialogi żeńskie) przechowuje osobny profil i własne "
            "parametry syntezy (tempo, CFG, ODE, seed). W projektach 1-głosowych używany jest wyłącznie Narrator."
        )
        self.lbl_voice_help_body.setWordWrap(True)
        self.lbl_voice_help_body.setStyleSheet("color: #cfd5df; background: transparent;")
        help_lay.addWidget(self.lbl_voice_help_title)
        help_lay.addWidget(self.lbl_voice_help_body)
        help_lay.addStretch()
        layout.addWidget(help_group, 1)

        # Wspólna blokada edycji na dole zakładki (chroni wszystkie 3 głosy przed przypadkową zmianą)
        lock_bar = QHBoxLayout()
        self.chk_edit_voice = QCheckBox("🔒 Edytuj ustawienia lektorów")
        self.chk_edit_voice.setChecked(False)
        self.chk_edit_voice.toggled.connect(self._on_toggle_voice_edit_lock)
        lock_hint = QLabel("Domyślnie zablokowane, aby nie zmienić przypadkiem wypracowanych parametrów głosów.")
        lock_hint.setStyleSheet("color: #8b949e; font-size: 9pt;")
        lock_bar.addWidget(self.chk_edit_voice)
        lock_bar.addSpacing(12)
        lock_bar.addWidget(lock_hint)
        lock_bar.addStretch()
        layout.addLayout(lock_bar)

        self._set_voice_edit_enabled(False)

        self.tabs.addTab(tab, "2. Lektorzy i Próbki")

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
        self.chk_qg = QCheckBox("Quality Guard (Whisper GPU + retry)")
        self.chk_qg.setChecked(True)
        opts_row.addWidget(self.chk_qg)

        opts_row.addSpacing(10)
        self.chk_turbo_batch = QCheckBox("⚡ Paczki GPU x3 (grupowanie po długości)")
        self.chk_turbo_batch.setChecked(True)
        self.chk_turbo_batch.setToolTip(
            "Sortuje oczekujące zdania według głosu i zbliżonej długości (nawet między rozdziałami) i liczy je "
            "paczkami po 3 naraz na GPU (~2,7× szybciej razem z Whisperem na GPU, przy identycznej jakości ODE=36). "
            "Odznacz, aby generować ściśle po kolei rozdział po rozdziale."
        )
        opts_row.addWidget(self.chk_turbo_batch)

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
        btn_row.addSpacing(10)
        self.lbl_batch_badge = QLabel("⚪ Gotowy")
        self.lbl_batch_badge.setAlignment(Qt.AlignCenter)
        self.lbl_batch_badge.setMinimumWidth(135)
        btn_row.addWidget(self.lbl_batch_badge)
        btn_row.addSpacing(8)
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
        filter_row.addWidget(QLabel("Filtr:"))
        self.combo_chunk_filter = QComboBox()
        self.combo_chunk_filter.addItems([
            "Wszystkie (zaznaczone)",
            "Tylko gotowe",
            "Tylko oczekujące",
            "Podejrzane (QG < 0.85)",
            "Tylko błędy",
            "Wszystkie gotowe (cała książka)",
        ])
        self.combo_chunk_filter.currentIndexChanged.connect(self._refresh_chunks_table)
        filter_row.addWidget(self.combo_chunk_filter)

        filter_row.addSpacing(6)
        filter_row.addWidget(QLabel("Sekcja:"))
        self.combo_chunk_chapter = QComboBox()
        self.combo_chunk_chapter.setMinimumWidth(190)
        self.combo_chunk_chapter.setMaximumWidth(270)
        self.combo_chunk_chapter.setToolTip("Szybko zawęź tabelę do jednej części lub konkretnego rozdziału bez przewijania")
        self.combo_chunk_chapter.addItem("Wszystkie rozdziały", ("all", None))
        self.combo_chunk_chapter.currentIndexChanged.connect(self._refresh_chunks_table)
        filter_row.addWidget(self.combo_chunk_chapter)

        self.btn_toggle_chunks_collapse = QPushButton("⊟ Zwiń części")
        self.btn_toggle_chunks_collapse.setToolTip("Zwija lub rozwija nagłówki części w tabeli (aktywne przy sortowaniu po #)")
        self.btn_toggle_chunks_collapse.setEnabled(False)
        self.btn_toggle_chunks_collapse.clicked.connect(self._on_toggle_tab3_collapse)
        filter_row.addWidget(self.btn_toggle_chunks_collapse)

        btn_refresh_chunks = QPushButton("🔄")
        btn_refresh_chunks.setFixedWidth(34)
        btn_refresh_chunks.setToolTip("Odśwież listę fragmentów")
        btn_refresh_chunks.clicked.connect(self._refresh_chunks_table)
        filter_row.addWidget(btn_refresh_chunks)
        filter_row.addStretch()

        self.btn_bielik_tag = QPushButton("🎭 Przypisz głosy (Bielik AI)")
        self.btn_bielik_tag.setToolTip(
            "Uruchamia lokalny polski model Bielik-4.5B (GPU) do automatycznego rozpoznania płci mówców w dialogach "
            "(M / K / N) dla zaznaczonych rozdziałów. Przed startem zwalnia VRAM z F5-TTS."
        )
        self.btn_bielik_tag.clicked.connect(self._on_run_bielik_tagger)

        self.btn_play_chunk = QPushButton("🔊 Odsłuchaj")
        self.btn_play_chunk.setToolTip("Odsłuchaj zaznaczony fragment (możesz też kliknąć ikonę ▶ po lewej)")
        self.btn_play_chunk.clicked.connect(self._on_play_selected_chunk)
        btn_stop_chunk = QPushButton("⏹ Cisza")
        btn_stop_chunk.clicked.connect(_stop_wav_playback)
        self.btn_regen_chunk = QPushButton("🔄 Regeneruj")
        self.btn_regen_chunk.setToolTip("Generuje ponownie zaznaczony fragment z nowym seedem (dwuklik na tekście pozwala go wcześniej edytować)")
        self.btn_regen_chunk.clicked.connect(self._on_regenerate_selected_chunk)

        filter_row.addWidget(self.btn_bielik_tag)
        filter_row.addSpacing(6)
        filter_row.addWidget(self.btn_play_chunk)
        filter_row.addWidget(btn_stop_chunk)
        filter_row.addWidget(self.btn_regen_chunk)
        layout.addLayout(filter_row)

        self.tbl_chunks = QTableWidget(0, _COL_COUNT)
        self.tbl_chunks.setHorizontalHeaderLabels(
            ["▶", "Głos", "Rozdział", "#", "Status", "Czas", "QG", "Tekst (do przeczytania — 2x klik edytuje)", "Usłyszano (Whisper)"]
        )
        # Wszystkie kolumny Interactive — można dowolnie zwężać Tekst i rozszerzać Whisper, a klik w nagłówek sortuje!
        hdr = self.tbl_chunks.horizontalHeader()
        hdr.setSectionResizeMode(_COL_PLAY, QHeaderView.Fixed)
        for col_i in range(1, _COL_COUNT):
            hdr.setSectionResizeMode(col_i, QHeaderView.Interactive)
        hdr.setStretchLastSection(True)
        hdr.setSortIndicatorShown(True)
        hdr.setSortIndicator(self._chunks_sort_col, self._chunks_sort_order)
        hdr.sectionClicked.connect(self._on_chunks_header_clicked)
        self.tbl_chunks.setColumnWidth(_COL_PLAY, 34)
        self.tbl_chunks.setColumnWidth(_COL_VOICE, 56)
        self.tbl_chunks.setColumnWidth(_COL_CHAPTER, 88)
        self.tbl_chunks.setColumnWidth(_COL_NUM, 45)
        self.tbl_chunks.setColumnWidth(_COL_STATUS, 64)
        self.tbl_chunks.setColumnWidth(_COL_DUR, 50)
        self.tbl_chunks.setColumnWidth(_COL_QG, 50)
        self.tbl_chunks.setColumnWidth(_COL_TEXT, 400)
        self.tbl_chunks.setColumnWidth(_COL_HEARD, 340)
        self.tbl_chunks.verticalHeader().setVisible(False)
        self.tbl_chunks.setAlternatingRowColors(True)
        self.tbl_chunks.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_chunks.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.tbl_chunks.itemChanged.connect(self._on_chunk_item_changed)
        self.tbl_chunks.cellClicked.connect(self._on_chunk_cell_clicked)
        self.tbl_chunks.cellDoubleClicked.connect(self._on_chunk_cell_double_clicked)
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
        self.txt_log.setMaximumBlockCount(1000)
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
        btn_apply_pending = QPushButton("🔄 Zastosuj do oczekujących w projekcie")
        btn_apply_pending.setToolTip(
            "Aktualizuje zapisany słownik i przelicza tekst wszystkich jeszcze niewygenerowanych (pending) zdań w otwartym projekcie"
        )
        btn_apply_pending.clicked.connect(self._on_pron_apply_to_pending)
        btn_save = QPushButton("💾 Zapisz słownik")
        btn_save.setObjectName("primaryBtn")
        btn_save.clicked.connect(self._on_pron_save)
        btn_bar.addWidget(btn_add)
        btn_bar.addWidget(btn_del)
        btn_bar.addStretch()
        btn_bar.addWidget(btn_apply_pending)
        btn_bar.addWidget(btn_save)
        layout.addLayout(btn_bar)

        self.tabs.addTab(tab, "4. Słownik Wymowy")

    # =========================================================================
    # ZAKŁADKA 5: ODTWARZACZ AUDIOBOOKÓW (Z KARAOKE TEKSTU)
    # =========================================================================
    def _build_player_tab(self) -> None:
        self.player_widget = AudiobookPlayerWidget(self)
        self.player_widget.edit_chunk_requested.connect(self._on_player_edit_chunk)
        self.player_widget.status_message.connect(lambda msg: self.lbl_global_status.setText(msg))
        self.tabs.addTab(self.player_widget, "5. Odtwarzacz")

    def _is_text_input_focused(self) -> bool:
        w = QApplication.focusWidget()
        return isinstance(w, (QLineEdit, QPlainTextEdit, QTextEdit, QAbstractSpinBox))

    def _run_player_shortcut(self, fn, require_player_tab: bool = False, ignore_text_focus: bool = False) -> None:
        if require_player_tab and self.tabs.currentWidget() is not self.player_widget:
            return
        if not ignore_text_focus and self._is_text_input_focused():
            return
        fn()

    def _setup_player_shortcuts(self) -> None:
        self._player_shortcuts: list[QShortcut] = []

        def _add(seq, fn, require_player_tab: bool = False, ignore_text_focus: bool = False) -> None:
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(
                lambda: self._run_player_shortcut(
                    fn,
                    require_player_tab=require_player_tab,
                    ignore_text_focus=ignore_text_focus,
                )
            )
            self._player_shortcuts.append(sc)

        # Globalne skróty działające w całym aktywnym oknie aplikacji (niezależnie od zakładki)
        _add("Ctrl+Space", self.player_widget.toggle_play_pause, ignore_text_focus=True)
        _add("Ctrl+Left", lambda: self.player_widget.seek_relative_ms(-10_000))
        _add("Ctrl+Right", lambda: self.player_widget.seek_relative_ms(10_000))
        _add("Ctrl+Shift+Left", lambda: self.player_widget.seek_relative_ms(-60_000))
        _add("Ctrl+Shift+Right", lambda: self.player_widget.seek_relative_ms(60_000))

        # Klawisze multimedialne klawiatury (zawsze aktywne)
        _add(Qt.Key_MediaTogglePlayPause, self.player_widget.toggle_play_pause, ignore_text_focus=True)
        _add(Qt.Key_MediaPlay, self.player_widget.toggle_play_pause, ignore_text_focus=True)
        _add(Qt.Key_MediaPause, self.player_widget.pause_playback, ignore_text_focus=True)
        _add(Qt.Key_MediaPrevious, lambda: self.player_widget.seek_relative_ms(-10_000), ignore_text_focus=True)
        _add(Qt.Key_MediaNext, lambda: self.player_widget.seek_relative_ms(10_000), ignore_text_focus=True)

        # Szybkie pojedyncze klawisze aktywne na Zakładce 5 (Odtwarzacz)
        _add(Qt.Key_Space, self.player_widget.toggle_play_pause, require_player_tab=True)
        _add(Qt.Key_Left, lambda: self.player_widget.seek_relative_ms(-10_000), require_player_tab=True)
        _add(Qt.Key_Right, lambda: self.player_widget.seek_relative_ms(10_000), require_player_tab=True)
        _add("Shift+Left", lambda: self.player_widget.seek_relative_ms(-60_000), require_player_tab=True)
        _add("Shift+Right", lambda: self.player_widget.seek_relative_ms(60_000), require_player_tab=True)
        _add("[", lambda: self.player_widget.adjust_playback_rate(-0.05), require_player_tab=True)
        _add("]", lambda: self.player_widget.adjust_playback_rate(+0.05), require_player_tab=True)

    def _on_tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.player_widget and self.db is not None:
            self.player_widget.refresh_new_chunks()

    def _on_player_edit_chunk(self, chunk_id: int) -> None:
        """Przechodzi z Odtwarzacza dokładnie do wskazanego fragmentu w Zakładce 3 (automatycznie rozwija część)."""
        self.tabs.setCurrentIndex(2)
        if self.db is not None:
            try:
                target_chunk = self.db.get_chunk(chunk_id)
                part_id = self._chapter_to_part_id.get(target_chunk.chapter_id, target_chunk.chapter_id)
                if part_id in self._tab3_collapsed_groups or target_chunk.chapter_id in self._tab3_collapsed_groups:
                    self._tab3_collapsed_groups.discard(part_id)
                    self._tab3_collapsed_groups.discard(target_chunk.chapter_id)
                    self._refresh_chunks_table()
            except KeyError:
                pass
        if chunk_id not in self._chunk_id_to_row:
            self.combo_chunk_chapter.blockSignals(True)
            self.combo_chunk_chapter.setCurrentIndex(0)
            self.combo_chunk_chapter.blockSignals(False)
            self.combo_chunk_filter.blockSignals(True)
            self.combo_chunk_filter.setCurrentIndex(0)
            self.combo_chunk_filter.blockSignals(False)
            self._refresh_chunks_table()
        if chunk_id not in self._chunk_id_to_row:
            self.combo_chunk_filter.setCurrentIndex(5)
            self._refresh_chunks_table()
        row = self._chunk_id_to_row.get(chunk_id)
        if row is not None and 0 <= row < self.tbl_chunks.rowCount():
            self.tbl_chunks.selectRow(row)
            item = self.tbl_chunks.item(row, _COL_TEXT)
            if item is not None:
                self.tbl_chunks.scrollToItem(item, QAbstractItemView.PositionAtCenter)
            self.tbl_chunks.setFocus()
            self.lbl_global_status.setText(
                "Zaznaczono fragment z odtwarzacza — widać tekst i Whisper; kliknij 2x tekst, aby edytować, lub Regeneruj."
            )

    # =========================================================================
    # LOGIKA: KSIĄŻKI I PROJEKTY
    # =========================================================================
    def _log(self, msg: str) -> None:
        self.txt_log.appendPlainText(msg)

    def _refresh_saved_projects(self, select_dir: Path | None = None) -> None:
        self.combo_projects.clear()
        target_res = select_dir.resolve() if select_dir is not None else None
        selected_idx = 0
        for idx, proj_dir in enumerate(list_saved_projects(WORK_DIR)):
            self.combo_projects.addItem(proj_dir.name, str(proj_dir))
            if target_res is not None and proj_dir.resolve() == target_res:
                selected_idx = idx
        if self.combo_projects.count() > 0:
            self.combo_projects.setCurrentIndex(selected_idx)

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
            use_multi = self.chk_multi_voice_open.isChecked()
            self.db = ProjectDB.create_from_book(
                book,
                pronunciation_dict=pron,
                work_root=WORK_DIR,
                base_seed=base_seed,
                multi_voice=use_multi,
            )
            self._refresh_saved_projects(select_dir=self.db.project_dir)
            self._load_project_into_ui()
            mode_txt = "tryb 3 głosów (mini-superprodukcja)" if use_multi else "tryb 1 lektora"
            self._log(f"Wczytano książkę: {book.title} ({len(book.chapters)} rozdziałów, {mode_txt}).")
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

    @staticmethod
    def _dir_size_mb(path: Path) -> float:
        if not path.exists():
            return 0.0
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024 * 1024)

    def _on_delete_saved_project(self) -> None:
        proj_str = self.combo_projects.currentData()
        if not proj_str:
            QMessageBox.information(self, "Brak projektu", "Wybierz projekt z listy do usunięcia.")
            return

        proj_dir = Path(proj_str)
        if not proj_dir.exists():
            self._refresh_saved_projects()
            return

        if (
            self.batch_worker
            and self.batch_worker.isRunning()
            and self.db is not None
            and self.db.project_dir.resolve() == proj_dir.resolve()
        ):
            QMessageBox.warning(
                self,
                "Projekt w trakcie generowania",
                "Najpierw zatrzymaj generowanie (przycisk Stop), zanim usuniesz ten projekt.",
            )
            return

        # Znajdź powiązany folder w output/ (po tytule książki z SQLite oraz po nazwie folderu projektu)
        out_candidates: list[Path] = []
        try:
            tmp_db = ProjectDB(proj_dir)
            title = tmp_db.get_meta("title", "")
            from f5_audiobook.audio.exporter import _safe_filename

            for name in {proj_dir.name, _safe_filename(title) if title else ""}:
                if name:
                    cand = OUTPUT_DIR / name
                    if cand.exists() and cand.is_dir() and cand not in out_candidates:
                        out_candidates.append(cand)
        except Exception:
            cand = OUTPUT_DIR / proj_dir.name
            if cand.exists() and cand.is_dir():
                out_candidates.append(cand)

        work_mb = self._dir_size_mb(proj_dir)
        out_mb = sum(self._dir_size_mb(p) for p in out_candidates)

        details = [f"• Roboczy (SQLite + pliki WAV): {proj_dir} ({work_mb:.1f} MB)"]
        for od in out_candidates:
            details.append(f"• Wyeksportowane audio (MP3/M4B): {od} ({self._dir_size_mb(od):.1f} MB)")

        msg = (
            f"Czy na pewno chcesz trwale usunąć projekt „{proj_dir.name}” "
            f"(łącznie {work_mb + out_mb:.1f} MB)?\n\n"
            + "\n".join(details)
        )
        reply = QMessageBox.warning(
            self,
            "Usuń projekt",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Jeśli usuwamy aktualnie otwarty projekt, najpierw czyścimy UI i zatrzymujemy odsłuch
        if self.db is not None and self.db.project_dir.resolve() == proj_dir.resolve():
            _stop_wav_playback()
            self.player_widget.set_project(None)
            self.db = None
            self._saved_book_meta = ("", "", "")
            for ed in (self.ed_book_title, self.ed_book_author, self.ed_project_name):
                ed.blockSignals(True)
                ed.clear()
                ed.blockSignals(False)
            self._set_book_meta_controls_enabled(False)
            self.lbl_book_stats.setText("Rozdziały: 0 | Fragmenty: 0 | Szacowany czas: —")
            self.lbl_book_path.setText("")
            self.lbl_cover.setPixmap(QPixmap())
            self.lbl_cover.setText("Brak okładki")
            self.tree_chapters.clear()
            self._chapters_cache = []
            self._chapter_to_part_id = {}
            self._part_titles = {}
            self._tab1_collapsed_part_ids.clear()
            self._tab3_collapsed_groups.clear()
            self.txt_chapter_preview.clear()
            self._refresh_chunk_chapter_combo()
            self._refresh_chunks_table()
            self._update_chapter_toggle_buttons()
            self._update_progress_widgets()

        errors: list[str] = []
        for target in [proj_dir, *out_candidates]:
            try:
                if target.exists():
                    shutil.rmtree(target)
            except Exception as e:
                errors.append(f"{target}: {e}")

        self._refresh_saved_projects()
        if errors:
            QMessageBox.warning(
                self,
                "Częściowo usunięto",
                "Nie udało się usunąć niektórych plików (mogą być otwarte w odtwarzaczu):\n" + "\n".join(errors),
            )
        else:
            self._log(f"Usunięto projekt: {proj_dir.name} (zwolniono {work_mb + out_mb:.1f} MB).")
            QMessageBox.information(
                self,
                "Usunięto projekt",
                f"Projekt „{proj_dir.name}” oraz jego pliki WAV/MP3/M4B zostały usunięte ({work_mb + out_mb:.1f} MB).",
            )

    def _set_book_meta_controls_enabled(self, enabled: bool) -> None:
        self.ed_book_title.setEnabled(enabled)
        self.ed_book_author.setEnabled(enabled)
        self.ed_project_name.setEnabled(enabled)
        self.btn_swap_title_author.setEnabled(enabled)
        self.btn_save_book_meta.setEnabled(enabled)
        self.btn_change_cover.setEnabled(enabled)

    def _has_unsaved_book_meta(self) -> bool:
        if self.db is None:
            return False
        cur_title = self.ed_book_title.text().strip()
        cur_author = self.ed_book_author.text().strip()
        cur_proj = slugify_project_name(self.ed_project_name.text()) if self.ed_project_name.text().strip() else ""
        saved_title, saved_author, saved_proj = self._saved_book_meta
        return (cur_title, cur_author, cur_proj) != (saved_title, saved_author, saved_proj)

    def _update_book_meta_preview(self) -> None:
        if self.db is None:
            self.lbl_book_path.setText("")
            return
        cur_title = self.ed_book_title.text().strip() or self.db.project_dir.name
        safe_title = _safe_filename(cur_title)
        proj_slug = slugify_project_name(self.ed_project_name.text()) if self.ed_project_name.text().strip() else self.db.project_dir.name
        dirty = self._has_unsaved_book_meta()

        m4b_rel = f"output/{safe_title}/{safe_title}.m4b"
        if dirty:
            self.lbl_book_path.setStyleSheet("color: #f0b429; font-size: 9pt;")
            self.lbl_book_path.setText(
                f"Plik M4B: {m4b_rel}  •  Folder projektu: work/{proj_slug}  (niezapisane zmiany — kliknij Zapisz dane)"
            )
            self.btn_save_book_meta.setObjectName("primaryBtn")
            self.btn_save_book_meta.setText("💾 Zapisz dane *")
        else:
            self.lbl_book_path.setStyleSheet("color: #8b949e; font-size: 9pt;")
            self.lbl_book_path.setText(f"Plik M4B: {m4b_rel}  •  Folder projektu: work/{proj_slug}")
            self.btn_save_book_meta.setObjectName("")
            self.btn_save_book_meta.setText("💾 Zapisz dane")
        self.btn_save_book_meta.style().unpolish(self.btn_save_book_meta)
        self.btn_save_book_meta.style().polish(self.btn_save_book_meta)

    def _on_swap_title_author(self) -> None:
        if self.db is None:
            return
        old_title = self.ed_book_title.text().strip()
        old_author = self.ed_book_author.text().strip()
        old_proj = self.ed_project_name.text().strip()

        self.ed_book_title.setText(old_author)
        self.ed_book_author.setText(old_title)

        # Jeśli nazwa folderu projektu była automatycznie pochodna od starego tytułu, też ją dopasuj
        if not old_proj or old_proj == slugify_project_name(old_title):
            if old_author:
                self.ed_project_name.setText(slugify_project_name(old_author))

    def _on_save_book_meta(self) -> None:
        if self.db is None:
            return
        new_title = self.ed_book_title.text().strip()
        new_author = self.ed_book_author.text().strip() or "Nieznany autor"
        raw_proj = self.ed_project_name.text().strip()
        if not new_title:
            QMessageBox.warning(self, "Brak tytułu", "Tytuł książki nie może być pusty.")
            self.ed_book_title.setFocus()
            return
        new_proj_slug = slugify_project_name(raw_proj) if raw_proj else self.db.project_dir.name
        old_proj_name = self.db.project_dir.name
        wants_rename = new_proj_slug != old_proj_name

        # Zawsze można zapisać tytuł i autora do SQLite (nawet w trakcie generowania dzięki WAL)
        try:
            self.db.update_book_metadata(new_title, new_author)
        except Exception as e:
            QMessageBox.critical(self, "Błąd zapisu metadanych", str(e))
            return

        renamed = False
        if wants_rename:
            is_busy = any(
                w is not None and w.isRunning()
                for w in (self.batch_worker, self.regen_worker, self.export_worker)
            )
            if is_busy:
                self.ed_project_name.blockSignals(True)
                self.ed_project_name.setText(old_proj_name)
                self.ed_project_name.blockSignals(False)
                self._saved_book_meta = (new_title, new_author, old_proj_name)
                self._update_book_meta_preview()
                QMessageBox.warning(
                    self,
                    "Zapisano tytuł i autora",
                    "Tytuł i autor zostały zapisane, ale zmiana nazwy folderu projektu w work/ "
                    "wymaga wcześniejszego zatrzymania generowania lub eksportu.",
                )
                return

            try:
                _stop_wav_playback()
                self.player_widget.persist_state()
                self.player_widget.release_media()
                self.db.rename_project(new_proj_slug)
                renamed = True
            except Exception as e:
                self.ed_project_name.blockSignals(True)
                self.ed_project_name.setText(old_proj_name)
                self.ed_project_name.blockSignals(False)
                self._saved_book_meta = (new_title, new_author, old_proj_name)
                self._update_book_meta_preview()
                QMessageBox.critical(self, "Nie udało się zmienić nazwy projektu", str(e))
                return

        self.ed_book_title.blockSignals(True)
        self.ed_book_author.blockSignals(True)
        self.ed_project_name.blockSignals(True)
        self.ed_book_title.setText(new_title)
        self.ed_book_author.setText(new_author)
        self.ed_project_name.setText(self.db.project_dir.name)
        self.ed_book_title.blockSignals(False)
        self.ed_book_author.blockSignals(False)
        self.ed_project_name.blockSignals(False)

        self._saved_book_meta = (new_title, new_author, self.db.project_dir.name)
        self._update_book_meta_preview()

        if renamed:
            self._refresh_saved_projects(select_dir=self.db.project_dir)
            self._refresh_chunks_table()
            self.player_widget.set_project(self.db)
            self._log(
                f"Zapisano dane książki i zmieniono projekt: {old_proj_name} → {self.db.project_dir.name} "
                f"(„{new_title}” — {new_author})."
            )
            self.lbl_global_status.setText(
                f"Zapisano: „{new_title}” ({new_author}) | projekt: {self.db.project_dir.name}"
            )
        else:
            self.player_widget.refresh_metadata()
            self._log(f"Zapisano metadane książki: „{new_title}” — {new_author}.")
            self.lbl_global_status.setText(f"Zapisano metadane książki: „{new_title}” ({new_author})")

    def _on_change_cover(self) -> None:
        if self.db is None:
            return
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz obraz okładki",
            "",
            "Obrazy (*.jpg *.jpeg *.png *.webp);;Wszystkie pliki (*.*)",
        )
        if not fpath:
            return
        try:
            dest = self.db.set_cover_image(Path(fpath))
            pix = QPixmap(str(dest)).scaled(96, 126, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_cover.setPixmap(pix)
            self.player_widget.refresh_metadata()
            self._log(f"Zaktualizowano okładkę projektu: {dest.name}")
            self.lbl_global_status.setText(f"Zaktualizowano okładkę: {dest.name}")
        except Exception as e:
            QMessageBox.critical(self, "Błąd okładki", str(e))

    def _load_project_into_ui(self) -> None:
        if self.db is None:
            return
        meta = self.db.get_all_meta()
        title = meta.get("title", self.db.project_dir.name)
        author = meta.get("author", "Nieznany autor")
        proj_name = self.db.project_dir.name

        for ed, val in (
            (self.ed_book_title, title),
            (self.ed_book_author, author),
            (self.ed_project_name, proj_name),
        ):
            ed.blockSignals(True)
            ed.setText(val)
            ed.blockSignals(False)

        self._saved_book_meta = (title.strip(), author.strip(), proj_name)
        self._set_book_meta_controls_enabled(True)
        self._update_book_meta_preview()

        cover_path = Path(meta["cover_path"]) if meta.get("cover_path") else None
        if cover_path and cover_path.exists():
            pix = QPixmap(str(cover_path)).scaled(96, 126, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_cover.setPixmap(pix)
        else:
            self.lbl_cover.setPixmap(QPixmap())
            self.lbl_cover.setText("Brak okładki")

        self._tab1_collapsed_part_ids.clear()
        self._tab3_collapsed_groups.clear()
        self._refresh_chapters_table(init_tab3_collapse=True)
        self._refresh_chunks_table()
        if not (self.batch_worker and self.batch_worker.isRunning()):
            self._batch_state = "idle"
        self._update_progress_widgets()
        self.player_widget.set_project(self.db)

    def _collect_chapters_from_tree_items(self, items: list[QTreeWidgetItem]) -> list[ChapterRow]:
        """Zwraca unikalne obiekty ChapterRow dla listy elementów drzewka (część obejmuje siebie + wszystkie podsekcje)."""
        by_id = {ch.id: ch for ch in self._chapters_cache}
        seen: set[int] = set()
        result: list[ChapterRow] = []

        def _add_item(it: QTreeWidgetItem) -> None:
            cid = it.data(0, Qt.UserRole)
            if cid is not None and int(cid) not in seen and int(cid) in by_id:
                seen.add(int(cid))
                result.append(by_id[int(cid)])
            for i in range(it.childCount()):
                _add_item(it.child(i))

        for it in items:
            _add_item(it)
        return result

    def _refresh_chapters_table(self, init_tab3_collapse: bool = False) -> None:
        if self.db is None:
            return
        chapters = self.db.list_chapters()
        self._chapters_cache = chapters
        groups = _group_chapters_into_parts(chapters)

        # Zbuduj mapowanie chapter_id -> part_id (współdzielone z Zakładką 3)
        self._chapter_to_part_id = {}
        self._part_titles = {}
        has_parts = any(len(children) > 0 for _, children in groups)
        total_all_chunks = sum(ch.total_chunks for ch in chapters)
        for parent_ch, children in groups:
            self._chapter_to_part_id[parent_ch.id] = parent_ch.id
            self._part_titles[parent_ch.id] = parent_ch.title
            for child_ch in children:
                self._chapter_to_part_id[child_ch.id] = parent_ch.id
            if init_tab3_collapse and has_parts and total_all_chunks > 1500 and children:
                part_done = parent_ch.done_chunks + sum(c.done_chunks for c in children)
                if part_done == 0:
                    self._tab3_collapsed_groups.add(parent_ch.id)

        self.tree_chapters.blockSignals(True)
        self.tree_chapters.clear()

        bold_font = QFont()
        bold_font.setBold(True)
        part_bg = QBrush(QColor("#1d2534"))
        part_fg = QBrush(QColor("#9dc1fb"))
        done_fg = QBrush(QColor("#57d9a3"))
        mv_fg = QBrush(QColor("#6cb6ff"))
        mv_partial_fg = QBrush(QColor("#f0b429"))
        single_fg = QBrush(QColor("#8b949e"))

        def _chapter_mode_cell(ch: ChapterRow) -> tuple[str, QBrush]:
            if not ch.multi_voice:
                return "🎙 1 głos", single_fg
            if ch.dialogue_untagged == 0 and ch.dialogue_tagged > 0:
                return "🎭 3 głosy ✅", done_fg
            return "🎭 3 głosy", mv_fg

        sel_ch = 0
        sel_chunks = 0
        for ch in chapters:
            if ch.selected:
                sel_ch += 1
                sel_chunks += ch.total_chunks

        for parent_ch, children in groups:
            is_part = len(children) > 0
            all_in_group = [parent_ch, *children]
            grp_total = sum(c.total_chunks for c in all_in_group)
            grp_done = sum(c.done_chunks for c in all_in_group)

            top_item = QTreeWidgetItem()
            top_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            top_item.setData(0, Qt.UserRole, parent_ch.id)
            top_item.setData(0, Qt.UserRole + 1, is_part)

            if is_part:
                all_sel = all(c.selected for c in all_in_group)
                any_sel = any(c.selected for c in all_in_group)
                top_item.setCheckState(
                    0, Qt.Checked if all_sel else (Qt.PartiallyChecked if any_sel else Qt.Unchecked)
                )
                top_item.setText(0, f"{parent_ch.title}  ({len(children)} rozdz.)")
                top_item.setText(1, str(grp_total))
                top_item.setText(2, f"{grp_done}/{grp_total}")
                mv_cnt = sum(1 for c in children if c.multi_voice)
                grp_untagged = sum(c.dialogue_untagged for c in all_in_group)
                grp_tagged = sum(c.dialogue_tagged for c in all_in_group)
                if mv_cnt == len(children) and parent_ch.multi_voice:
                    if grp_untagged == 0 and grp_tagged > 0:
                        p_mode_txt, p_mode_fg = "🎭 3 głosy ✅", done_fg
                    else:
                        p_mode_txt, p_mode_fg = "🎭 3 głosy", mv_fg
                elif mv_cnt > 0 or parent_ch.multi_voice:
                    p_mode_txt, p_mode_fg = f"🎭 część ({mv_cnt}/{len(children)})", mv_partial_fg
                else:
                    p_mode_txt, p_mode_fg = "🎙 1 głos", single_fg
                top_item.setText(3, p_mode_txt)
                for col_i in range(4):
                    top_item.setFont(col_i, bold_font)
                    top_item.setBackground(col_i, part_bg)
                top_item.setForeground(0, part_fg)
                top_item.setForeground(3, p_mode_fg)
                if grp_total > 0 and grp_done >= grp_total:
                    top_item.setForeground(2, done_fg)
            else:
                top_item.setCheckState(0, Qt.Checked if parent_ch.selected else Qt.Unchecked)
                top_item.setText(0, parent_ch.title)
                top_item.setText(1, str(parent_ch.total_chunks))
                top_item.setText(2, f"{parent_ch.done_chunks}/{parent_ch.total_chunks}")
                m_txt, m_fg = _chapter_mode_cell(parent_ch)
                top_item.setText(3, m_txt)
                top_item.setForeground(3, m_fg)
                if parent_ch.total_chunks > 0 and parent_ch.done_chunks >= parent_ch.total_chunks:
                    top_item.setForeground(2, done_fg)

            top_item.setTextAlignment(1, Qt.AlignCenter)
            top_item.setTextAlignment(2, Qt.AlignCenter)
            top_item.setTextAlignment(3, Qt.AlignCenter)
            self.tree_chapters.addTopLevelItem(top_item)

            for child_ch in children:
                child_item = QTreeWidgetItem(top_item)
                child_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                child_item.setData(0, Qt.UserRole, child_ch.id)
                child_item.setData(0, Qt.UserRole + 1, False)
                child_item.setCheckState(0, Qt.Checked if child_ch.selected else Qt.Unchecked)
                child_item.setText(0, child_ch.title)
                child_item.setText(1, str(child_ch.total_chunks))
                child_item.setText(2, f"{child_ch.done_chunks}/{child_ch.total_chunks}")
                cm_txt, cm_fg = _chapter_mode_cell(child_ch)
                child_item.setText(3, cm_txt)
                child_item.setForeground(3, cm_fg)
                child_item.setTextAlignment(1, Qt.AlignCenter)
                child_item.setTextAlignment(2, Qt.AlignCenter)
                child_item.setTextAlignment(3, Qt.AlignCenter)
                if child_ch.total_chunks > 0 and child_ch.done_chunks >= child_ch.total_chunks:
                    child_item.setForeground(2, done_fg)

            if is_part:
                top_item.setExpanded(parent_ch.id not in self._tab1_collapsed_part_ids)

        self.tree_chapters.blockSignals(False)
        est_min = round(sel_chunks * 6.0 / 60.0)
        mode_lbl = "🎭 3 głosy" if self.db.is_multi_voice() else "🎙 1 głos"
        self.lbl_book_stats.setText(
            f"Zaznaczone rozdziały: {sel_ch}/{len(chapters)} | "
            f"Fragmenty: {sel_chunks} | Szacowana długość: ~{est_min} min | Tryb: {mode_lbl}"
        )
        self._refresh_chunk_chapter_combo()
        self._update_chapter_toggle_buttons()

    def _on_tab1_item_collapsed(self, item: QTreeWidgetItem) -> None:
        cid = item.data(0, Qt.UserRole)
        if cid is not None and item.childCount() > 0:
            self._tab1_collapsed_part_ids.add(int(cid))
            self._update_chapter_toggle_buttons()

    def _on_tab1_item_expanded(self, item: QTreeWidgetItem) -> None:
        cid = item.data(0, Qt.UserRole)
        if cid is not None and item.childCount() > 0:
            self._tab1_collapsed_part_ids.discard(int(cid))
            self._update_chapter_toggle_buttons()

    def _on_toggle_tab1_collapse(self) -> None:
        part_items = [
            self.tree_chapters.topLevelItem(i)
            for i in range(self.tree_chapters.topLevelItemCount())
            if self.tree_chapters.topLevelItem(i).childCount() > 0
        ]
        if not part_items:
            return
        any_expanded = any(it.isExpanded() for it in part_items)
        self.tree_chapters.blockSignals(True)
        for it in part_items:
            cid = int(it.data(0, Qt.UserRole))
            if any_expanded:
                it.setExpanded(False)
                self._tab1_collapsed_part_ids.add(cid)
            else:
                it.setExpanded(True)
                self._tab1_collapsed_part_ids.discard(cid)
        self.tree_chapters.blockSignals(False)
        self._update_chapter_toggle_buttons()

    def _update_chapter_toggle_buttons(self) -> None:
        if self.db is None or not self._chapters_cache:
            self.btn_toggle_all_chapters.setText("☑ Zaznacz wszystkie")
            self.btn_toggle_all_chapters.setEnabled(False)
            self.btn_toggle_highlighted_chapters.setText("☑ Zaznacz podświetlone")
            self.btn_toggle_highlighted_chapters.setEnabled(False)
            self.btn_toggle_tree_collapse.setText("⊟ Zwiń części")
            self.btn_toggle_tree_collapse.setEnabled(False)
            self.btn_convert_multi_voice.setEnabled(False)
            self.lbl_highlighted_info.setText("")
            return

        self.btn_toggle_all_chapters.setEnabled(True)
        self.btn_convert_multi_voice.setEnabled(any(ch.selected for ch in self._chapters_cache))
        all_checked = all(ch.selected for ch in self._chapters_cache)
        self.btn_toggle_all_chapters.setText(
            "☐ Odznacz wszystkie" if all_checked else "☑ Zaznacz wszystkie"
        )

        part_items = [
            self.tree_chapters.topLevelItem(i)
            for i in range(self.tree_chapters.topLevelItemCount())
            if self.tree_chapters.topLevelItem(i).childCount() > 0
        ]
        if part_items:
            self.btn_toggle_tree_collapse.setEnabled(True)
            any_expanded = any(it.isExpanded() for it in part_items)
            self.btn_toggle_tree_collapse.setText("⊟ Zwiń części" if any_expanded else "⊞ Rozwiń części")
        else:
            self.btn_toggle_tree_collapse.setEnabled(False)
            self.btn_toggle_tree_collapse.setText("⊟ Zwiń części")

        sel_items = self.tree_chapters.selectedItems()
        sel_chapters = self._collect_chapters_from_tree_items(sel_items)
        if not sel_chapters:
            self.btn_toggle_highlighted_chapters.setText("☑ Zaznacz podświetlone")
            self.btn_toggle_highlighted_chapters.setEnabled(False)
            self.lbl_highlighted_info.setText("")
            return

        self.btn_toggle_highlighted_chapters.setEnabled(True)
        all_hl_checked = all(ch.selected for ch in sel_chapters)
        self.btn_toggle_highlighted_chapters.setText(
            "☐ Odznacz podświetlone" if all_hl_checked else "☑ Zaznacz podświetlone"
        )

        hl_total = sum(ch.total_chunks for ch in sel_chapters)
        hl_done = sum(ch.done_chunks for ch in sel_chapters)
        hl_pending = max(0, hl_total - hl_done)
        est_gen = _format_eta(hl_pending * 10.0) if hl_pending > 0 else "gotowe!"
        self.lbl_highlighted_info.setText(
            f"Podświetlono: {len(sel_chapters)} sekcji | {hl_total} fragm. (do przeliczenia: {hl_pending} ≈ {est_gen})"
        )

    def _on_toggle_all_chapters(self) -> None:
        if self.db is None:
            return
        chapters = self.db.list_chapters()
        target_state = not all(ch.selected for ch in chapters)
        self.db.set_chapters_selected_bulk([ch.id for ch in chapters], target_state)
        self._sync_tree_checkstates_inplace()
        self._refresh_chunks_table()
        self._update_progress_widgets()

    def _on_toggle_highlighted_chapters(self) -> None:
        if self.db is None:
            return
        sel_chapters = self._collect_chapters_from_tree_items(self.tree_chapters.selectedItems())
        if not sel_chapters:
            return
        target_state = not all(ch.selected for ch in sel_chapters)
        self.db.set_chapters_selected_bulk([ch.id for ch in sel_chapters], target_state)
        self._sync_tree_checkstates_inplace()
        self._refresh_chunks_table()
        self._update_progress_widgets()

    def _sync_tree_checkstates_inplace(self) -> None:
        """Odświeża stan checkboxów i statystyk w drzewku bez usuwania węzłów (bezpieczne wewnątrz itemChanged)."""
        if self.db is None:
            return
        chapters = self.db.list_chapters()
        self._chapters_cache = chapters
        by_id = {ch.id: ch for ch in chapters}

        self.tree_chapters.blockSignals(True)
        try:
            for i in range(self.tree_chapters.topLevelItemCount()):
                top_it = self.tree_chapters.topLevelItem(i)
                top_id = int(top_it.data(0, Qt.UserRole))
                if top_it.childCount() > 0:
                    group_selected: list[bool] = [by_id[top_id].selected] if top_id in by_id else []
                    for j in range(top_it.childCount()):
                        ch_it = top_it.child(j)
                        cid = int(ch_it.data(0, Qt.UserRole))
                        sel = by_id[cid].selected if cid in by_id else False
                        ch_it.setCheckState(0, Qt.Checked if sel else Qt.Unchecked)
                        group_selected.append(sel)
                    all_sel = all(group_selected)
                    any_sel = any(group_selected)
                    top_it.setCheckState(
                        0, Qt.Checked if all_sel else (Qt.PartiallyChecked if any_sel else Qt.Unchecked)
                    )
                else:
                    sel = by_id[top_id].selected if top_id in by_id else False
                    top_it.setCheckState(0, Qt.Checked if sel else Qt.Unchecked)
        finally:
            self.tree_chapters.blockSignals(False)

        sel_ch = sum(1 for ch in chapters if ch.selected)
        sel_chunks = sum(ch.total_chunks for ch in chapters if ch.selected)
        est_min = round(sel_chunks * 6.0 / 60.0)
        mode_lbl = "🎭 3 głosy" if self.db.is_multi_voice() else "🎙 1 głos"
        self.lbl_book_stats.setText(
            f"Zaznaczone rozdziały: {sel_ch}/{len(chapters)} | "
            f"Fragmenty: {sel_chunks} | Szacowana długość: ~{est_min} min | Tryb: {mode_lbl}"
        )
        self._update_chapter_toggle_buttons()

    def _on_chapter_checkbox_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self.db is None or column != 0:
            return
        ch_id = item.data(0, Qt.UserRole)
        if ch_id is None:
            return
        checked = item.checkState(0) == Qt.Checked
        if item.childCount() > 0:
            # Zaznaczenie/odznaczenie nagłówka części zaznacza/odznacza całą część razem z podsekcjami w 1 transakcji!
            ids = [int(ch_id)]
            for i in range(item.childCount()):
                child_id = item.child(i).data(0, Qt.UserRole)
                if child_id is not None:
                    ids.append(int(child_id))
            self.db.set_chapters_selected_bulk(ids, checked)
        else:
            self.db.set_chapter_selected(int(ch_id), checked)
        # Aktualizujemy stany w miejscu (NIGDY nie wywołujemy tree.clear() wewnątrz itemChanged!)
        self._sync_tree_checkstates_inplace()
        self._refresh_chunks_table()
        self._update_progress_widgets()

    def _on_convert_selected_to_multi_voice(self) -> None:
        if self.db is None:
            return
        if any(
            w is not None and w.isRunning()
            for w in (self.batch_worker, self.bielik_worker, self.regen_worker, self.export_worker)
        ):
            QMessageBox.warning(
                self,
                "Projekt zajęty",
                "Najpierw zatrzymaj bieżące generowanie, analizę Bielika lub eksport.",
            )
            return

        selected = [ch for ch in self.db.list_chapters() if ch.selected]
        if not selected:
            QMessageBox.information(
                self,
                "Brak zaznaczonych rozdziałów",
                "Zaznacz ptaszkiem rozdziały lub całą część, którą chcesz przeliczyć na 3 głosy.",
            )
            return

        done_sel = [ch for ch in selected if ch.done_chunks > 0]
        pending_sel = [ch for ch in selected if ch.done_chunks == 0]
        if not pending_sel:
            QMessageBox.warning(
                self,
                "Rozdziały mają już wygenerowane audio",
                f"Wszystkie zaznaczone rozdziały ({len(done_sel)}) mają już wygenerowane pliki WAV.\n"
                "Aby nie stracić gotowego nagrania, odznacz gotowe części i zostaw zaznaczone tylko nietknięte rozdziały.",
            )
            return

        # Sprawdź plik źródłowy EPUB/TXT; jeśli przeniesiony, poproś użytkownika o wskazanie
        src_str = self.db.get_meta("source_path", "")
        src_path = Path(src_str) if src_str else None
        if src_path is None or not src_path.exists() or not src_path.is_file():
            fpath, _ = QFileDialog.getOpenFileName(
                self,
                "Wskaż źródłowy plik EPUB / TXT tej książki",
                "",
                "E-booki (*.epub *.txt);;Wszystkie pliki (*.*)",
            )
            if not fpath:
                return
            src_path = Path(fpath)

        info_skip = (
            f"\n(Pominięte bezpiecznie rozdziały z gotowym audio: {len(done_sel)})"
            if done_sel
            else ""
        )
        reply = QMessageBox.question(
            self,
            "Przeliczyć na 3 głosy?",
            f"Przeliczyć {len(pending_sel)} zaznaczonych (jeszcze niewygenerowanych) rozdziałów na tryb 3 głosów "
            f"(podział dialogów i didaskaliów)?{info_skip}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            voice = self._current_voice_from_ui()
            base_seed = voice.seed if voice else 101
            res = self.db.rechunk_selected_chapters_multi_voice(
                pronunciation_dict=load_pronunciation_dict(),
                source_path=src_path,
                base_seed=base_seed,
                skip_with_done_audio=True,
            )
        except Exception as e:
            QMessageBox.critical(self, "Błąd przeliczania na 3 głosy", str(e))
            return

        self._refresh_chapters_table()
        self._refresh_chunks_table()
        self._on_chapter_selected_preview()
        self._update_progress_widgets()
        self._log(
            f"[3 głosy] Przeliczono {res.converted_chapters} rozdz. → {res.new_chunks} fragmentów "
            f"(w tym {res.dialogue_chunks} dialogów; pominięto gotowych: {res.skipped_done_chapters})."
        )

        ok_bielik, _ = is_bielik_available()
        if res.converted_chapters > 0 and ok_bielik:
            run_now = QMessageBox.question(
                self,
                "Przeliczono na 3 głosy!",
                f"Gotowe! Przeliczono {res.converted_chapters} rozdziałów ({res.new_chunks} fragmentów, "
                f"w tym {res.dialogue_chunks} kwestii dialogowych).\n\n"
                f"Czy od razu uruchomić Bielika AI, aby rozpoznał płeć mówców (M / K) w tych rozdziałach?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if run_now == QMessageBox.Yes:
                self.tabs.setCurrentIndex(2)
                self._on_run_bielik_tagger()
        else:
            QMessageBox.information(
                self,
                "Przeliczono na 3 głosy",
                f"Przeliczono {res.converted_chapters} rozdziałów ({res.new_chunks} fragmentów, "
                f"w tym {res.dialogue_chunks} dialogów).",
            )

    def _on_chapter_selected_preview(self) -> None:
        self._update_chapter_toggle_buttons()
        if self.db is None:
            return
        sel_items = self.tree_chapters.selectedItems()
        if not sel_items:
            return
        ch_id = sel_items[0].data(0, Qt.UserRole)
        if ch_id is None:
            return
        chunks = self.db.list_chunks(chapter_id=int(ch_id))
        if self.db.is_multi_voice():
            r_tag = {"narrator": "N", "male": "M", "female": "K", "dialogue": "?"}
            lines = [
                f"[{c.chunk_index + 1:03d}] [{r_tag.get(c.voice_role, 'N')}] ({c.pause_after_ms}ms)  {c.normalized_text}"
                for c in chunks
            ]
        else:
            lines = [f"[{c.chunk_index + 1:03d}] ({c.pause_after_ms}ms)  {c.normalized_text}" for c in chunks]
        self.txt_chapter_preview.setPlainText("\n".join(lines))

    # =========================================================================
    # LOGIKA: PROFILE LEKTORÓW (3 PODZAKŁADKI: NARRATOR / M / K)
    # =========================================================================
    def _refresh_voices(self, select_name: str | None = None, target_role: str | None = None) -> None:
        profiles = list_voice_profiles(VOICES_DIR)
        roles = load_voice_roles()
        if target_role and select_name:
            roles[target_role] = select_name
            save_voice_roles(roles)
        for role_key, panel in self._voice_panels.items():
            wanted = roles.get(role_key)
            panel.set_profiles(profiles, select_name=wanted)

    def _on_voice_profiles_changed(self, role_key: str, selected_name: str) -> None:
        self._refresh_voices(select_name=selected_name, target_role=role_key)

    def _on_role_profile_assigned(self, role_key: str, profile_name: str) -> None:
        roles = load_voice_roles()
        if roles.get(role_key) != profile_name:
            roles[role_key] = profile_name
            save_voice_roles(roles)

    def _current_voice_from_ui(self) -> VoiceProfile | None:
        return self._voice_panels["narrator"].current_profile()

    def _current_voice_map(self) -> dict[str, VoiceProfile]:
        result: dict[str, VoiceProfile] = {}
        for role_key, panel in self._voice_panels.items():
            prof = panel.current_profile()
            if prof is not None:
                result[role_key] = prof
        return result

    def _set_voice_edit_enabled(self, enabled: bool) -> None:
        for panel in self._voice_panels.values():
            panel.set_edit_enabled(enabled)
        self.chk_edit_voice.setText(
            "🔓 Edytuj ustawienia lektorów (odblokowane)" if enabled else "🔒 Edytuj ustawienia lektorów"
        )

    def _on_toggle_voice_edit_lock(self, checked: bool) -> None:
        if checked:
            reply = QMessageBox.warning(
                self,
                "Uwaga — edycja lektorów",
                "Możesz zepsuć ustawienia swoich lektorów! Czy na pewno chcesz odblokować edycję parametrów?",
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

    def _on_run_single_test_for_role(self, role_key: str, profile: object, raw_text: str) -> None:
        if (self.batch_worker and self.batch_worker.isRunning()) or (
            self.bielik_worker and self.bielik_worker.isRunning()
        ):
            QMessageBox.warning(self, "Zajęte", "Najpierw zatrzymaj bieżące zadanie na GPU.")
            return
        if not isinstance(profile, VoiceProfile) or not profile.ref_audio_path.exists() or not profile.ref_text:
            QMessageBox.warning(
                self,
                "Niepełny profil lektora",
                "Uzupełnij istniejący plik próbki WAV oraz jego dokładną transkrypcję.",
            )
            return
        norm_text = normalize_polish_text(raw_text, load_pronunciation_dict())
        self._active_test_role = role_key
        for p in self._voice_panels.values():
            p.btn_run_test.setEnabled(False)
        self._voice_panels[role_key].lbl_test_status.setText("Generuję próbkę na GPU...")
        out_wav = OUTPUT_DIR / f"_voice_test_{role_key}.wav"
        self.test_worker = SingleTestWorker(self.engine, profile, norm_text, out_wav)
        self.test_worker.finished_ok.connect(self._on_single_test_done)
        self.test_worker.failed.connect(self._on_single_test_failed)
        self.test_worker.start()

    def _on_single_test_done(self, wav_path: str, dur: float) -> None:
        for p in self._voice_panels.values():
            p.btn_run_test.setEnabled(True)
        panel = self._voice_panels.get(self._active_test_role)
        if panel is not None:
            panel.lbl_test_status.setText(f"Gotowe ({dur:.2f}s) — odtwarzam...")
        _play_wav_async(Path(wav_path))

    def _on_single_test_failed(self, err: str) -> None:
        for p in self._voice_panels.values():
            p.btn_run_test.setEnabled(True)
        panel = self._voice_panels.get(self._active_test_role)
        if panel is not None:
            panel.lbl_test_status.setText("Błąd!")
        QMessageBox.critical(self, "Błąd syntezy testowej", err)

    # =========================================================================
    # LOGIKA: GENEROWANIE WSADOWE, TABELA CHUNKÓW I EKSPORT
    # =========================================================================
    def _current_batch_options(self) -> BatchOptions:
        return BatchOptions(
            use_quality_guard=self.chk_qg.isChecked(),
            qg_max_retries=int(self.spn_qg_retries.value()),
            qg_min_similarity=float(self.spn_qg_thresh.value()),
            qg_device="auto",
            gpu_batch_size=3 if self.chk_turbo_batch.isChecked() else 1,
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

        state = self._batch_state
        if state in ("idle", "stopped") and total > 0 and done >= total:
            state = "done"

        # Wyraźny status i kolor paska postępu — od razu widać czy generowanie trwa, czy stoi
        badge_styles = {
            "running": ("🟢 GENEROWANIE", "#163b29", "#2ea043", "#7ee787", "#2d6cdf"),
            "paused": ("🟡 PAUZA", "#3d3112", "#d29922", "#f2cc60", "#d29922"),
            "stopping": ("🟠 ZATRZYMYWANIE...", "#422512", "#db6d28", "#ffa657", "#db6d28"),
            "stopped": ("⏹ ZATRZYMANO", "#252932", "#484f58", "#adb5bd", "#4a5568"),
            "done": ("✅ UKOŃCZONO", "#163b29", "#2ea043", "#7ee787", "#2ea043"),
            "error": ("🔴 BŁĄD", "#451a1c", "#da3633", "#ff7b72", "#da3633"),
            "idle": ("⚪ GOTOWY", "#21252e", "#3c4352", "#9aa3b2", "#4a5568"),
        }
        badge_text, bg, border, fg, bar_color = badge_styles.get(state, badge_styles["idle"])
        self.lbl_batch_badge.setText(badge_text)
        self.lbl_batch_badge.setStyleSheet(
            f"background-color: {bg}; border: 1px solid {border}; color: {fg}; "
            f"border-radius: 5px; padding: 4px 10px; font-weight: bold; font-size: 9pt;"
        )
        bar_css = (
            f"QProgressBar {{ border: 1px solid #323846; border-radius: 5px; text-align: center; "
            f"background-color: #13151a; }} "
            f"QProgressBar::chunk {{ background-color: {bar_color}; border-radius: 4px; }}"
        )
        self.progress_bar.setStyleSheet(bar_css)
        self.status_progress_bar.setStyleSheet(bar_css)

        if not self._ai_libs_ready:
            # Nie zasłaniaj paska ładowania bibliotek AI w tle statycznym stanem projektu
            self.lbl_global_status.setText(
                "⏳ Wczytuję biblioteki AI (PyTorch / CUDA) w tle — możesz już otwierać projekty i słuchać..."
            )
            self.status_progress_bar.setRange(0, 0)
            self.status_progress_bar.setVisible(True)
        elif state in ("running", "paused", "stopping"):
            self.lbl_progress.setText(
                f"Postęp: {done} / {total} ({pct}%) | Błędów: {err} | Upłynęło: {elapsed_s} | ETA: {eta_s} | VRAM: {vram_s}"
            )
            self.lbl_global_status.setText(
                f"{badge_text}: {done} / {total} ({pct}%) | Upłynęło: {elapsed_s} | ETA: {eta_s}"
            )
            self.status_progress_bar.setRange(0, 100)
            self.status_progress_bar.setValue(pct)
            self.status_progress_bar.setVisible(True)
        elif self.db is not None and total > 0:
            self.lbl_progress.setText(
                f"Postęp: {done} / {total} ({pct}%) | Błędów: {err} | Czas ostatniej sesji: {elapsed_s}"
            )
            self.lbl_global_status.setText(
                f"{badge_text} ({self.db.project_dir.name}): {done} / {total} ({pct}%)"
            )
            self.status_progress_bar.setRange(0, 100)
            self.status_progress_bar.setValue(pct)
            self.status_progress_bar.setVisible(done > 0)
        else:
            self.lbl_progress.setText("Postęp: 0 / 0 | Upłynęło: 0s | ETA: — | VRAM: —")
            self.lbl_global_status.setText("Gotowy")
            self.status_progress_bar.setVisible(False)

    def _populate_chunk_row(self, row_idx: int, c: ChunkRow) -> None:
        ro_flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        ed_flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

        was_blocked = self.tbl_chunks.blockSignals(True)
        try:
            # Kolumna 0: lekki przycisk Play (kliknięcie od razu odtwarza gotowy fragment)
            play_item = QTableWidgetItem("▶" if c.status == "done" else "·")
            play_item.setFlags(ro_flags)
            play_item.setTextAlignment(Qt.AlignCenter)
            play_item.setData(Qt.UserRole, c.id)
            if c.status == "done":
                play_item.setForeground(QColor("#57d9a3"))
                play_item.setToolTip(f"Kliknij, aby odtworzyć fragment #{c.chunk_index + 1}")
            else:
                play_item.setForeground(QColor("#434c5e"))
                play_item.setToolTip("Fragment jeszcze nie wygenerowany")
            self.tbl_chunks.setItem(row_idx, _COL_PLAY, play_item)

            # Kolumna 1: rola głosowa (1-klik przełącza N -> M -> K -> N)
            v_lbl, v_col, v_tip = _VOICE_ROLE_DISPLAY.get(c.voice_role, _VOICE_ROLE_DISPLAY["narrator"])
            voice_item = QTableWidgetItem(v_lbl)
            voice_item.setFlags(ro_flags)
            voice_item.setTextAlignment(Qt.AlignCenter)
            voice_item.setForeground(QColor(v_col))
            voice_item.setToolTip(v_tip)
            self.tbl_chunks.setItem(row_idx, _COL_VOICE, voice_item)

            it_ch = QTableWidgetItem(c.chapter_title[:28])
            it_ch.setFlags(ro_flags)
            self.tbl_chunks.setItem(row_idx, _COL_CHAPTER, it_ch)

            it_num = QTableWidgetItem(str(c.chunk_index + 1))
            it_num.setFlags(ro_flags)
            self.tbl_chunks.setItem(row_idx, _COL_NUM, it_num)

            st_item = QTableWidgetItem(c.status)
            st_item.setFlags(ro_flags)
            if c.status == "done":
                st_item.setForeground(QColor("#57d9a3"))
            elif c.status == "error":
                st_item.setForeground(QColor("#ff6b6b"))
            self.tbl_chunks.setItem(row_idx, _COL_STATUS, st_item)

            dur_s = f"{c.duration_sec:.1f}s" if c.duration_sec > 0 else "—"
            it_dur = QTableWidgetItem(dur_s)
            it_dur.setFlags(ro_flags)
            self.tbl_chunks.setItem(row_idx, _COL_DUR, it_dur)

            qg_s = f"{c.qg_score:.2f}" if c.qg_score is not None else "—"
            qg_item = QTableWidgetItem(qg_s)
            qg_item.setFlags(ro_flags)
            if c.qg_score is not None and c.qg_score < 0.80:
                qg_item.setForeground(QColor("#f5a623"))
            self.tbl_chunks.setItem(row_idx, _COL_QG, qg_item)

            txt_item = QTableWidgetItem(c.normalized_text)
            txt_item.setFlags(ed_flags)  # tylko kolumna tekstu jest edytowalna dwuklikiem!
            txt_item.setToolTip(c.normalized_text)
            self.tbl_chunks.setItem(row_idx, _COL_TEXT, txt_item)

            heard_str = c.heard_text or c.error_msg
            heard_item = QTableWidgetItem(heard_str)
            heard_item.setFlags(ro_flags)
            heard_item.setToolTip(heard_str)
            self.tbl_chunks.setItem(row_idx, _COL_HEARD, heard_item)
        finally:
            self.tbl_chunks.blockSignals(was_blocked)

    def _sort_chunks_list(self, rows: list[ChunkRow]) -> list[ChunkRow]:
        col = self._chunks_sort_col
        reverse = self._chunks_sort_order == Qt.DescendingOrder

        if col in (_COL_PLAY, _COL_CHAPTER, _COL_NUM):
            key = lambda c: (c.chapter_index, c.chunk_index)
        elif col == _COL_VOICE:
            key = lambda c: (c.voice_role, c.chapter_index, c.chunk_index)
        elif col == _COL_STATUS:
            key = lambda c: (c.status, c.chapter_index, c.chunk_index)
        elif col == _COL_DUR:
            key = lambda c: (c.duration_sec, c.chapter_index, c.chunk_index)
        elif col == _COL_QG:
            # Przy sortowaniu po QG rosnąco: najniższe oceny (np. 0.45, 0.72) lądują na samej górze,
            # a puste (pending / None) na dole!
            if not reverse:
                return sorted(
                    rows,
                    key=lambda c: (1 if c.qg_score is None else 0, c.qg_score if c.qg_score is not None else 9.9, c.chapter_index, c.chunk_index),
                )
            return sorted(
                rows,
                key=lambda c: (1 if c.qg_score is None else 0, -(c.qg_score if c.qg_score is not None else -1.0), c.chapter_index, c.chunk_index),
            )
        elif col == _COL_TEXT:
            key = lambda c: c.normalized_text.lower()
        else:
            key = lambda c: (c.heard_text or c.error_msg).lower()

        return sorted(rows, key=key, reverse=reverse)

    def _on_chunks_header_clicked(self, col: int) -> None:
        if col == _COL_PLAY:
            return
        if self._chunks_sort_col == col:
            self._chunks_sort_order = (
                Qt.DescendingOrder if self._chunks_sort_order == Qt.AscendingOrder else Qt.AscendingOrder
            )
        else:
            self._chunks_sort_col = col
            # Kliknięcie w QG domyślnie sortuje rosnąco (najniższe/najgorsze na górze!)
            self._chunks_sort_order = Qt.AscendingOrder

        self.tbl_chunks.horizontalHeader().setSortIndicator(self._chunks_sort_col, self._chunks_sort_order)
        self._refresh_chunks_table()

    def _chunk_at_table_row(self, row_idx: int) -> ChunkRow | None:
        if 0 <= row_idx < len(self._tbl_row_meta):
            kind, payload = self._tbl_row_meta[row_idx]
            if kind == "chunk":
                return payload  # type: ignore[return-value]
        return None

    def _refresh_chunk_chapter_combo(self) -> None:
        """Odświeża listę wyboru części/rozdziału w Zakładce 3 z zachowaniem bieżącego wyboru."""
        prev_data = self.combo_chunk_chapter.currentData()
        self.combo_chunk_chapter.blockSignals(True)
        self.combo_chunk_chapter.clear()
        self.combo_chunk_chapter.addItem("Wszystkie rozdziały", ("all", None))

        selected_idx = 0
        groups = _group_chapters_into_parts(self._chapters_cache)
        for parent_ch, children in groups:
            if children:
                all_grp = [parent_ch, *children]
                g_done = sum(c.done_chunks for c in all_grp)
                g_tot = sum(c.total_chunks for c in all_grp)
                data = ("part", parent_ch.id)
                self.combo_chunk_chapter.addItem(f"▸ {parent_ch.title} ({g_done}/{g_tot})", data)
                if prev_data == data:
                    selected_idx = self.combo_chunk_chapter.count() - 1
                for ch in children:
                    c_data = ("chapter", ch.id)
                    self.combo_chunk_chapter.addItem(f"    {ch.title} ({ch.done_chunks}/{ch.total_chunks})", c_data)
                    if prev_data == c_data:
                        selected_idx = self.combo_chunk_chapter.count() - 1
            else:
                c_data = ("chapter", parent_ch.id)
                self.combo_chunk_chapter.addItem(
                    f"{parent_ch.title} ({parent_ch.done_chunks}/{parent_ch.total_chunks})", c_data
                )
                if prev_data == c_data:
                    selected_idx = self.combo_chunk_chapter.count() - 1

        self.combo_chunk_chapter.setCurrentIndex(selected_idx)
        self.combo_chunk_chapter.blockSignals(False)

    def _on_toggle_tab3_collapse(self) -> None:
        if not self._tab3_active_group_ids:
            return
        any_expanded = any(gid not in self._tab3_collapsed_groups for gid in self._tab3_active_group_ids)
        if any_expanded:
            self._tab3_collapsed_groups.update(self._tab3_active_group_ids)
        else:
            for gid in self._tab3_active_group_ids:
                self._tab3_collapsed_groups.discard(gid)
        self._refresh_chunks_table()

    def _populate_group_banner_row(
        self, row_idx: int, group_id: int, title: str, done_cnt: int, total_cnt: int, collapsed: bool
    ) -> None:
        arrow = "▶" if collapsed else "▼"
        hint = "kliknij, aby rozwinąć" if collapsed else "kliknij, aby zwinąć"
        pct = int((done_cnt / total_cnt) * 100) if total_cnt > 0 else 0
        banner_text = f"  {arrow}  {title}    •    {done_cnt} / {total_cnt} gotowych ({pct}%)    —  {hint}"

        banner_item = QTableWidgetItem(banner_text)
        banner_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        banner_item.setData(Qt.UserRole, group_id)
        f = QFont()
        f.setBold(True)
        banner_item.setFont(f)
        banner_item.setBackground(QBrush(QColor("#1d2534")))
        banner_item.setForeground(QBrush(QColor("#57d9a3" if (total_cnt > 0 and done_cnt >= total_cnt) else "#9dc1fb")))
        self.tbl_chunks.setItem(row_idx, 0, banner_item)
        self.tbl_chunks.setSpan(row_idx, 0, 1, _COL_COUNT)

    def _on_chunk_item_changed(self, item: QTableWidgetItem) -> None:
        if self.db is None or item.column() != _COL_TEXT:
            return
        row_idx = item.row()
        c = self._chunk_at_table_row(row_idx)
        if c is None:
            return
        new_text = item.text().strip()
        if not new_text:
            # Nie pozwalamy zostawić pustego tekstu — przywracamy stary
            self._populate_chunk_row(row_idx, c)
            return
        if new_text == c.normalized_text:
            return

        self.db.update_chunk_text(c.id, new_text)
        fresh = self.db.get_chunk(c.id)
        self._tbl_row_meta[row_idx] = ("chunk", fresh)
        self._populate_chunk_row(row_idx, fresh)
        self._update_progress_widgets()
        self._log(f"[Edycja] [{fresh.chapter_title}] #{fresh.chunk_index + 1} zmieniono tekst → status: pending")

    def _refresh_chunks_table(self) -> None:
        if self.db is None:
            self.tbl_chunks.clearSpans()
            self.tbl_chunks.setRowCount(0)
            self._chunk_rows_cache = []
            self._tbl_row_meta = []
            self._chunk_id_to_row = {}
            self._group_id_to_row = {}
            self._tab3_active_group_ids = []
            self.btn_toggle_chunks_collapse.setEnabled(False)
            return

        mode = self.combo_chunk_filter.currentIndex()
        if mode == 5:
            rows = [c for c in self.db.list_chunks(only_selected=False) if c.status == "done"]
        else:
            all_chunks = self.db.list_chunks(only_selected=True)
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

        # Filtr po wybranej części / rozdziale w combo Sekcja
        scope = self.combo_chunk_chapter.currentData() or ("all", None)
        scope_kind, scope_id = scope
        if scope_kind == "part" and scope_id is not None:
            rows = [c for c in rows if self._chapter_to_part_id.get(c.chapter_id, c.chapter_id) == scope_id]
        elif scope_kind == "chapter" and scope_id is not None:
            rows = [c for c in rows if c.chapter_id == scope_id]

        rows = self._sort_chunks_list(rows)
        self._chunk_rows_cache = rows

        # Grupowanie ze zwijanymi banerami aktywne przy kolejności książkowej (# / Rozdział rosnąco)
        is_book_order = self._chunks_sort_col in (_COL_PLAY, _COL_CHAPTER, _COL_NUM) and self._chunks_sort_order == Qt.AscendingOrder
        has_parts = len(set(self._chapter_to_part_id.values())) < len(self._chapter_to_part_id)

        use_grouping = False
        group_by_chapter = False
        if is_book_order and rows:
            if scope_kind == "all" and (has_parts or len({c.chapter_id for c in rows}) > 1):
                use_grouping = True
                group_by_chapter = not has_parts
            elif scope_kind == "part" and len({c.chapter_id for c in rows}) > 1:
                use_grouping = True
                group_by_chapter = True

        self._tbl_row_meta = []
        self._chunk_id_to_row = {}
        self._group_id_to_row = {}
        self._tab3_active_group_ids = []

        self.tbl_chunks.setUpdatesEnabled(False)
        self.tbl_chunks.blockSignals(True)
        try:
            self.tbl_chunks.clearSpans()
            if not use_grouping:
                self.tbl_chunks.setRowCount(len(rows))
                for i, c in enumerate(rows):
                    self._tbl_row_meta.append(("chunk", c))
                    self._chunk_id_to_row[c.id] = i
                    self._populate_chunk_row(i, c)
            else:
                # Zgrupuj wiersze zachowując kolejność występowania
                ordered_groups: list[tuple[int, str, list[ChunkRow]]] = []
                grp_map: dict[int, int] = {}
                for c in rows:
                    gid = c.chapter_id if group_by_chapter else self._chapter_to_part_id.get(c.chapter_id, c.chapter_id)
                    if gid not in grp_map:
                        g_title = c.chapter_title if group_by_chapter else self._part_titles.get(gid, c.chapter_title)
                        grp_map[gid] = len(ordered_groups)
                        ordered_groups.append((gid, g_title, []))
                    ordered_groups[grp_map[gid]][2].append(c)

                # Policz łączną liczbę widocznych wierszy (banery + rozwinięte chunki)
                visible_count = 0
                for gid, _g_title, g_chunks in ordered_groups:
                    self._tab3_active_group_ids.append(gid)
                    visible_count += 1
                    if gid not in self._tab3_collapsed_groups:
                        visible_count += len(g_chunks)

                self.tbl_chunks.setRowCount(visible_count)
                r_i = 0
                for gid, g_title, g_chunks in ordered_groups:
                    collapsed = gid in self._tab3_collapsed_groups
                    g_done = sum(1 for c in g_chunks if c.status == "done")
                    self._tbl_row_meta.append(("group", gid))
                    self._group_id_to_row[gid] = r_i
                    self._populate_group_banner_row(r_i, gid, g_title, g_done, len(g_chunks), collapsed)
                    r_i += 1
                    if not collapsed:
                        for c in g_chunks:
                            self._tbl_row_meta.append(("chunk", c))
                            self._chunk_id_to_row[c.id] = r_i
                            self._populate_chunk_row(r_i, c)
                            r_i += 1
        finally:
            self.tbl_chunks.blockSignals(False)
            self.tbl_chunks.setUpdatesEnabled(True)

        if self._tab3_active_group_ids:
            self.btn_toggle_chunks_collapse.setEnabled(True)
            any_exp = any(gid not in self._tab3_collapsed_groups for gid in self._tab3_active_group_ids)
            lbl_unit = "rozdziały" if group_by_chapter else "części"
            self.btn_toggle_chunks_collapse.setText(f"⊟ Zwiń {lbl_unit}" if any_exp else f"⊞ Rozwiń {lbl_unit}")
        else:
            self.btn_toggle_chunks_collapse.setEnabled(False)
            self.btn_toggle_chunks_collapse.setText("⊟ Zwiń części")

    def _on_start_batch(self) -> None:
        if not self._ai_libs_ready:
            QMessageBox.information(self, "Trwa uruchamianie AI", "Poczekaj kilka sekund na dokończenie wczytywania bibliotek AI.")
            return
        if self.db is None:
            QMessageBox.warning(self, "Brak projektu", "Najpierw wczytaj książkę w zakładce 1.")
            return
        if self.bielik_worker and self.bielik_worker.isRunning():
            QMessageBox.warning(self, "Bielik pracuje", "Poczekaj aż zakończy się analiza dialogów Bielik AI.")
            return
        voice = self._current_voice_from_ui()
        if voice is None or not voice.ref_audio_path.exists() or not voice.ref_text:
            QMessageBox.warning(self, "Brak profilu narratora", "Sprawdź ustawienia Narratora w zakładce 2.")
            return

        voice_map = self._current_voice_map()
        self._batch_state = "running"
        self.btn_start.setEnabled(False)
        self.btn_bielik_tag.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸ Pauza")
        self.btn_stop.setEnabled(True)
        self.btn_stop.setText("⏹ Stop")
        self.btn_export.setEnabled(False)

        self._elapsed_base_sec = 0.0
        self._run_segment_start = time.monotonic()
        self._last_progress = None
        self._elapsed_timer.start()
        self._update_progress_widgets()

        self.batch_worker = BatchWorker(
            self.engine, self.db, voice, self._current_batch_options(), voice_map=voice_map
        )
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
        self._batch_state = "paused" if paused else "running"
        if paused:
            if self._run_segment_start is not None:
                self._elapsed_base_sec += max(0.0, time.monotonic() - self._run_segment_start)
                self._run_segment_start = None
        else:
            self._run_segment_start = time.monotonic()
        self.btn_pause.setText("▶ Wznów" if paused else "⏸ Pauza")
        self._update_progress_widgets(self._last_progress)
        self._log("Wstrzymano (pauza) — dokończę bieżące zdanie." if paused else "Wznowiono generowanie.")

    def _on_stop_batch(self) -> None:
        if self.batch_worker:
            self._batch_state = "stopping"
            self.btn_stop.setEnabled(False)
            self.btn_stop.setText("⏳ Zatrzymuję...")
            self.btn_pause.setEnabled(False)
            self.batch_worker.request_stop()
            self._update_progress_widgets(self._last_progress)
            self._log("Zatrzymuję po bieżącym zdaniu...")

    def _on_batch_progress(self, prog: BatchProgress) -> None:
        self._update_progress_widgets(prog)
        # Aktualizujemy w O(1) TYLKO ten 1 ukończony wiersz (nigdy nie przebudowujemy całej tabeli co 5 zdań!)
        if self.db is not None and prog.current_chunk is not None:
            row_idx = self._chunk_id_to_row.get(prog.current_chunk.id)
            if row_idx is not None and 0 <= row_idx < len(self._tbl_row_meta):
                fresh = self.db.get_chunk(prog.current_chunk.id)
                self._tbl_row_meta[row_idx] = ("chunk", fresh)
                self._populate_chunk_row(row_idx, fresh)
            if self._last_chapter_id_seen != prog.current_chunk.chapter_id:
                self._last_chapter_id_seen = prog.current_chunk.chapter_id
                self._refresh_chapters_table()

    def _on_batch_finished(self, prog: BatchProgress) -> None:
        was_stopping = self._batch_state == "stopping"
        self._batch_state = "done" if (prog.total > 0 and prog.done >= prog.total) else ("stopped" if was_stopping else "idle")
        if self._run_segment_start is not None:
            self._elapsed_base_sec += max(0.0, time.monotonic() - self._run_segment_start)
            self._run_segment_start = None
        self._elapsed_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_bielik_tag.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸ Pauza")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("⏹ Stop")
        self.btn_export.setEnabled(True)
        self._update_progress_widgets(prog)
        self._refresh_chapters_table()
        self._refresh_chunks_table()
        self.player_widget.refresh_new_chunks()
        self._log(prog.message)

    def _on_batch_failed(self, err: str) -> None:
        self._batch_state = "error"
        if self._run_segment_start is not None:
            self._elapsed_base_sec += max(0.0, time.monotonic() - self._run_segment_start)
            self._run_segment_start = None
        self._elapsed_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_bielik_tag.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸ Pauza")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("⏹ Stop")
        self.btn_export.setEnabled(True)
        self._update_progress_widgets(self._last_progress)
        self._log(f"[BŁĄD KRYTYCZNY] {err}")
        QMessageBox.critical(self, "Błąd generowania", err)

    def _on_run_bielik_tagger(self) -> None:
        if self.db is None:
            QMessageBox.warning(self, "Brak projektu", "Najpierw wczytaj książkę w zakładce 1.")
            return
        if self.bielik_worker and self.bielik_worker.isRunning():
            self.bielik_worker.request_stop()
            self.btn_bielik_tag.setEnabled(False)
            self.btn_bielik_tag.setText("⏳ Przerywam Bielika...")
            return
        if any(w is not None and w.isRunning() for w in (self.batch_worker, self.regen_worker, self.test_worker)):
            QMessageBox.warning(self, "GPU zajęte", "Najpierw zatrzymaj bieżące generowanie lub test audio.")
            return
        ok, info = is_bielik_available()
        if not ok:
            QMessageBox.warning(
                self,
                "Brak modelu Bielik AI",
                f"Nie znaleziono lokalnych plików Bielika ({info}).\n"
                "Uruchom skrypt scripts/setup_bielik.py, aby pobrać model Bielik-4.5B GGUF i llama.cpp.",
            )
            return

        # Zwalniamy F5-TTS z GPU, żeby Bielik-4.5B miał czyste ~5+ GB VRAM na RTX 4050!
        self.engine.unload()
        self.btn_start.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_regen_chunk.setEnabled(False)
        self.btn_bielik_tag.setText("⏹ Przerwij Bielika")
        self.progress_bar.setRange(0, 0)
        self.status_progress_bar.setRange(0, 0)
        self.status_progress_bar.setVisible(True)
        self.lbl_progress.setText(f"🎭 Bielik AI ({info}): uruchamiam i analizuję dialogi...")
        self.lbl_global_status.setText("🎭 Bielik AI analizuje dialogi na GPU...")
        self._log(f"Uruchamiam Bielik AI ({info}) dla zaznaczonych rozdziałów...")

        self.bielik_worker = BielikTaggerWorker(self.db, only_selected=True)
        self.bielik_worker.progress.connect(self._on_bielik_progress)
        self.bielik_worker.log.connect(self._log)
        self.bielik_worker.finished_ok.connect(self._on_bielik_finished)
        self.bielik_worker.failed.connect(self._on_bielik_failed)
        self.bielik_worker.start()

    def _on_bielik_progress(self, done_ch: int, total_ch: int, msg: str) -> None:
        if total_ch > 0:
            pct = int((done_ch / total_ch) * 100)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(pct)
            self.status_progress_bar.setRange(0, 100)
            self.status_progress_bar.setValue(pct)
        self.lbl_progress.setText(f"🎭 {msg}")
        self.lbl_global_status.setText(f"🎭 {msg}")

    def _on_bielik_finished(self, updated: int, male_cnt: int, female_cnt: int) -> None:
        self.btn_start.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_regen_chunk.setEnabled(True)
        self.btn_bielik_tag.setEnabled(True)
        self.btn_bielik_tag.setText("🎭 Przypisz głosy (Bielik AI)")
        self.progress_bar.setRange(0, 100)
        self.status_progress_bar.setRange(0, 100)
        self._refresh_chapters_table()
        self._refresh_chunks_table()
        self._on_chapter_selected_preview()
        self._update_progress_widgets()
        summary = (
            f"Bielik AI zakończył analizę dialogów!\n\n"
            f"• Dialogi męskie (M): {male_cnt}\n"
            f"• Dialogi żeńskie (K): {female_cnt}\n"
            f"• Zaktualizowanych fragmentów: {updated}\n\n"
            f"W kolumnie „Głos” możesz zawsze jednym kliknięciem ręcznie skorygować dowolną linijkę (N → M → K)."
        )
        self._log(f"Bielik AI zakończył: M={male_cnt}, K={female_cnt}, zmieniono={updated} (VRAM zwolniony).")
        QMessageBox.information(self, "Bielik AI — przypisywanie głosów", summary)

    def _on_bielik_failed(self, err: str) -> None:
        self.btn_start.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_regen_chunk.setEnabled(True)
        self.btn_bielik_tag.setEnabled(True)
        self.btn_bielik_tag.setText("🎭 Przypisz głosy (Bielik AI)")
        self.progress_bar.setRange(0, 100)
        self.status_progress_bar.setRange(0, 100)
        self._update_progress_widgets()
        self._log(f"[BŁĄD BIELIK AI] {err}")
        QMessageBox.critical(self, "Błąd Bielik AI", err)

    def _selected_chunk_row(self) -> ChunkRow | None:
        rows = self.tbl_chunks.selectionModel().selectedRows()
        if not rows:
            return None
        return self._chunk_at_table_row(rows[0].row())

    def _play_chunk_at_row(self, row: int) -> None:
        c = self._chunk_at_table_row(row)
        if self.db is None or c is None:
            return
        if c.status != "done":
            return
        p = self.db.resolve_chunk_audio_path(c)
        if p is None or not p.exists():
            return
        self.player_widget.pause_playback()
        _play_wav_async(p)
        self.lbl_global_status.setText(
            f"🔊 Odsłuch: [{c.chapter_title}] #{c.chunk_index + 1} ({c.duration_sec:.1f}s)"
        )

    def _on_chunk_cell_clicked(self, row: int, col: int) -> None:
        if 0 <= row < len(self._tbl_row_meta) and self._tbl_row_meta[row][0] == "group":
            gid = int(self._tbl_row_meta[row][1])
            if gid in self._tab3_collapsed_groups:
                self._tab3_collapsed_groups.discard(gid)
            else:
                self._tab3_collapsed_groups.add(gid)
            self._refresh_chunks_table()
            return
        if col == _COL_PLAY:
            self._play_chunk_at_row(row)
        elif col == _COL_VOICE and self.db is not None:
            c = self._chunk_at_table_row(row)
            if c is None:
                return
            next_role = _VOICE_ROLE_CYCLE.get(c.voice_role, "narrator")
            self.db.set_chunk_voice_role(c.id, next_role, reset_to_pending=(c.status == "done"))
            fresh = self.db.get_chunk(c.id)
            self._tbl_row_meta[row] = ("chunk", fresh)
            self._populate_chunk_row(row, fresh)
            self._update_progress_widgets()
            v_lbl = _VOICE_ROLE_DISPLAY.get(fresh.voice_role, ("🎙 N",))[0]
            self._log(f"[Głos] [{fresh.chapter_title}] #{fresh.chunk_index + 1} → {v_lbl}")

    def _on_chunk_cell_double_clicked(self, row: int, col: int) -> None:
        if 0 <= row < len(self._tbl_row_meta) and self._tbl_row_meta[row][0] == "group":
            return
        # Kolumna Tekst służy do edycji dwuklikiem; Play i Głos reagują na pojedynczy klik
        if col in (_COL_PLAY, _COL_VOICE, _COL_TEXT):
            return
        self._play_chunk_at_row(row)

    def _on_play_selected_chunk(self) -> None:
        c = self._selected_chunk_row()
        p = self.db.resolve_chunk_audio_path(c) if (c is not None and self.db is not None) else None
        if p is None or not p.exists():
            QMessageBox.information(self, "Brak audio", "Zaznaczony fragment nie ma jeszcze pliku WAV.")
            return
        self.player_widget.pause_playback()
        _play_wav_async(p)

    def _on_regenerate_selected_chunk(self) -> None:
        if not self._ai_libs_ready:
            QMessageBox.information(self, "Trwa uruchamianie AI", "Poczekaj kilka sekund na dokończenie wczytywania bibliotek AI.")
            return
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

        self.player_widget.persist_state()
        self.player_widget.release_media()
        self.btn_regen_chunk.setEnabled(False)
        self._log(f"Regeneruję fragment #{c.chunk_index + 1} ({c.chapter_title})...")

        self.regen_worker = RegenerateChunkWorker(
            self.engine,
            self.db,
            c.id,
            voice,
            self._current_batch_options(),
            new_text=None,
            voice_map=self._current_voice_map(),
        )
        self.regen_worker.log.connect(self._log)
        self.regen_worker.finished_ok.connect(self._on_regen_done)
        self.regen_worker.failed.connect(self._on_regen_failed)
        self.regen_worker.start()

    def _on_regen_done(self, updated: ChunkRow) -> None:
        self.btn_regen_chunk.setEnabled(True)
        self._refresh_chunks_table()
        self._update_progress_widgets()
        self.player_widget.refresh_new_chunks()
        self._log(f"Zregenerowano #{updated.chunk_index + 1} (seed={updated.seed}, {updated.duration_sec:.1f}s)")
        if updated.audio_path:
            _play_wav_async(Path(updated.audio_path))

    def _on_regen_failed(self, err: str) -> None:
        self.btn_regen_chunk.setEnabled(True)
        QMessageBox.critical(self, "Błąd regeneracji", err)

    def _on_export(self) -> None:
        if self.db is None:
            QMessageBox.warning(self, "Brak projektu", "Najpierw wczytaj projekt.")
            return
        if self._has_unsaved_book_meta():
            reply = QMessageBox.question(
                self,
                "Niezapisane informacje o książce",
                "Masz niezapisane zmiany w tytule, autorze lub nazwie projektu (Zakładka 1).\n"
                "Czy zapisać je teraz, zanim rozpocznie się eksport M4B/MP3?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Yes:
                self._on_save_book_meta()
                if self._has_unsaved_book_meta():
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
        self.lbl_global_status.setText(f"⏳ Eksportuję audiobooka (FFmpeg EBU R128, {bitrate})...")
        self.status_progress_bar.setRange(0, 0)
        self.status_progress_bar.setVisible(True)
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
        self.status_progress_bar.setRange(0, 100)
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
        self.status_progress_bar.setRange(0, 100)
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

    def _collect_pronunciation_from_table(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for r in range(self.tbl_pron.rowCount()):
            k_item = self.tbl_pron.item(r, 0)
            v_item = self.tbl_pron.item(r, 1)
            k = k_item.text().strip() if k_item else ""
            v = v_item.text().strip() if v_item else ""
            if k and v:
                mapping[k] = v
        return mapping

    def _on_pron_save(self) -> None:
        mapping = self._collect_pronunciation_from_table()
        save_pronunciation_dict(mapping)
        QMessageBox.information(self, "Zapisano", f"Zapisano słownik wymowy ({len(mapping)} wpisów).")

    def _on_pron_apply_to_pending(self) -> None:
        mapping = self._collect_pronunciation_from_table()
        save_pronunciation_dict(mapping)
        if self.db is None:
            QMessageBox.information(
                self,
                "Zapisano słownik",
                f"Zapisano słownik ({len(mapping)} wpisów). Brak otwartego projektu do przeliczenia.",
            )
            return
        changed = self.db.reapply_pronunciation_to_pending(mapping)
        self._refresh_chunks_table()
        self._on_chapter_selected_preview()
        QMessageBox.information(
            self,
            "Zastosowano słownik",
            f"Zapisano słownik ({len(mapping)} wpisów) i zaktualizowano {changed} oczekujących (niewygenerowanych) fragmentów.",
        )

    def closeEvent(self, event) -> None:
        _stop_wav_playback()
        self.player_widget.persist_state()
        self.player_widget.release_media()
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.request_stop()
            self.batch_worker.wait(5000)
        if self.bielik_worker and self.bielik_worker.isRunning():
            self.bielik_worker.request_stop()
            self.bielik_worker.wait(5000)
        if self.preload_worker and self.preload_worker.isRunning():
            self.preload_worker.wait(3000)
        self.engine.unload()
        super().closeEvent(event)
