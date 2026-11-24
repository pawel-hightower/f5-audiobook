"""Główne okno aplikacji desktopowej F5-Audiobook (PySide6)."""

from __future__ import annotations

import os
import shutil
import sys
import time
import winsound
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    QVBoxLayout,
    QWidget,
)

from f5_audiobook.audio.exporter import ExportResult, _safe_filename
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
    slugify_project_name,
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
        self._chunk_id_to_row: dict[int, int] = {}
        self._last_chapter_id_seen: int | None = None
        self._chunks_sort_col: int = 1
        self._chunks_sort_order: Qt.SortOrder = Qt.AscendingOrder

        # Licznik czasu od startu generowania
        self._elapsed_base_sec: float = 0.0
        self._run_segment_start: float | None = None
        self._last_progress: BatchProgress | None = None
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
        self._build_global_status_bar()

        self._refresh_saved_projects()
        self._refresh_voices()
        self._load_pronunciation_table()

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

        top_bar.addWidget(self.btn_open_book)
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

        sel_bar.addWidget(self.btn_toggle_all_chapters)
        sel_bar.addSpacing(6)
        sel_bar.addWidget(self.btn_toggle_highlighted_chapters)
        sel_bar.addSpacing(10)
        self.lbl_highlighted_info = QLabel("")
        self.lbl_highlighted_info.setStyleSheet("color: #8ab4f8; font-size: 9pt;")
        sel_bar.addWidget(self.lbl_highlighted_info)
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
        self.tbl_chapters.setSelectionMode(QAbstractItemView.ExtendedSelection)
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
            "Przywraca wypracowane domyślne parametry syntezy (speed=0.94, CFG=2.20, ODE=36, Seed=202)"
        )
        self.btn_reset_voice_defaults.clicked.connect(self._on_reset_voice_defaults)
        top_bar.addWidget(self.btn_reset_voice_defaults)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Parametry profilu
        form_group = QGroupBox("Ustawienia głosu i silnika F5-TTS IPA")
        form = QFormLayout(form_group)

        self.ed_voice_display = QLineEdit()
        lbl_disp = QLabel("Nazwa wyświetlana:")
        form.addRow(lbl_disp, self.ed_voice_display)

        ref_row_w = QWidget()
        ref_row = QHBoxLayout(ref_row_w)
        ref_row.setContentsMargins(0, 0, 0, 0)
        self.ed_ref_audio = QLineEdit()
        self.btn_browse_ref = QPushButton("Wybierz WAV...")
        self.btn_browse_ref.clicked.connect(self._on_browse_ref_wav)
        btn_play_ref = QPushButton("🔊 Odsłuchaj próbkę")
        btn_play_ref.clicked.connect(self._on_play_ref_wav)
        ref_row.addWidget(self.ed_ref_audio, 1)
        ref_row.addWidget(self.btn_browse_ref)
        ref_row.addWidget(btn_play_ref)
        lbl_ref_wav = QLabel("Plik próbki (6–10 s WAV):")
        form.addRow(lbl_ref_wav, ref_row_w)

        ref_txt_w = QWidget()
        ref_txt_l = QVBoxLayout(ref_txt_w)
        ref_txt_l.setContentsMargins(0, 0, 0, 0)
        self.ed_ref_text = QPlainTextEdit()
        self.ed_ref_text.setMaximumHeight(72)
        self.ed_ref_text.setPlaceholderText("Dokładna transkrypcja słowo w słowo tego, co mówi lektor na próbce WAV...")
        ref_txt_l.addWidget(self.ed_ref_text)
        lbl_ref_txt = QLabel("Transkrypcja próbki:")
        form.addRow(lbl_ref_txt, ref_txt_w)

        params_row = QHBoxLayout()

        def _make_param_box(label_text: str, spin: QWidget, help_title: str, help_desc: str) -> QWidget:
            box = QWidget()
            bl = QHBoxLayout(box)
            bl.setContentsMargins(2, 2, 10, 2)
            lbl = QLabel(label_text)
            bl.addWidget(lbl)
            bl.addWidget(spin)
            # Kontener pozostaje aktywny nawet przy zablokowanym spinboxie -> podpowiedź działa ZAWSZE po najechaniu!
            self._register_voice_help(box, help_title, help_desc)
            self._register_voice_help(lbl, help_title, help_desc)
            self._register_voice_help(spin, help_title, help_desc)
            return box

        self.spn_speed = QDoubleSpinBox()
        self.spn_speed.setRange(0.75, 1.50)
        self.spn_speed.setSingleStep(0.02)
        self.spn_speed.setValue(0.94)

        self.spn_cfg = QDoubleSpinBox()
        self.spn_cfg.setRange(1.0, 3.5)
        self.spn_cfg.setSingleStep(0.1)
        self.spn_cfg.setValue(2.2)

        self.spn_nfe = QSpinBox()
        self.spn_nfe.setRange(16, 48)
        self.spn_nfe.setSingleStep(4)
        self.spn_nfe.setValue(36)

        self.spn_seed = QSpinBox()
        self.spn_seed.setRange(1, 9999999)
        self.spn_seed.setValue(202)

        params_row.addWidget(
            _make_param_box(
                "Tempo (speed):",
                self.spn_speed,
                "Tempo (speed) — prędkość czytania",
                "Określa tempo mowy względem próbki wzorcowej. Niższe wartości (0.88–0.95) dają spokojną, "
                "głębszą narrację i bardzo wyraźną dykcję. Wyższe (1.05–1.20) przyspieszają lektora, "
                "ale powyżej 1.15 mogą prowadzić do połykania końcówek wyrazów. (Wypracowane optimum: 0.94)",
            )
        )
        params_row.addWidget(
            _make_param_box(
                "CFG:",
                self.spn_cfg,
                "CFG (siła prowadzenia / ekspresja głosu)",
                "Steruje tym, jak mocno model trzyma się barwy i wyrazistości głosu z próbki. "
                "Niższe wartości (1.5–1.8) dają miękki, spokojny, bardziej jednostajny ton. "
                "Wyższe (2.1–2.5) podbijają ekspresję, akcenty i „charakter” lektora. Zbyt wysokie (> 2.8) "
                "mogą brzmieć zbyt ostro lub nienaturalnie. (Wypracowane optimum: 2.20)",
            )
        )
        params_row.addWidget(
            _make_param_box(
                "Kroki ODE:",
                self.spn_nfe,
                "Kroki ODE — dokładność generowania dźwięku",
                "Liczba kroków, w których sieć neuronowa „wygładza” dźwięk mowy. "
                "Mniej kroków (16–24) przyspiesza generowanie, ale delikatnie obniża szczegółowość głosu. "
                "Więcej kroków (32–36) daje najczystszą artykulację spółgłosek i naturalną barwę kosztem "
                "nieco dłuższego czasu obliczeń. (Wypracowane optimum: 36)",
            )
        )
        params_row.addWidget(
            _make_param_box(
                "Seed:",
                self.spn_seed,
                "Seed — ziarno losowości (wariant intonacji)",
                "Liczba startowa generatora losowego. Ten sam tekst i ten sam Seed dają powtarzalną intonację; "
                "inna liczba daje nieco inny rozkład akcentów w zdaniu. Gdy Quality Guard ponawia nieudany "
                "fragment, losuje dla niego nowy Seed. (Domyślnie: 202)",
            )
        )
        params_row.addStretch()
        form.addRow("Parametry syntezy:", params_row)
        layout.addWidget(form_group)

        # Wbudowana wycinarka próbki z audiobooka
        cut_wrap = QWidget()
        cut_wrap_l = QVBoxLayout(cut_wrap)
        cut_wrap_l.setContentsMargins(0, 0, 0, 0)
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
        cut_wrap_l.addWidget(self.cut_group)
        layout.addWidget(cut_wrap)

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

        # Panel kontekstowej pomocy (wykorzystuje dolną część zakładki — aktualizuje się po najechaniu myszką!)
        help_group = QGroupBox("Pomoc kontekstowa")
        help_group.setStyleSheet(
            "QGroupBox { border: 1px solid #2d6cdf; background-color: #141821; border-radius: 6px; margin-top: 10px; padding: 10px; }"
            "QGroupBox::title { color: #8ab4f8; }"
        )
        help_lay = QVBoxLayout(help_group)
        self.lbl_voice_help_title = QLabel("💡 Najedź myszką na dowolny parametr powyżej")
        self.lbl_voice_help_title.setStyleSheet("font-weight: bold; font-size: 10.5pt; color: #8ab4f8; background: transparent;")
        self.lbl_voice_help_body = QLabel(
            "Pojawi się tu proste wyjaśnienie, do czego służy dany parametr i jak jego zmiana wpływa na głos lektora.\n"
            "Domyślnie wszystko jest już dostrojone pod Krzysztofa Gosztyłę i bezpiecznie zablokowane — nie musisz tu nic zmieniać."
        )
        self.lbl_voice_help_body.setWordWrap(True)
        self.lbl_voice_help_body.setStyleSheet("color: #cfd5df; background: transparent;")
        help_lay.addWidget(self.lbl_voice_help_title)
        help_lay.addWidget(self.lbl_voice_help_body)
        help_lay.addStretch()
        layout.addWidget(help_group, 1)

        # Rejestracja podpowiedzi dla pozostałych pól Zakładki 2
        self._register_voice_help(
            lbl_ref_wav,
            "Plik próbki głosu (6–10 s WAV)",
            "Krótki, czysty wycinek głosu lektora (optymalnie 7–10 s spokojnej narracji). Model klonuje z niego barwę, "
            "tembr i naturalny rytm. Zbyt długa próbka (> 12 s) psuje tempo, a zbyt krótka (< 5 s) słabiej oddaje głos.",
        )
        self._register_voice_help(
            ref_row_w,
            "Plik próbki głosu (6–10 s WAV)",
            "Krótki, czysty wycinek głosu lektora (optymalnie 7–10 s spokojnej narracji). Model klonuje z niego barwę, "
            "tembr i naturalny rytm. Zbyt długa próbka (> 12 s) psuje tempo, a zbyt krótka (< 5 s) słabiej oddaje głos.",
        )
        self._register_voice_help(
            lbl_ref_txt,
            "Transkrypcja próbki referencyjnej",
            "Dokładny, słowo w słowo zapis tekstu wypowiadanego w pliku WAV (wraz ze znakami interpunkcyjnymi). "
            "Pozwala modelowi idealnie dopasować brzmienie głosu do sylab — literówka może powodować zacinanie.",
        )
        self._register_voice_help(
            ref_txt_w,
            "Transkrypcja próbki referencyjnej",
            "Dokładny, słowo w słowo zapis tekstu wypowiadanego w pliku WAV (wraz ze znakami interpunkcyjnymi). "
            "Pozwala modelowi idealnie dopasować brzmienie głosu do sylab — literówka może powodować zacinanie.",
        )
        self._register_voice_help(
            cut_wrap,
            "Wycinarka próbki (FFmpeg)",
            "Pomocnicze narzędzie do wycięcia czystego 7–10 s fragmentu z istniejącego audiobooka (MP3/M4B/WAV) "
            "do formatu 24 kHz mono WAV. Przydatne, jeśli kiedyś zechcesz dodać profil innego lektora.",
        )
        self._register_voice_help(
            test_group,
            "Szybki test 1 zdania",
            "Generuje pojedyncze zdanie w kilka sekund z aktualnymi ustawieniami i od razu je odtwarza — "
            "świetne do szybkiego sprawdzenia wymowy nowego słowa lub drobnej zmiany parametrów przed całą książką.",
        )
        self._register_voice_help(
            self.btn_reset_voice_defaults,
            "Przycisk „Domyślne”",
            "Bezpiecznie przywraca wypracowane, sprawdzone ustawienia Krzysztofa Gosztyły "
            "(ref_gosztyla.wav, Tempo 0.94, CFG 2.20, Kroki ODE 36, Seed 202), jeśli eksperymenty pójdą w złą stronę.",
        )

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
        self.btn_regen_chunk.setToolTip("Generuje ponownie zaznaczony fragment z nowym seedem (dwuklik na tekście pozwala go wcześniej edytować)")
        self.btn_regen_chunk.clicked.connect(self._on_regenerate_selected_chunk)

        filter_row.addWidget(self.btn_play_chunk)
        filter_row.addWidget(btn_stop_chunk)
        filter_row.addWidget(self.btn_regen_chunk)
        layout.addLayout(filter_row)

        self.tbl_chunks = QTableWidget(0, 7)
        self.tbl_chunks.setHorizontalHeaderLabels(
            ["Rozdział", "#", "Status", "Czas", "QG", "Tekst (do przeczytania — dwuklik edytuje)", "Usłyszano (Whisper)"]
        )
        # Wszystkie kolumny Interactive — można dowolnie zwężać Tekst i rozszerzać Whisper, a klik w nagłówek sortuje!
        hdr = self.tbl_chunks.horizontalHeader()
        for col_i in range(7):
            hdr.setSectionResizeMode(col_i, QHeaderView.Interactive)
        hdr.setStretchLastSection(True)
        hdr.setSortIndicatorShown(True)
        hdr.setSortIndicator(self._chunks_sort_col, self._chunks_sort_order)
        hdr.sectionClicked.connect(self._on_chunks_header_clicked)
        self.tbl_chunks.setColumnWidth(0, 95)
        self.tbl_chunks.setColumnWidth(1, 45)
        self.tbl_chunks.setColumnWidth(2, 68)
        self.tbl_chunks.setColumnWidth(3, 55)
        self.tbl_chunks.setColumnWidth(4, 55)
        self.tbl_chunks.setColumnWidth(5, 430)
        self.tbl_chunks.setColumnWidth(6, 380)
        self.tbl_chunks.verticalHeader().setVisible(False)
        self.tbl_chunks.setAlternatingRowColors(True)
        self.tbl_chunks.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_chunks.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.tbl_chunks.itemChanged.connect(self._on_chunk_item_changed)
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
            self.db = ProjectDB.create_from_book(
                book,
                pronunciation_dict=pron,
                work_root=WORK_DIR,
                base_seed=base_seed,
            )
            self._refresh_saved_projects(select_dir=self.db.project_dir)
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
            self.tbl_chapters.setRowCount(0)
            self.txt_chapter_preview.clear()
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
            self._log(
                f"Zapisano dane książki i zmieniono projekt: {old_proj_name} → {self.db.project_dir.name} "
                f"(„{new_title}” — {new_author})."
            )
            self.lbl_global_status.setText(
                f"Zapisano: „{new_title}” ({new_author}) | projekt: {self.db.project_dir.name}"
            )
        else:
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
        self._update_chapter_toggle_buttons()

    def _update_chapter_toggle_buttons(self) -> None:
        if self.db is None or self.tbl_chapters.rowCount() == 0:
            self.btn_toggle_all_chapters.setText("☑ Zaznacz wszystkie")
            self.btn_toggle_all_chapters.setEnabled(False)
            self.btn_toggle_highlighted_chapters.setText("☑ Zaznacz podświetlone")
            self.btn_toggle_highlighted_chapters.setEnabled(False)
            self.lbl_highlighted_info.setText("")
            return

        self.btn_toggle_all_chapters.setEnabled(True)
        all_checked = all(
            self.tbl_chapters.item(r, 0) is not None
            and self.tbl_chapters.item(r, 0).checkState() == Qt.Checked
            for r in range(self.tbl_chapters.rowCount())
        )
        self.btn_toggle_all_chapters.setText(
            "☐ Odznacz wszystkie" if all_checked else "☑ Zaznacz wszystkie"
        )

        sel_rows = sorted({idx.row() for idx in self.tbl_chapters.selectionModel().selectedRows()})
        if not sel_rows:
            self.btn_toggle_highlighted_chapters.setText("☑ Zaznacz podświetlone")
            self.btn_toggle_highlighted_chapters.setEnabled(False)
            self.lbl_highlighted_info.setText("")
            return

        self.btn_toggle_highlighted_chapters.setEnabled(True)
        all_hl_checked = all(
            self.tbl_chapters.item(r, 0) is not None
            and self.tbl_chapters.item(r, 0).checkState() == Qt.Checked
            for r in sel_rows
        )
        self.btn_toggle_highlighted_chapters.setText(
            "☐ Odznacz podświetlone" if all_hl_checked else "☑ Zaznacz podświetlone"
        )

        # Podsumowanie podświetlonych rozdziałów (liczba sekcji, fragmentów i szacowany czas generowania ~10s/chunk)
        hl_total = 0
        hl_done = 0
        for r in sel_rows:
            prog_item = self.tbl_chapters.item(r, 3)
            if prog_item and "/" in prog_item.text():
                d_s, t_s = prog_item.text().split("/", 1)
                hl_done += int(d_s)
                hl_total += int(t_s)
        hl_pending = max(0, hl_total - hl_done)
        est_gen = _format_eta(hl_pending * 10.0) if hl_pending > 0 else "gotowe!"
        self.lbl_highlighted_info.setText(
            f"Podświetlono: {len(sel_rows)} sekcji | {hl_total} fragm. (do przeliczenia: {hl_pending} ≈ {est_gen})"
        )

    def _on_toggle_all_chapters(self) -> None:
        if self.db is None:
            return
        chapters = self.db.list_chapters()
        target_state = not all(ch.selected for ch in chapters)
        for ch in chapters:
            self.db.set_chapter_selected(ch.id, target_state)
        self._refresh_chapters_table()
        self._refresh_chunks_table()
        self._update_progress_widgets()

    def _on_toggle_highlighted_chapters(self) -> None:
        if self.db is None:
            return
        sel_rows = sorted({idx.row() for idx in self.tbl_chapters.selectionModel().selectedRows()})
        if not sel_rows:
            return
        all_hl_checked = all(
            self.tbl_chapters.item(r, 0) is not None
            and self.tbl_chapters.item(r, 0).checkState() == Qt.Checked
            for r in sel_rows
        )
        target_state = not all_hl_checked
        for row_idx in sel_rows:
            item0 = self.tbl_chapters.item(row_idx, 0)
            if item0 is not None and item0.data(Qt.UserRole) is not None:
                self.db.set_chapter_selected(int(item0.data(Qt.UserRole)), target_state)
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
        self._update_chapter_toggle_buttons()
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
            "Przywrócić wypracowane domyślne parametry syntezy (speed=0.94, CFG=2.20, ODE=36, Seed=202)"
            + (" oraz oryginalną próbkę ref_gosztyla.wav?" if profile.name == "moj_lektor" else "?"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.spn_speed.setValue(0.94)
        self.spn_cfg.setValue(2.2)
        self.spn_nfe.setValue(36)
        self.spn_seed.setValue(202)
        if profile.name == "moj_lektor":
            self.ed_voice_display.setText("Krzysztof Gosztyła (domyślny)")
            self.ed_ref_audio.setText("ref_gosztyla.wav")
            self.ed_ref_text.setPlainText(
                "A niektóre nawet jeszcze się powiększyły, jak to zwykle bywa, "
                "pojawili się też nowi milionerzy żerujący na czasach kryzysu."
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

        # Globalny pasek statusu na dole (widoczny na KAŻDEJ zakładce!)
        is_running = self.batch_worker is not None and self.batch_worker.isRunning()
        is_paused = is_running and self.batch_worker.pause_event.is_set()
        if is_running:
            prefix = "⏸ Pauza" if is_paused else "▶ Generowanie"
            self.lbl_global_status.setText(
                f"{prefix}: {done} / {total} ({pct}%) | Upłynęło: {elapsed_s} | ETA: {eta_s}"
            )
            self.status_progress_bar.setRange(0, 100)
            self.status_progress_bar.setValue(pct)
            self.status_progress_bar.setVisible(True)
        elif self.db is not None and total > 0:
            self.lbl_global_status.setText(
                f"Projekt: {self.db.project_dir.name} — {done} / {total} ({pct}%)"
            )
            self.status_progress_bar.setRange(0, 100)
            self.status_progress_bar.setValue(pct)
            self.status_progress_bar.setVisible(done > 0)
        else:
            self.lbl_global_status.setText("Gotowy")
            self.status_progress_bar.setVisible(False)

    def _populate_chunk_row(self, row_idx: int, c: ChunkRow) -> None:
        ro_flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        ed_flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

        was_blocked = self.tbl_chunks.blockSignals(True)
        try:
            it0 = QTableWidgetItem(c.chapter_title[:28])
            it0.setFlags(ro_flags)
            it0.setData(Qt.UserRole, c.id)
            self.tbl_chunks.setItem(row_idx, 0, it0)

            it1 = QTableWidgetItem(str(c.chunk_index + 1))
            it1.setFlags(ro_flags)
            self.tbl_chunks.setItem(row_idx, 1, it1)

            st_item = QTableWidgetItem(c.status)
            st_item.setFlags(ro_flags)
            if c.status == "done":
                st_item.setForeground(QColor("#57d9a3"))
            elif c.status == "error":
                st_item.setForeground(QColor("#ff6b6b"))
            self.tbl_chunks.setItem(row_idx, 2, st_item)

            dur_s = f"{c.duration_sec:.1f}s" if c.duration_sec > 0 else "—"
            it3 = QTableWidgetItem(dur_s)
            it3.setFlags(ro_flags)
            self.tbl_chunks.setItem(row_idx, 3, it3)

            qg_s = f"{c.qg_score:.2f}" if c.qg_score is not None else "—"
            qg_item = QTableWidgetItem(qg_s)
            qg_item.setFlags(ro_flags)
            if c.qg_score is not None and c.qg_score < 0.80:
                qg_item.setForeground(QColor("#f5a623"))
            self.tbl_chunks.setItem(row_idx, 4, qg_item)

            txt_item = QTableWidgetItem(c.normalized_text)
            txt_item.setFlags(ed_flags)  # tylko kolumna tekstu jest edytowalna dwuklikiem!
            txt_item.setToolTip(c.normalized_text)
            self.tbl_chunks.setItem(row_idx, 5, txt_item)

            heard_str = c.heard_text or c.error_msg
            heard_item = QTableWidgetItem(heard_str)
            heard_item.setFlags(ro_flags)
            heard_item.setToolTip(heard_str)
            self.tbl_chunks.setItem(row_idx, 6, heard_item)
        finally:
            self.tbl_chunks.blockSignals(was_blocked)

    def _sort_chunks_list(self, rows: list[ChunkRow]) -> list[ChunkRow]:
        col = self._chunks_sort_col
        reverse = self._chunks_sort_order == Qt.DescendingOrder

        if col == 0:
            key = lambda c: (c.chapter_index, c.chunk_index)
        elif col == 1:
            key = lambda c: (c.chapter_index, c.chunk_index)
        elif col == 2:
            key = lambda c: (c.status, c.chapter_index, c.chunk_index)
        elif col == 3:
            key = lambda c: (c.duration_sec, c.chapter_index, c.chunk_index)
        elif col == 4:
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
        elif col == 5:
            key = lambda c: c.normalized_text.lower()
        else:
            key = lambda c: (c.heard_text or c.error_msg).lower()

        return sorted(rows, key=key, reverse=reverse)

    def _on_chunks_header_clicked(self, col: int) -> None:
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

    def _on_chunk_item_changed(self, item: QTableWidgetItem) -> None:
        if self.db is None or item.column() != 5:
            return
        row_idx = item.row()
        if not (0 <= row_idx < len(self._chunk_rows_cache)):
            return
        c = self._chunk_rows_cache[row_idx]
        new_text = item.text().strip()
        if not new_text:
            # Nie pozwalamy zostawić pustego tekstu — przywracamy stary
            self._populate_chunk_row(row_idx, c)
            return
        if new_text == c.normalized_text:
            return

        self.db.update_chunk_text(c.id, new_text)
        fresh = self.db.get_chunk(c.id)
        self._chunk_rows_cache[row_idx] = fresh
        self._populate_chunk_row(row_idx, fresh)
        self._update_progress_widgets()
        self._log(f"[Edycja] [{fresh.chapter_title}] #{fresh.chunk_index + 1} zmieniono tekst → status: pending")

    def _refresh_chunks_table(self) -> None:
        if self.db is None:
            self.tbl_chunks.setRowCount(0)
            self._chunk_rows_cache = []
            self._chunk_id_to_row = {}
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

        rows = self._sort_chunks_list(rows)
        self._chunk_rows_cache = rows
        self._chunk_id_to_row = {c.id: i for i, c in enumerate(rows)}

        self.tbl_chunks.setUpdatesEnabled(False)
        self.tbl_chunks.blockSignals(True)
        try:
            self.tbl_chunks.setRowCount(len(rows))
            for i, c in enumerate(rows):
                self._populate_chunk_row(i, c)
        finally:
            self.tbl_chunks.blockSignals(False)
            self.tbl_chunks.setUpdatesEnabled(True)

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
        # Aktualizujemy w O(1) TYLKO ten 1 ukończony wiersz (nigdy nie przebudowujemy całej tabeli co 5 zdań!)
        if self.db is not None and prog.current_chunk is not None:
            row_idx = self._chunk_id_to_row.get(prog.current_chunk.id)
            if row_idx is not None:
                fresh = self.db.get_chunk(prog.current_chunk.id)
                self._chunk_rows_cache[row_idx] = fresh
                self._populate_chunk_row(row_idx, fresh)
            if self._last_chapter_id_seen != prog.current_chunk.chapter_id:
                self._last_chapter_id_seen = prog.current_chunk.chapter_id
                self._refresh_chapters_table()

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
        p = self.db.resolve_chunk_audio_path(c) if (c is not None and self.db is not None) else None
        if p is None or not p.exists():
            QMessageBox.information(self, "Brak audio", "Zaznaczony fragment nie ma jeszcze pliku WAV.")
            return
        _play_wav_async(p)

    def _on_regenerate_selected_chunk(self) -> None:
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

        self.btn_regen_chunk.setEnabled(False)
        self._log(f"Regeneruję fragment #{c.chunk_index + 1} ({c.chapter_title})...")

        self.regen_worker = RegenerateChunkWorker(
            self.engine,
            self.db,
            c.id,
            voice,
            self._current_batch_options(),
            new_text=None,
        )
        self.regen_worker.log.connect(self._log)
        self.regen_worker.finished_ok.connect(self._on_regen_done)
        self.regen_worker.failed.connect(self._on_regen_failed)
        self.regen_worker.start()

    def _on_regen_done(self, updated: ChunkRow) -> None:
        self.btn_regen_chunk.setEnabled(True)
        self._refresh_chunks_table()
        self._update_progress_widgets()
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
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.request_stop()
            self.batch_worker.wait(5000)
        self.engine.unload()
        super().closeEvent(event)
