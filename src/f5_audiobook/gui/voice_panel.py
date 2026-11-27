"""Panel konfiguracji pojedynczej roli głosowej (Narrator / Dialogi męskie / Dialogi żeńskie) w Zakładce 2."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from f5_audiobook.config import VOICES_DIR
from f5_audiobook.voices import (
    VoiceProfile,
    cut_reference_sample,
    save_voice_profile,
    slugify_voice_name,
)


ROLE_DEFAULTS: dict[str, dict] = {
    "narrator": {
        "speed": 0.94,
        "cfg": 2.2,
        "nfe": 36,
        "seed": 202,
        "default_profile": "moj_lektor",
        "test_sentence": (
            "Wieczór zapadał nad miastem powoli, a chłodny wiatr niósł zapach zbliżającego się deszczu."
        ),
        "speed_hint": "Wypracowane optimum narratora (Gosztyła): 0.94",
    },
    "male": {
        "speed": 0.98,
        "cfg": 2.0,
        "nfe": 36,
        "seed": 202,
        "default_profile": "kosior",
        "test_sentence": (
            "Słuchaj, nie mamy na to czasu! Jeśli nie ruszymy teraz, za godzinę będzie już za późno."
        ),
        "speed_hint": "Wypracowane optimum dla Filipa Kosiora (dialogi męskie): speed=0.98, CFG=2.00",
    },
    "female": {
        "speed": 0.98,
        "cfg": 1.55,
        "nfe": 36,
        "seed": 505,
        "default_profile": "kaminska",
        "test_sentence": (
            "Naprawdę myślisz, że uwierzę w tę historię? Powiedz mi wreszcie, co się tam wydarzyło!"
        ),
        "speed_hint": "Wypracowane optimum dla Kamińskiej (dialogi żeńskie): speed=0.98, CFG=1.55",
    },
}


class VoiceRolePanel(QWidget):
    """Podzakładka konfiguracji jednej roli głosowej w Zakładce 2 (niezależne parametry i szybki test)."""

    profiles_changed = Signal(str, str)        # role_key, selected_profile_name
    role_profile_assigned = Signal(str, str)   # role_key, profile_name
    run_test_requested = Signal(str, object, str)  # role_key, VoiceProfile, raw_text
    play_wav_requested = Signal(object)        # Path
    stop_wav_requested = Signal()

    def __init__(
        self,
        role_key: str,
        role_label: str,
        register_help_cb: Callable[[QWidget, str, str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.role_key = role_key
        self.role_label = role_label
        self._register_help = register_help_cb
        self._defaults = ROLE_DEFAULTS.get(role_key, ROLE_DEFAULTS["narrator"])
        self._editable_widgets: list[QWidget] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(8)

        # Pasek wyboru profilu dla tej roli
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel(f"Profil dla roli ({self.role_label}):"))
        self.combo_voices = QComboBox()
        self.combo_voices.setMinimumWidth(250)
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
            f"Przywraca domyślne parametry syntezy dla roli {self.role_label} "
            f"(speed={self._defaults['speed']:.2f}, CFG={self._defaults['cfg']:.2f}, "
            f"ODE={self._defaults['nfe']}, Seed={self._defaults['seed']})"
        )
        self.btn_reset_voice_defaults.clicked.connect(self._on_reset_voice_defaults)
        top_bar.addWidget(self.btn_reset_voice_defaults)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Parametry profilu
        form_group = QGroupBox(f"Ustawienia głosu — {self.role_label}")
        form = QFormLayout(form_group)
        form.setVerticalSpacing(6)

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
        self.ed_ref_text.setMaximumHeight(64)
        self.ed_ref_text.setPlaceholderText(
            "Dokładna transkrypcja słowo w słowo tego, co mówi lektor na próbce WAV..."
        )
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
            self._register_help(box, help_title, help_desc)
            self._register_help(lbl, help_title, help_desc)
            self._register_help(spin, help_title, help_desc)
            return box

        self.spn_speed = QDoubleSpinBox()
        self.spn_speed.setRange(0.75, 1.50)
        self.spn_speed.setSingleStep(0.02)
        self.spn_speed.setValue(float(self._defaults["speed"]))

        self.spn_cfg = QDoubleSpinBox()
        self.spn_cfg.setRange(1.0, 3.5)
        self.spn_cfg.setSingleStep(0.1)
        self.spn_cfg.setValue(float(self._defaults["cfg"]))

        self.spn_nfe = QSpinBox()
        self.spn_nfe.setRange(16, 48)
        self.spn_nfe.setSingleStep(4)
        self.spn_nfe.setValue(int(self._defaults["nfe"]))

        self.spn_seed = QSpinBox()
        self.spn_seed.setRange(1, 9999999)
        self.spn_seed.setValue(int(self._defaults["seed"]))

        params_row.addWidget(
            _make_param_box(
                "Tempo (speed):",
                self.spn_speed,
                "Tempo (speed) — prędkość czytania",
                "Określa tempo mowy względem próbki wzorcowej. Niższe wartości (0.88–0.95) dają spokojną, "
                "głębszą narrację; 1.00 zachowuje w 100% naturalną dynamikę nagrania wzorcowego. "
                f"({self._defaults['speed_hint']})",
            )
        )
        params_row.addWidget(
            _make_param_box(
                "CFG:",
                self.spn_cfg,
                "CFG (siła prowadzenia / ekspresja głosu)",
                "Steruje tym, jak mocno model trzyma się barwy i wyrazistości głosu z próbki. "
                "Niższe wartości (1.5–1.8) dają miękki, spokojny ton; 2.1–2.5 podbijają ekspresję "
                "i charakter aktora. (Wypracowane optimum: 2.20)",
            )
        )
        params_row.addWidget(
            _make_param_box(
                "Kroki ODE:",
                self.spn_nfe,
                "Kroki ODE — dokładność generowania dźwięku",
                "Liczba kroków, w których sieć neuronowa wygładza dźwięk mowy. "
                "32–36 daje najczystszą artykulację spółgłosek i naturalną barwę głosu. (Optimum: 36)",
            )
        )
        params_row.addWidget(
            _make_param_box(
                "Seed:",
                self.spn_seed,
                "Seed — ziarno losowości (wariant intonacji)",
                "Liczba startowa generatora losowego. Ten sam tekst i Seed dają powtarzalną intonację; "
                "zmiana daje inny rozkład akcentów. (Domyślnie: 202)",
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

        # Szybki test 1 zdania dla tego głosu
        test_group = QGroupBox(f"Szybki test 1 zdania — {self.role_label}")
        test_lay = QVBoxLayout(test_group)
        self.ed_test_sentence = QLineEdit(str(self._defaults["test_sentence"]))
        test_lay.addWidget(self.ed_test_sentence)

        test_btns = QHBoxLayout()
        self.btn_run_test = QPushButton("▶ Generuj i odsłuchaj test")
        self.btn_run_test.setObjectName("primaryBtn")
        self.btn_run_test.clicked.connect(self._on_run_test_clicked)
        btn_stop_snd = QPushButton("⏹ Zatrzymaj odsłuch")
        btn_stop_snd.clicked.connect(self.stop_wav_requested.emit)
        self.lbl_test_status = QLabel("")
        test_btns.addWidget(self.btn_run_test)
        test_btns.addWidget(btn_stop_snd)
        test_btns.addWidget(self.lbl_test_status, 1)
        test_lay.addLayout(test_btns)
        layout.addWidget(test_group)

        # Rejestracja kontekstowej pomocy
        self._register_help(
            lbl_ref_wav,
            "Plik próbki głosu (6–10 s WAV)",
            "Krótki, czysty wycinek głosu lektora (optymalnie 6–10 s). Model klonuje z niego barwę, "
            "tembr i naturalny rytm mowy.",
        )
        self._register_help(
            ref_row_w,
            "Plik próbki głosu (6–10 s WAV)",
            "Krótki, czysty wycinek głosu lektora (optymalnie 6–10 s). Model klonuje z niego barwę, "
            "tembr i naturalny rytm mowy.",
        )
        self._register_help(
            lbl_ref_txt,
            "Transkrypcja próbki referencyjnej",
            "Dokładny, słowo w słowo zapis tekstu wypowiadanego w pliku WAV (wraz z interpunkcją). "
            "Pozwala modelowi idealnie dopasować brzmienie głosu do sylab.",
        )
        self._register_help(
            ref_txt_w,
            "Transkrypcja próbki referencyjnej",
            "Dokładny, słowo w słowo zapis tekstu wypowiadanego w pliku WAV (wraz z interpunkcją).",
        )
        self._register_help(
            cut_wrap,
            "Wycinarka próbki (FFmpeg)",
            "Pomocnicze narzędzie do wycięcia czystego 6–10 s fragmentu z audiobooka (MP3/M4B/WAV) "
            "do formatu 24 kHz mono WAV.",
        )
        self._register_help(
            test_group,
            f"Szybki test 1 zdania ({self.role_label})",
            "Generuje pojedyncze zdanie w kilka sekund głosem z tej podzakładki i od razu je odtwarza.",
        )
        self._register_help(
            self.btn_reset_voice_defaults,
            "Przycisk „Domyślne”",
            f"Przywraca wypracowane domyślne parametry syntezy dla roli {self.role_label}.",
        )

        self._editable_widgets = [
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

    def set_edit_enabled(self, enabled: bool) -> None:
        for w in self._editable_widgets:
            w.setEnabled(enabled)

    def set_profiles(self, profiles: list[VoiceProfile], select_name: str | None = None) -> None:
        target = select_name or self._defaults["default_profile"]
        self.combo_voices.blockSignals(True)
        self.combo_voices.clear()
        sel_idx = 0
        for i, p in enumerate(profiles):
            self.combo_voices.addItem(f"{p.display_name} ({p.name})", p)
            if p.name == target:
                sel_idx = i
        self.combo_voices.blockSignals(False)
        if profiles:
            self.combo_voices.setCurrentIndex(sel_idx)
            self._populate_fields(profiles[sel_idx])

    def _populate_fields(self, p: VoiceProfile) -> None:
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
            self._populate_fields(p)
            self.role_profile_assigned.emit(self.role_key, p.name)

    def current_profile(self) -> VoiceProfile | None:
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
        name, ok = QInputDialog.getText(
            self, f"Nowy profil ({self.role_label})", "Podaj nazwę nowego lektora:"
        )
        if not ok or not name.strip():
            return
        slug = slugify_voice_name(name)
        profile = VoiceProfile(
            name=slug,
            display_name=name.strip(),
            ref_audio="ref.wav",
            ref_text="",
            speed=float(self._defaults["speed"]),
            cfg_strength=float(self._defaults["cfg"]),
            nfe_step=int(self._defaults["nfe"]),
            seed=int(self._defaults["seed"]),
        )
        save_voice_profile(profile, VOICES_DIR)
        self.profiles_changed.emit(self.role_key, slug)

    def _on_save_voice_profile(self) -> None:
        profile = self.current_profile()
        if profile is None:
            return
        save_voice_profile(profile, VOICES_DIR)
        self.profiles_changed.emit(self.role_key, profile.name)
        QMessageBox.information(self, "Zapisano", f"Zapisano profil lektora: {profile.display_name}")

    def _on_reset_voice_defaults(self) -> None:
        profile = self.current_profile()
        if profile is None:
            return
        d = self._defaults
        reply = QMessageBox.question(
            self,
            "Przywrócić domyślne?",
            f"Przywrócić domyślne parametry syntezy dla roli „{self.role_label}” "
            f"(speed={d['speed']:.2f}, CFG={d['cfg']:.2f}, ODE={d['nfe']}, Seed={d['seed']})"
            + (" oraz oryginalną próbkę ref_gosztyla.wav?" if profile.name == "moj_lektor" else "?"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.spn_speed.setValue(float(d["speed"]))
        self.spn_cfg.setValue(float(d["cfg"]))
        self.spn_nfe.setValue(int(d["nfe"]))
        self.spn_seed.setValue(int(d["seed"]))
        if profile.name == "moj_lektor":
            self.ed_voice_display.setText("Krzysztof Gosztyła (domyślny)")
            self.ed_ref_audio.setText("ref_gosztyla.wav")
            self.ed_ref_text.setPlainText(
                "A niektóre nawet jeszcze się powiększyły, jak to zwykle bywa, "
                "pojawili się też nowi milionerzy żerujący na czasach kryzysu."
            )
        updated = self.current_profile()
        if updated is not None:
            save_voice_profile(updated, VOICES_DIR)
            self.profiles_changed.emit(self.role_key, updated.name)
        QMessageBox.information(self, "Przywrócono", "Przywrócono i zapisano domyślne ustawienia lektora.")

    def _on_browse_ref_wav(self) -> None:
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Wybierz plik WAV próbki głosu", str(VOICES_DIR), "Pliki audio (*.wav)"
        )
        if fpath:
            self.ed_ref_audio.setText(fpath)

    def _on_play_ref_wav(self) -> None:
        profile = self.current_profile()
        if profile and profile.ref_audio_path.exists():
            self.play_wav_requested.emit(profile.ref_audio_path)
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
        profile = self.current_profile()
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
            updated = self.current_profile()
            if updated is not None:
                save_voice_profile(updated, VOICES_DIR)
            self.play_wav_requested.emit(out_wav)
            QMessageBox.information(
                self,
                "Wycięto próbkę",
                f"Zapisano {out_wav.name} i uruchomiono odsłuch.\n"
                "Upewnij się, że wpisałeś dokładną transkrypcję w polu powyżej i zapisz profil!",
            )
        except Exception as e:
            QMessageBox.critical(self, "Błąd FFmpeg", str(e))

    def _on_run_test_clicked(self) -> None:
        profile = self.current_profile()
        raw_text = self.ed_test_sentence.text().strip()
        if not raw_text:
            return
        self.run_test_requested.emit(self.role_key, profile, raw_text)
