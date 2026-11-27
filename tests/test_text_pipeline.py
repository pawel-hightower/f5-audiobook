"""Testy jednostkowe normalizatora polskiego, chunkera, parsera TXT/EPUB i bazy SQLite."""

from pathlib import Path

from f5_audiobook.config import DEFAULT_PRONUNCIATION
from f5_audiobook.pipeline.project_db import ProjectDB
from f5_audiobook.text.chunker import PAUSE_HEADING_MS, PAUSE_PARAGRAPH_MS, chunk_paragraphs
from f5_audiobook.text.epub_parser import BookData, ChapterData, parse_txt
from f5_audiobook.text.normalizer_pl import normalize_polish_text


def test_polish_normalizer() -> None:
    s = "Rozdział IV: W 1989 r. o godz. 14:30 było ok. 50% ludzi, np. prof. Kowalski na ul. Długiej 12."
    out = normalize_polish_text(s, DEFAULT_PRONUNCIATION)
    assert "czwarty" in out.lower()
    assert "w tysiąc dziewięćset osiemdziesiątym dziewiątym roku" in out.lower()
    assert "o godzinie czternastej trzydzieści" in out.lower()
    assert "pięćdziesiąt procent" in out.lower()
    assert "na przykład" in out.lower()
    assert "profesor" in out.lower()
    assert "ulica" in out.lower()
    assert "dwanaście" in out.lower()

    # Przypadki z notki o autorze (Graham Masterton)
    m1 = normalize_polish_text("Graham Masterton urodził się w 1946 roku w Edynburgu.")
    assert "w tysiąc dziewięćset czterdziestym szóstym roku" in m1.lower(), m1

    m2 = normalize_polish_text("Swój pierwszy horror Manitou wydał na początku lat 70. XX wieku.")
    assert "lat siedemdziesiątych dwudziestego wieku." in m2.lower(), m2
    assert ". dwudziestego" not in m2.lower(), m2  # brak rozbicia zdania kropką!

    m3 = normalize_polish_text("Wydarzyło się to 15 maja 2026 roku, a także w XX wieku i od 1980 roku.")
    assert "piętnastego maja dwa tysiące dwudziestego szóstego roku" in m3.lower(), m3
    assert "w dwudziestym wieku" in m3.lower(), m3
    assert "od tysiąc dziewięćset osiemdziesiątego roku" in m3.lower(), m3

    s2 = "Słuchał audiobooka na karcie NVIDIA."
    out2 = normalize_polish_text(s2, DEFAULT_PRONUNCIATION)
    assert "audiobuka" in out2
    assert "Enwidia" in out2

    # Dialogi z kropką przed myślnikiem nie powinny zostawiać ".,"
    d1 = normalize_polish_text("— Nie wolno ci — powiedział porucznik Bartosz. — Wiatr znowu wieje.")
    assert ".," not in d1, d1
    assert "Bartosz. Wiatr" in d1, d1
    assert "Nie wolno ci. Powiedział porucznik Bartosz." in d1, d1

    d2 = normalize_polish_text("— Nie wolno ci ponownie schodzić pod wodę, Kowalska — powiedział porucznik Bartosz.")
    assert "Kowalska. Powiedział" in d2, d2


def test_chunker_merges_short_and_splits_long() -> None:
    paras = [
        "Rozdział 1",
        "— Tak. — Odparł cicho.",
        "Wieczór zapadał nad miastem powoli, a chłodny wiatr niósł zapach zbliżającego się deszczu, kiedy zatrzymał się na skraju pustej ulicy i zaczął nasłuchiwać odległego echa kroków.",
    ]
    chunks = chunk_paragraphs(paras, DEFAULT_PRONUNCIATION, max_chars=110, chapter_title="Rozdział 1")
    assert chunks[0].pause_after_ms == PAUSE_HEADING_MS
    assert "rozdział pierwszy." in chunks[0].normalized_text.lower(), chunks[0].normalized_text
    assert chunks[-1].pause_after_ms == PAUSE_PARAGRAPH_MS
    for c in chunks[1:]:
        assert len(c.normalized_text) <= 130, f"Za długi chunk: {c.normalized_text}"

    # Goły numer rozdziału ("3") powinien rozwinąć się do "Rozdział trzeci."
    c_num = chunk_paragraphs(["Pierwsze zdanie."], DEFAULT_PRONUNCIATION, chapter_title="3")
    assert c_num[0].normalized_text == "Rozdział trzeci.", c_num[0].normalized_text

    # "Część 2 – Właściwa miara" -> "Część druga, Właściwa miara."
    c_part = chunk_paragraphs(["Część 2 – Właściwa miara"], DEFAULT_PRONUNCIATION, chapter_title="Część 2 – Właściwa miara")
    assert c_part[0].normalized_text == "Część druga, Właściwa miara.", c_part[0].normalized_text


def test_project_db_roundtrip(tmp_path: Path) -> None:
    book = BookData(
        source_path=tmp_path / "test.txt",
        title="Testowa Książka",
        author="Jan Testowy",
        chapters=[
            ChapterData(index=0, title="Prolog", paragraphs=["To jest pierwsze zdanie testowe."], selected=True),
            ChapterData(index=1, title="Spis treści", paragraphs=["Prolog 1"], selected=False),
        ],
    )
    db = ProjectDB.create_from_book(book, DEFAULT_PRONUNCIATION, work_root=tmp_path)
    chapters = db.list_chapters()
    assert len(chapters) == 2
    assert chapters[0].selected is True
    assert chapters[1].selected is False

    pending = db.get_pending_chunks()
    assert len(pending) >= 1
    assert all(c.chapter_title == "Prolog" for c in pending)


def test_parse_txt(tmp_path: Path) -> None:
    f = tmp_path / "mini.txt"
    f.write_text(
        "Rozdział I\n\nPierwszy akapit książki.\n\nRozdział II\n\nDrugi akapit książki.\n",
        encoding="utf-8",
    )
    book = parse_txt(f)
    assert len(book.chapters) == 2
    assert book.chapters[0].title == "Rozdział I"
    assert book.chapters[1].title == "Rozdział II"


def test_short_carrier_slicing_helpers() -> None:
    import numpy as np
    from f5_audiobook.engine.quality_guard import _basic_audio_check, _find_acoustic_pause_cut, _find_quietest_sample

    sr = 24000
    # Symulacja: 1.2s nośnika (ton) + 0.35s ciszy + 0.45s krótkiej frazy ("Tak.")
    t1 = np.linspace(0, 1.2, int(sr * 1.2), dtype=np.float32)
    t2 = np.linspace(0, 0.45, int(sr * 0.45), dtype=np.float32)
    carrier = 0.2 * np.sin(2 * np.pi * 220 * t1)
    silence = np.zeros(int(sr * 0.35), dtype=np.float32)
    target = 0.2 * np.sin(2 * np.pi * 220 * t2)
    wav = np.concatenate([carrier, silence, target])

    q_idx = _find_quietest_sample(wav, sr, 1.15, 1.55)
    assert int(sr * 1.20) <= q_idx <= int(sr * 1.55), q_idx

    ac_cut = _find_acoustic_pause_cut(wav, sr, expected_ratio=0.72)
    assert ac_cut is not None and int(sr * 1.20) <= ac_cut <= int(sr * 1.55), ac_cut

    # Krótkie 0.22s "Tak." nie może być błędnie odrzucane przez _basic_audio_check
    short_ok = 0.15 * np.sin(2 * np.pi * 220 * np.linspace(0, 0.22, int(sr * 0.22), dtype=np.float32))
    assert _basic_audio_check("Tak.", short_ok, sr) is None

    # Artefakt spacji Whispera ("po wiedział" vs "powiedział") ma nospace_sim == 1.0
    from difflib import SequenceMatcher
    assert SequenceMatcher(None, "powiedział", "powiedział").ratio() >= 0.95
    assert SequenceMatcher(None, "silos", "silusz").ratio() < 0.80


def test_multi_voice_chunking_and_db_migration(tmp_path: Path) -> None:
    import sqlite3
    from f5_audiobook.text.chunker import PAUSE_DIALOGUE_SWITCH_MS

    paras = [
        "Rozdział 1",
        "Holston spojrzał przez okno na szare wzgórza.",
        "— Nie jestem pewien — powiedział Holston. — Może jutro wrócimy.",
        "— Ja idę teraz! — krzyknęła Jahns.",
        "— Zaczekaj chwilę.",
    ]
    chunks = chunk_paragraphs(paras, DEFAULT_PRONUNCIATION, chapter_title="Rozdział 1", multi_voice=True)
    # Nagłówek + narracja + (dialog M, didaskalia N, dialog M) + (dialog K, didaskalia N) + (dialog ?)
    assert chunks[0].voice_role == "narrator"
    assert chunks[1].voice_role == "narrator"
    roles = [c.voice_role for c in chunks[2:]]
    assert roles == ["male", "narrator", "male", "female", "narrator", "dialogue"], roles
    # Między dialogiem a didaskaliami krótka pauza 220ms
    assert chunks[2].pause_after_ms == PAUSE_DIALOGUE_SWITCH_MS
    assert chunks[3].pause_after_ms == PAUSE_DIALOGUE_SWITCH_MS
    assert chunks[5].pause_after_ms == PAUSE_DIALOGUE_SWITCH_MS
    assert chunks[2].normalized_text.endswith("."), chunks[2].normalized_text

    # Sprawdź roundtrip multi_voice w ProjectDB + bulk update
    book = BookData(
        source_path=tmp_path / "mv.txt",
        title="Mini Superprodukcja",
        author="Autor",
        chapters=[ChapterData(index=0, title="Rozdział 1", paragraphs=paras, selected=True)],
    )
    db = ProjectDB.create_from_book(book, DEFAULT_PRONUNCIATION, work_root=tmp_path, multi_voice=True)
    assert db.is_multi_voice() is True
    db_chunks = db.list_chunks()
    assert db_chunks[2].voice_role == "male"
    assert db_chunks[5].voice_role == "female"
    assert db_chunks[-1].voice_role == "dialogue"

    changed = db.set_chunks_voice_roles_bulk({db_chunks[-1].id: "female"})
    assert changed == 1
    assert db.get_chunk(db_chunks[-1].id).voice_role == "female"

    # Symulacja starej bazy SQLite bez kolumny voice_role -> automatyczna migracja
    legacy_dir = tmp_path / "LegacyBook"
    legacy_dir.mkdir()
    conn = sqlite3.connect(str(legacy_dir / "project.db"))
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE chapters (id INTEGER PRIMARY KEY AUTOINCREMENT, chapter_index INTEGER NOT NULL UNIQUE, title TEXT NOT NULL, selected INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE chunks (id INTEGER PRIMARY KEY AUTOINCREMENT, chapter_id INTEGER NOT NULL, chunk_index INTEGER NOT NULL, raw_text TEXT NOT NULL, normalized_text TEXT NOT NULL, pause_after_ms INTEGER NOT NULL DEFAULT 380, status TEXT NOT NULL DEFAULT 'pending', audio_path TEXT, duration_sec REAL NOT NULL DEFAULT 0.0, seed INTEGER NOT NULL DEFAULT 101, qg_score REAL, qg_note TEXT NOT NULL DEFAULT '', heard_text TEXT NOT NULL DEFAULT '', error_msg TEXT NOT NULL DEFAULT '');
        INSERT INTO chapters(chapter_index, title, selected) VALUES (0, 'Ch1', 1);
        INSERT INTO chunks(chapter_id, chunk_index, raw_text, normalized_text) VALUES (1, 0, 'Test.', 'Test.');
        """
    )
    conn.commit()
    conn.close()

    migrated = ProjectDB(legacy_dir)
    assert migrated.is_multi_voice() is False
    m_chunks = migrated.list_chunks()
    assert len(m_chunks) == 1 and m_chunks[0].voice_role == "narrator"

    # Test konwersji tylko zaznaczonych nietkniętych rozdziałów w istniejącym 1-głosowym projekcie
    txt_file = tmp_path / "book_convert.txt"
    txt_file.write_text(
        "Rozdział 1\n\n— Cześć — powiedział Jan.\n\nRozdział 2\n\n— Idziemy? — zapytała Anna.\n\n— Tak — odparł Jan.\n",
        encoding="utf-8",
    )
    b2 = parse_txt(txt_file)
    db2 = ProjectDB.create_from_book(b2, DEFAULT_PRONUNCIATION, work_root=tmp_path, multi_voice=False)
    chs = db2.list_chapters()
    assert all(not c.multi_voice for c in chs)
    # Oznaczamy rozdział 1 jako wygenerowany (done), zaznaczamy oba i konwertujemy -> tylko rozdział 2 ma się przeliczyć!
    ch1_chunks = db2.list_chunks(chapter_id=chs[0].id)
    db2.mark_chunk_done(ch1_chunks[0].id, tmp_path / "fake.wav", 1.5, 101)
    res = db2.rechunk_selected_chapters_multi_voice(DEFAULT_PRONUNCIATION, skip_with_done_audio=True)
    assert res.converted_chapters == 1
    assert res.skipped_done_chapters == 1
    chs_after = db2.list_chapters()
    assert chs_after[0].multi_voice is False
    assert chs_after[1].multi_voice is True
    assert chs_after[1].dialogue_tagged >= 2


def test_player_auto_rewind_logic() -> None:
    from f5_audiobook.gui.player_widget import (
        AUTO_REWIND_LONG_MS,
        AUTO_REWIND_SHORT_MS,
        compute_auto_rewind_ms,
    )

    now = 1_700_000_000.0
    assert compute_auto_rewind_ms(None, now) == 0
    assert compute_auto_rewind_ms(now - 5.0, now) == AUTO_REWIND_SHORT_MS
    assert compute_auto_rewind_ms(now - 600.0, now) == AUTO_REWIND_SHORT_MS
    assert compute_auto_rewind_ms(now - 3600.0, now) == AUTO_REWIND_SHORT_MS
    assert compute_auto_rewind_ms(now - 3601.0, now) == AUTO_REWIND_LONG_MS
    assert compute_auto_rewind_ms(now - 86400.0, now) == AUTO_REWIND_LONG_MS
    assert compute_auto_rewind_ms(0.0, now) == AUTO_REWIND_LONG_MS


def test_length_bucketed_batching() -> None:
    from f5_audiobook.pipeline.batch_runner import build_length_buckets
    from f5_audiobook.pipeline.project_db import ChunkRow
    from f5_audiobook.voices import VoiceProfile

    v_n = VoiceProfile(name="moj_lektor", display_name="N", ref_audio="a.wav", ref_text="x")
    v_m = VoiceProfile(name="kosior", display_name="M", ref_audio="b.wav", ref_text="x")
    v_map = {"narrator": v_n, "male": v_m}

    def _mk(cid: int, ch_i: int, role: str, txt: str) -> ChunkRow:
        return ChunkRow(
            id=cid, chapter_id=ch_i + 1, chapter_index=ch_i, chapter_title=f"Ch{ch_i}",
            chunk_index=cid, raw_text=txt, normalized_text=txt, pause_after_ms=600,
            status="pending", audio_path=None, duration_sec=0.0, seed=101, qg_score=None,
            qg_note="", heard_text="", error_msg="", voice_role=role,
        )

    chunks = [
        _mk(1, 0, "narrator", "Krótko."),                                             # < 22 znaków -> zawsze single!
        _mk(2, 0, "narrator", "To jest długie zdanie narratora w rozdziale pierwszym o średniej długości."),
        _mk(3, 0, "male", "To jest średniej długości kwestia dialogowa Kosiora w pierwszym rozdziale."),
        _mk(4, 14, "narrator", "To jest drugie zdanie narratora z piętnastego rozdziału o zbliżonej mierze."),
        _mk(5, 14, "male", "To jest niemal identycznej długości kwestia dialogowa Kosiora z rozdziału."),
        _mk(6, 5, "narrator", "A to trzecie zdanie narratora o prawie dokładnie tej samej długości tekstu."),
        _mk(7, 8, "narrator", "Krótsze zdanie narratora bez bliźniaka o innej długości."),                 # ~56 zn. -> musi iść osobno!
    ]
    buckets = build_length_buckets(chunks, lambda c: v_map[c.voice_role], batch_size=3)
    # 3 narrator (~75 zn.) -> 1 trojaczek; 2 male (~74 zn.) -> 1 para; 1 krótki + 1 samotny ~56 zn. -> 2 osobne!
    sizes = sorted(len(b) for _, b in buckets)
    assert sizes == [1, 1, 2, 3], sizes
    # Każdy batch musi mieć spójny głos
    for v, grp in buckets:
        assert all(v_map[c.voice_role].name == v.name for c in grp)
    # Krótki chunk (#1) oraz samotny średni (#7) muszą iść osobno (batch=1)
    assert any(len(b) == 1 and b[0].id == 1 for _, b in buckets)
    assert any(len(b) == 1 and b[0].id == 7 for _, b in buckets)


if __name__ == "__main__":
    import tempfile

    test_polish_normalizer()
    test_chunker_merges_short_and_splits_long()
    test_short_carrier_slicing_helpers()
    test_player_auto_rewind_logic()
    test_length_bucketed_batching()
    with tempfile.TemporaryDirectory() as td:
        test_project_db_roundtrip(Path(td))
        test_parse_txt(Path(td))
        test_multi_voice_chunking_and_db_migration(Path(td))
    print("Wszystkie testy jednostkowe przeszły pomyślnie!")
