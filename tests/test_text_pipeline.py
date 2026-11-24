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


if __name__ == "__main__":
    import tempfile

    test_polish_normalizer()
    test_chunker_merges_short_and_splits_long()
    with tempfile.TemporaryDirectory() as td:
        test_project_db_roundtrip(Path(td))
        test_parse_txt(Path(td))
    print("Wszystkie testy jednostkowe przeszły pomyślnie!")
