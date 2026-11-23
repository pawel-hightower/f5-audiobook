"""Parser plików EPUB i TXT wyciągający metadane, okładkę oraz czystą listę rozdziałów."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Nagłówki sekcji, które zazwyczaj nie są treścią książki (domyślnie odznaczone w GUI)
_SKIP_TITLE_PATTERNS = re.compile(
    r"^(okładka|cover|strona tytułowa|title\s*page|spis treści|table of contents|"
    r"contents|toc|stopka|redakcj|copyright|prawa autorskie|kolofon|colophon|"
    r"podziękowania|acknowledg|bibliografia|indeks|index|przypisy|o autorze|"
    r"about the author|reklamy|inne książki|wydawnictwo)",
    re.IGNORECASE,
)

_TXT_CHAPTER_RE = re.compile(
    r"^(rozdział\s+[0-9ivxlcdm]+.*|część\s+[0-9ivxlcdm]+.*|prolog\b.*|epilog\b.*|"
    r"chapter\s+[0-9ivxlcdm]+.*|[ivxlcdm]{1,6}\.?\s*)$",
    re.IGNORECASE,
)


@dataclass
class ChapterData:
    index: int
    title: str
    paragraphs: list[str]
    selected: bool = True

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def char_count(self) -> int:
        return sum(len(p) for p in self.paragraphs)


@dataclass
class BookData:
    source_path: Path
    title: str
    author: str
    chapters: list[ChapterData] = field(default_factory=list)
    cover_bytes: bytes | None = None
    cover_ext: str = "jpg"


def _clean_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()


def _extract_paragraphs_from_html(html_bytes: bytes) -> tuple[str | None, list[str]]:
    soup = BeautifulSoup(html_bytes, "lxml")

    # Usuwamy skrypty, style, nawigację i indeksy przypisów (np. <sup>1</sup>)
    for tag in soup.find_all(["script", "style", "nav", "sup", "svg", "noscript"]):
        tag.decompose()

    # Usuwamy typowe linki do przypisów (noteref)
    for a in soup.find_all("a"):
        epub_type = str(a.get("epub:type", "") or a.get("role", ""))
        if "noteref" in epub_type.lower() or re.fullmatch(r"\[?\d+\]?", a.get_text(strip=True)):
            a.decompose()

    heading_title: str | None = None
    for htag in ("h1", "h2", "h3"):
        h = soup.find(htag)
        if h:
            htxt = _clean_whitespace(h.get_text(" ", strip=True))
            if htxt:
                heading_title = htxt
                break

    paragraphs: list[str] = []
    body = soup.body or soup
    block_tags = body.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "div"])

    # Zbieramy tylko bloki-liście (żeby nie duplikować zagnieżdżonych <div><p>...</p></div>)
    for el in block_tags:
        if el.name == "div" and el.find(["p", "h1", "h2", "h3", "h4", "div", "blockquote", "li"]):
            continue
        txt = _clean_whitespace(el.get_text(" ", strip=True))
        if not txt:
            continue
        # Pomijamy same numery stron (np. "142" albo "- 12 -")
        if re.fullmatch(r"[-–—\s]*\d{1,4}[-–—\s]*", txt):
            continue
        paragraphs.append(txt)

    # Fallback gdy dokument nie miał znaczników blokowych
    if not paragraphs:
        raw = _clean_whitespace(body.get_text("\n", strip=True))
        for line in raw.splitlines():
            line = _clean_whitespace(line)
            if line:
                paragraphs.append(line)

    # Deduplikacja sąsiednich identycznych nagłówków
    deduped: list[str] = []
    for p in paragraphs:
        if not deduped or deduped[-1] != p:
            deduped.append(p)

    return heading_title, deduped


def _flatten_toc(toc_items, mapping: dict[str, str]) -> None:
    from ebooklib import epub

    for item in toc_items:
        if isinstance(item, epub.Link):
            href = item.href.split("#")[0]
            if href and item.title and href not in mapping:
                mapping[href] = _clean_whitespace(item.title)
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            section, children = item[0], item[1]
            if hasattr(section, "href") and hasattr(section, "title"):
                href = section.href.split("#")[0]
                if href and section.title and href not in mapping:
                    mapping[href] = _clean_whitespace(section.title)
            if isinstance(children, (list, tuple)):
                _flatten_toc(children, mapping)


def _extract_cover(book) -> tuple[bytes | None, str]:
    import ebooklib

    # 1. Metadane cover
    try:
        cover_meta = book.get_metadata("OPF", "cover")
        if cover_meta:
            cover_id = cover_meta[0][1].get("content")
            if cover_id:
                item = book.get_item_with_id(cover_id)
                if item is not None:
                    ext = Path(item.get_name()).suffix.lstrip(".").lower() or "jpg"
                    return item.get_content(), ext
    except Exception:
        pass

    # 2. Item typu ITEM_COVER lub nazwa zawierająca "cover"
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_COVER:
            ext = Path(item.get_name()).suffix.lstrip(".").lower() or "jpg"
            return item.get_content(), ext
    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        name = item.get_name().lower()
        if "cover" in name or "okladk" in name or "okładk" in name:
            ext = Path(name).suffix.lstrip(".").lower() or "jpg"
            return item.get_content(), ext

    return None, "jpg"


def parse_epub(path: Path) -> BookData:
    import ebooklib
    from ebooklib import epub

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        book = epub.read_epub(str(path), options={"ignore_ncx": False})

    title_meta = book.get_metadata("DC", "title")
    title = _clean_whitespace(title_meta[0][0]) if title_meta else path.stem

    author_meta = book.get_metadata("DC", "creator")
    author = _clean_whitespace(author_meta[0][0]) if author_meta else "Nieznany autor"

    toc_titles: dict[str, str] = {}
    try:
        _flatten_toc(book.toc, toc_titles)
    except Exception:
        pass

    cover_bytes, cover_ext = _extract_cover(book)

    chapters: list[ChapterData] = []
    chap_idx = 0

    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        href = item.get_name()
        heading_title, paragraphs = _extract_paragraphs_from_html(item.get_content())
        if not paragraphs:
            continue

        total_chars = sum(len(p) for p in paragraphs)
        if total_chars < 15:
            continue

        # Dobór nazwy rozdziału: TOC -> nagłówek HTML -> domyślny numer
        base_href = Path(href).name
        ch_title = (
            toc_titles.get(href)
            or toc_titles.get(base_href)
            or heading_title
            or f"Sekcja {chap_idx + 1}"
        )

        # Czy sekcja wygląda na stronę techniczną (spis treści, okładka, stopka)?
        is_meta_section = bool(_SKIP_TITLE_PATTERNS.search(ch_title)) or (
            total_chars < 120 and chap_idx == 0
        )

        chapters.append(
            ChapterData(
                index=chap_idx,
                title=ch_title,
                paragraphs=paragraphs,
                selected=not is_meta_section,
            )
        )
        chap_idx += 1

    # Jeśli wszystkie zostały odznaczone (np. bardzo krótki testowy EPUB), zaznacz wszystkie
    if chapters and not any(c.selected for c in chapters):
        for c in chapters:
            c.selected = True

    return BookData(
        source_path=path,
        title=title,
        author=author,
        chapters=chapters,
        cover_bytes=cover_bytes,
        cover_ext=cover_ext,
    )


def parse_txt(path: Path) -> BookData:
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    blocks = [
        _clean_whitespace(b.replace("\n", " "))
        for b in re.split(r"\n\s*\n+", raw)
        if _clean_whitespace(b)
    ]

    chapters: list[ChapterData] = []
    current_title = path.stem
    current_paras: list[str] = []

    for block in blocks:
        if len(block) <= 80 and _TXT_CHAPTER_RE.match(block):
            if current_paras:
                chapters.append(
                    ChapterData(
                        index=len(chapters),
                        title=current_title,
                        paragraphs=current_paras,
                        selected=True,
                    )
                )
            current_title = block
            current_paras = [block]
        else:
            current_paras.append(block)

    if current_paras:
        chapters.append(
            ChapterData(
                index=len(chapters),
                title=current_title,
                paragraphs=current_paras,
                selected=True,
            )
        )

    return BookData(
        source_path=path,
        title=path.stem,
        author="Nieznany autor",
        chapters=chapters,
    )


def parse_book(path: Path) -> BookData:
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return parse_epub(path)
    if suffix == ".txt":
        return parse_txt(path)
    raise ValueError(f"Nieobsługiwany format pliku: {suffix} (obsługiwane: .epub, .txt)")
