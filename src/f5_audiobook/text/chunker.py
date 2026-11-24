"""Inteligentny podział polskich akapitów na chunki optymalne dla F5-TTS IPA (60-115 znaków)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from f5_audiobook.text.normalizer_pl import normalize_polish_text

# Domyślne pauzy w milisekundach po chunku (dopasowane pod spokojnego lektora audiobooków)
PAUSE_CLAUSE_MS = 180      # podział długiego zdania w środku (po przecinku/średniku)
PAUSE_SENTENCE_MS = 600    # wyraźna, spokojna pauza między zdaniami
PAUSE_PARAGRAPH_MS = 1200  # wyraźny oddech między akapitami
PAUSE_HEADING_MS = 1600    # po tytule rozdziału

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:—])\s+")


@dataclass
class TextChunk:
    index: int
    raw_text: str
    normalized_text: str
    pause_after_ms: int
    is_paragraph_end: bool = False


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Dzieli długie zdanie na porcje <= max_chars w naturalnych miejscach (przecinki, spójniki)."""
    if len(sentence) <= max_chars:
        return [sentence]

    clauses = _CLAUSE_SPLIT_RE.split(sentence)
    parts: list[str] = []
    buf = ""

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if not buf:
            buf = clause
        elif len(buf) + 1 + len(clause) <= max_chars:
            buf = f"{buf} {clause}"
        else:
            parts.append(buf)
            buf = clause

    if buf:
        parts.append(buf)

    # Jeśli jakaś pojedyncza klauzula nadal przekracza max_chars, dzielimy po spacji blisko połowy
    final_parts: list[str] = []
    for p in parts:
        while len(p) > max_chars:
            cut = p.rfind(" ", int(max_chars * 0.45), max_chars + 1)
            if cut == -1:
                cut = p.rfind(" ", 0, max_chars + 1)
            if cut == -1:
                cut = max_chars
            final_parts.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            final_parts.append(p)

    # Nie zostawiaj mikro-ogona (< 22 znaki) jako osobnego chunka — doklej do poprzedniego
    if len(final_parts) >= 2 and len(final_parts[-1]) < 22:
        final_parts[-2] = f"{final_parts[-2]} {final_parts[-1]}"
        final_parts.pop()

    return final_parts


def chunk_paragraphs(
    paragraphs: list[str],
    pronunciation_dict: dict[str, str] | None = None,
    min_chars: int = 45,
    max_chars: int = 165,
    chapter_title: str | None = None,
    include_title: bool = True,
) -> list[TextChunk]:
    """Zamienia listę akapitów rozdziału na znormalizowane chunki z dopasowanymi pauzami."""
    chunks: list[TextChunk] = []

    work_paragraphs = list(paragraphs)
    if include_title and chapter_title:
        # Jeśli pierwszy akapit nie jest już tym samym tytułem, dodajemy go na początek
        if not work_paragraphs or work_paragraphs[0].strip().lower() != chapter_title.strip().lower():
            work_paragraphs.insert(0, chapter_title)

    for p_idx, para in enumerate(work_paragraphs):
        raw_para = para.strip()
        if not raw_para:
            continue

        norm_para = normalize_polish_text(raw_para, pronunciation_dict)
        if not norm_para:
            continue

        is_heading = include_title and p_idx == 0 and len(norm_para) <= 90

        # Jeśli nagłówek rozdziału to sama goła liczba ("1", "2", "IV"), rozwijamy go do "Rozdział pierwszy/drugi/czwarty.",
        # dzięki czemu brzmi naturalnie jak prawdziwy audiobook i nie jest za krótki dla F5-TTS!
        if is_heading and re.fullmatch(r"(?:\d{1,3}|[IVXL]{1,6})\.?", raw_para, re.IGNORECASE):
            norm_para = normalize_polish_text(f"Rozdział {raw_para.rstrip('.')}.", pronunciation_dict)

        # Tytuł rozdziału zawsze w osobnym chunku z dłuższą pauzą
        if is_heading:
            if not re.search(r"[.!?]$", norm_para):
                norm_para = norm_para + "."
            chunks.append(
                TextChunk(
                    index=len(chunks),
                    raw_text=raw_para,
                    normalized_text=norm_para,
                    pause_after_ms=PAUSE_HEADING_MS,
                    is_paragraph_end=True,
                )
            )
            continue

        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(norm_para) if s.strip()]
        if not sentences:
            continue

        # Łączymy bardzo krótkie sąsiednie zdania (np. "Tak." + "Odparł cicho.")
        # żeby F5-TTS nie halucynował na jednosłownych chunkach
        merged_units: list[tuple[str, bool]] = []  # (tekst, czy_koniec_zdania)
        acc = ""
        for sent in sentences:
            if not acc:
                acc = sent
            elif len(acc) < min_chars and len(acc) + 1 + len(sent) <= max_chars:
                acc = f"{acc} {sent}"
            else:
                merged_units.append((acc, True))
                acc = sent
        if acc:
            if merged_units and len(acc) < min_chars and len(merged_units[-1][0]) + 1 + len(acc) <= max_chars + 15:
                prev_txt, _ = merged_units[-1]
                merged_units[-1] = (f"{prev_txt} {acc}", True)
            else:
                merged_units.append((acc, True))

        # Dzielimy zbyt długie zdania w naturalnych miejscach
        para_pieces: list[tuple[str, int]] = []
        for unit_idx, (unit, _) in enumerate(merged_units):
            sub_parts = _split_long_sentence(unit, max_chars=max_chars)
            is_last_unit = unit_idx == len(merged_units) - 1
            for sp_idx, sp in enumerate(sub_parts):
                is_last_sp = sp_idx == len(sub_parts) - 1
                if is_last_unit and is_last_sp:
                    pause = PAUSE_PARAGRAPH_MS
                elif is_last_sp:
                    pause = PAUSE_SENTENCE_MS
                else:
                    pause = PAUSE_CLAUSE_MS
                para_pieces.append((sp, pause))

        for piece_idx, (piece_text, pause_ms) in enumerate(para_pieces):
            chunks.append(
                TextChunk(
                    index=len(chunks),
                    raw_text=raw_para if len(para_pieces) == 1 else piece_text,
                    normalized_text=piece_text,
                    pause_after_ms=pause_ms,
                    is_paragraph_end=(piece_idx == len(para_pieces) - 1),
                )
            )

    return chunks
