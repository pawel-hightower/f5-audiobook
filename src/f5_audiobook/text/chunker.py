"""Inteligentny podział polskich akapitów na chunki optymalne dla F5-TTS IPA (60-115 znaków)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from f5_audiobook.text.normalizer_pl import normalize_polish_text

# Domyślne pauzy w milisekundach po chunku (dopasowane pod spokojnego lektora audiobooków)
PAUSE_CLAUSE_MS = 180              # podział długiego zdania w środku (po przecinku/średniku)
PAUSE_DIALOGUE_SWITCH_MS = 220     # płynna zmiana głosu między dialogiem a didaskaliami w tym samym akapicie
PAUSE_SENTENCE_MS = 600            # wyraźna, spokojna pauza między zdaniami
PAUSE_DIALOGUE_PARAGRAPH_MS = 700  # szybsza wymiana replik między akapitami dialogu (mini-superprodukcja)
PAUSE_PARAGRAPH_MS = 1200          # wyraźny oddech między akapitami narracji
PAUSE_HEADING_MS = 1600            # po tytule rozdziału

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:—])\s+")
_DIALOGUE_LEAD_RE = re.compile(r"^(?:[—–]\s*|-\s+)")
_DIALOGUE_MID_SPLIT_RE = re.compile(r"\s+[—–]\s+|(?<=[.!?…,:;])\s*[—–]\s+")


@dataclass
class TextChunk:
    index: int
    raw_text: str
    normalized_text: str
    pause_after_ms: int
    is_paragraph_end: bool = False
    voice_role: str = "narrator"  # narrator | male | female | dialogue


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


def _split_dialogue_paragraph(raw_para: str) -> list[tuple[str, str]]:
    """Dzieli polski akapit dialogowy (rozpoczynający się od — / – / - ) na naprzemienne segmenty:
    ('dialogue', tekst_kwestii) i ('narrator', tekst_didaskaliów).

    Dla akapitów czysto narracyjnych zwraca [('narrator', raw_para)].
    """
    stripped = raw_para.strip()
    m = _DIALOGUE_LEAD_RE.match(stripped)
    if not m:
        return [("narrator", stripped)]

    body = stripped[m.end():].strip()
    if not body:
        return []

    raw_parts = [p.strip() for p in _DIALOGUE_MID_SPLIT_RE.split(body) if p.strip()]
    if not raw_parts:
        return []

    segments: list[tuple[str, str]] = []
    for i, part in enumerate(raw_parts):
        role = "dialogue" if (i % 2 == 0) else "narrator"
        # W polskiej typografii przed didaskaliami pomija się kropkę ("— Tak — powiedział.").
        # Dopisujemy kropkę (lub przecinek przed kontynuacją małą literą), żeby F5-TTS miał naturalną kadencję.
        if role == "dialogue" and part and part[-1] not in ".!?…,:;":
            next_dialogue = raw_parts[i + 2] if (i + 2) < len(raw_parts) else ""
            if next_dialogue and next_dialogue[0].islower():
                part = part + ","
            else:
                part = part + "."
        segments.append((role, part))
    return segments


def _chunk_single_segment(
    norm_text: str,
    min_chars: int,
    max_chars: int,
    end_pause_ms: int,
) -> list[tuple[str, int]]:
    """Dzieli pojedynczy znormalizowany segment (narracji lub dialogu) na porcje (tekst, pauza_ms)."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(norm_text) if s.strip()]
    if not sentences:
        return []

    merged_units: list[tuple[str, bool]] = []
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

    pieces: list[tuple[str, int]] = []
    for unit_idx, (unit, _) in enumerate(merged_units):
        sub_parts = _split_long_sentence(unit, max_chars=max_chars)
        is_last_unit = unit_idx == len(merged_units) - 1
        for sp_idx, sp in enumerate(sub_parts):
            is_last_sp = sp_idx == len(sub_parts) - 1
            if is_last_unit and is_last_sp:
                pause = end_pause_ms
            elif is_last_sp:
                pause = PAUSE_SENTENCE_MS
            else:
                pause = PAUSE_CLAUSE_MS
            pieces.append((sp, pause))
    return pieces


def chunk_paragraphs(
    paragraphs: list[str],
    pronunciation_dict: dict[str, str] | None = None,
    min_chars: int = 45,
    max_chars: int = 165,
    chapter_title: str | None = None,
    include_title: bool = True,
    multi_voice: bool = False,
) -> list[TextChunk]:
    """Zamienia listę akapitów rozdziału na znormalizowane chunki z dopasowanymi pauzami.

    Gdy multi_voice=True, rozdziela akapity dialogowe (zaczynające się od półpauzy/pauzy)
    na osobne chunki kwestii dialogowych i didaskaliów narratora z krótszą pauzą przełączenia
    oraz wstępnym tagowaniem płci (polska heurystyka gramatyczna; pełne tagowanie robi Bielik AI).
    """
    chunks: list[TextChunk] = []

    work_paragraphs = list(paragraphs)
    if include_title and chapter_title:
        # Jeśli pierwszy akapit nie jest już tym samym tytułem, dodajemy go na początek
        if not work_paragraphs or work_paragraphs[0].strip().lower() != chapter_title.strip().lower():
            work_paragraphs.insert(0, chapter_title)

    # Leniwy import heurystyki (tylko w trybie 3 głosów)
    guess_gender = None
    if multi_voice:
        from f5_audiobook.engine.bielik_tagger import heuristic_guess_dialogue_gender
        guess_gender = heuristic_guess_dialogue_gender

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

        # Tytuł rozdziału zawsze w osobnym chunku z dłuższą pauzą (zawsze narrator)
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
                    voice_role="narrator",
                )
            )
            continue

        if not multi_voice:
            para_pieces = _chunk_single_segment(norm_para, min_chars, max_chars, PAUSE_PARAGRAPH_MS)
            for piece_idx, (piece_text, pause_ms) in enumerate(para_pieces):
                chunks.append(
                    TextChunk(
                        index=len(chunks),
                        raw_text=raw_para if len(para_pieces) == 1 else piece_text,
                        normalized_text=piece_text,
                        pause_after_ms=pause_ms,
                        is_paragraph_end=(piece_idx == len(para_pieces) - 1),
                        voice_role="narrator",
                    )
                )
            continue

        # Tryb 3 głosów (mini-superprodukcja): rozdzielamy akapit na segmenty dialogowe i narracyjne
        segments = _split_dialogue_paragraph(raw_para)
        if not segments:
            continue

        is_dialogue_para = any(r == "dialogue" for r, _ in segments)
        para_end_pause = PAUSE_DIALOGUE_PARAGRAPH_MS if is_dialogue_para else PAUSE_PARAGRAPH_MS

        # Wstępne odgadnięcie płci mówcy z didaskaliów tego akapitu (np. "— Tak — odparła.")
        inferred_para_gender: str | None = None
        if is_dialogue_para and guess_gender is not None:
            narr_bits = " ".join(txt for r, txt in segments if r == "narrator")
            dial_bits = " ".join(txt for r, txt in segments if r == "dialogue")
            # Poprzedni akapit narracyjny podpowiada mówcę tylko gdy kończy się dwukropkiem (np. "Anna zapytała:")
            prev_narr = (
                chunks[-1].raw_text
                if (chunks and chunks[-1].voice_role == "narrator" and chunks[-1].raw_text.rstrip().endswith(":"))
                else ""
            )
            inferred_para_gender = guess_gender(dial_bits, narr_bits, prev_narr)

        for seg_idx, (seg_role, seg_raw) in enumerate(segments):
            seg_norm = normalize_polish_text(seg_raw, pronunciation_dict)
            seg_norm = re.sub(r"^[—–\-\s]+|[—–\s]+$", "", seg_norm).strip()
            if not seg_norm:
                continue

            is_last_seg = seg_idx == len(segments) - 1
            seg_end_pause = para_end_pause if is_last_seg else PAUSE_DIALOGUE_SWITCH_MS
            eff_role = (inferred_para_gender or "dialogue") if seg_role == "dialogue" else "narrator"

            seg_pieces = _chunk_single_segment(seg_norm, min_chars, max_chars, seg_end_pause)
            for piece_idx, (piece_text, pause_ms) in enumerate(seg_pieces):
                is_para_end = is_last_seg and (piece_idx == len(seg_pieces) - 1)
                raw_display = f"— {seg_raw}" if (seg_role == "dialogue" and piece_idx == 0) else (
                    seg_raw if len(seg_pieces) == 1 else piece_text
                )
                chunks.append(
                    TextChunk(
                        index=len(chunks),
                        raw_text=raw_display,
                        normalized_text=piece_text,
                        pause_after_ms=pause_ms,
                        is_paragraph_end=is_para_end,
                        voice_role=eff_role,
                    )
                )

    return chunks
