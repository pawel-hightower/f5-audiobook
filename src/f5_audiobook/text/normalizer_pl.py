"""Normalizator tekstu polskiego pod syntezę mowy (skróty, liczby, daty, lata, dekady, wieki, godziny, słownik wymowy)."""

from __future__ import annotations

import re
from num2words import num2words

# Popularne polskie skróty (z kropką i bez kropki)
_ABBREVIATIONS: list[tuple[str, str]] = [
    (r"\bm\.in\.", "między innymi"),
    (r"\btzn\.", "to znaczy"),
    (r"\btj\.", "to jest"),
    (r"\btzw\.", "tak zwany"),
    (r"\bnp\.", "na przykład"),
    (r"\bitd\.", "i tak dalej"),
    (r"\bitp\.", "i tym podobne"),
    (r"\bcdn\.", "ciąg dalszy nastąpi"),
    (r"\bds\.", "do spraw"),
    (r"\bws\.", "w sprawie"),
    (r"\bpt\.", "pod tytułem"),
    (r"\bvs\.", "versus"),
    (r"\bwg\.?(?=\s)", "według"),
    (r"\bok\.", "około"),
    (r"\bo\s+godz\.", "o godzinie"),
    (r"\bdo\s+godz\.", "do godziny"),
    (r"\bod\s+godz\.", "od godziny"),
    (r"\bpo\s+godz\.", "po godzinie"),
    (r"\bprzed\s+godz\.", "przed godziną"),
    (r"\bgodz\.", "godzina"),
    (r"\bmin\.", "minut"),
    (r"\bsek\.", "sekund"),
    (r"\btys\.", "tysięcy"),
    (r"\bmln\.?(?=\s|[.,!?]|$)", "milionów"),
    (r"\bmld\.?(?=\s|[.,!?]|$)", "miliardów"),
    (r"\bul\.", "ulica"),
    (r"\bal\.", "aleja"),
    (r"\bpl\.", "plac"),
    (r"\bos\.", "osiedle"),
    (r"\bstr\.", "strona"),
    (r"\brozdz\.", "rozdział"),
    (r"\bnr\.?(?=\s+\d)", "numer"),
    (r"\bprof\.", "profesor"),
    (r"\bhab\.", "habilitowany"),
    (r"\binż\.", "inżynier"),
    (r"\bmgr\.?(?=\s)", "magister"),
    (r"\bdr\.?(?=\s)", "doktor"),
    (r"\bks\.", "ksiądz"),
    (r"\bśw\.", "święty"),
    (r"\bgen\.", "generał"),
    (r"\bpłk\.?(?=\s)", "pułkownik"),
    (r"\bppłk\.?(?=\s)", "podpułkownik"),
    (r"\bmjr\.?(?=\s)", "major"),
    (r"\bkpt\.", "kapitan"),
    (r"\bpor\.", "porucznik"),
    (r"\bsierż\.", "sierżant"),
    (r"\bkm/h\b", "kilometrów na godzinę"),
    (r"\bkm\b", "kilometrów"),
    (r"\bcm\b", "centymetrów"),
    (r"\bmm\b", "milimetrów"),
    (r"\bkg\b", "kilogramów"),
    (r"\bml\b", "mililitrów"),
    (r"\bzł\b", "złotych"),
    (r"\bgr\b", "groszy"),
    (r"\bEUR\b", "euro"),
    (r"\bUSD\b", "dolarów"),
]

_ROMAN_MAP: dict[str, int] = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
    "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20, "XXI": 21,
    "XXII": 22, "XXIII": 23, "XXIV": 24, "XXV": 25, "XXX": 30, "XL": 40, "L": 50,
}

# Polskie liczebniki porządkowe 1..99: (dopełniacz męski -ego, miejscownik męski -ym/-im)
_ORD_UNITS: dict[int, tuple[str, str]] = {
    1: ("pierwszego", "pierwszym"),
    2: ("drugiego", "drugim"),
    3: ("trzeciego", "trzecim"),
    4: ("czwartego", "czwartym"),
    5: ("piątego", "piątym"),
    6: ("szóstego", "szóstym"),
    7: ("siódmego", "siódmym"),
    8: ("ósmego", "ósmym"),
    9: ("dziewiątego", "dziewiątym"),
    10: ("dziesiątego", "dziesiątym"),
    11: ("jedenastego", "jedenastym"),
    12: ("dwunastego", "dwunastym"),
    13: ("trzynastego", "trzynastym"),
    14: ("czternastego", "czternastym"),
    15: ("piętnastego", "piętnastym"),
    16: ("szesnastego", "szesnastym"),
    17: ("siedemnastego", "siedemnastym"),
    18: ("osiemnastego", "osiemnastym"),
    19: ("dziewiętnastego", "dziewiętnastym"),
}

_ORD_TENS: dict[int, tuple[str, str]] = {
    20: ("dwudziestego", "dwudziestym"),
    30: ("trzydziestego", "trzydziestym"),
    40: ("czterdziestego", "czterdziestym"),
    50: ("pięćdziesiątego", "pięćdziesiątym"),
    60: ("sześćdziesiątego", "sześćdziesiątym"),
    70: ("siedemdziesiątego", "siedemdziesiątym"),
    80: ("osiemdziesiątego", "osiemdziesiątym"),
    90: ("dziewięćdziesiątego", "dziewięćdziesiątym"),
}

_DECADES_PL: dict[int, str] = {
    20: "dwudziestych",
    30: "trzydziestych",
    40: "czterdziestych",
    50: "pięćdziesiątych",
    60: "sześćdziesiątych",
    70: "siedemdziesiątych",
    80: "osiemdziesiątych",
    90: "dziewięćdziesiątych",
}

_HUNDREDS_ORD: dict[int, tuple[str, str]] = {
    100: ("setnego", "setnym"),
    200: ("dwusetnego", "dwusetnym"),
    300: ("trzysetnego", "trzysetnym"),
    400: ("czterysetnego", "czterysetnym"),
    500: ("pięćsetnego", "pięćsetnym"),
    600: ("sześćsetnego", "sześćsetnym"),
    700: ("siedemsetnego", "siedemsetnym"),
    800: ("osiemsetnego", "osiemsetnym"),
    900: ("dziewięćsetnego", "dziewięćsetnym"),
}

_HOURS_NOM = {
    0: "zerowa", 1: "pierwsza", 2: "druga", 3: "trzecia", 4: "czwarta",
    5: "piąta", 6: "szósta", 7: "siódma", 8: "ósma", 9: "dziewiąta",
    10: "dziesiąta", 11: "jedenasta", 12: "dwunasta", 13: "trzynasta",
    14: "czternasta", 15: "piętnasta", 16: "szesnasta", 17: "siedemnasta",
    18: "osiemnasta", 19: "dziewiętnasta", 20: "dwudziesta",
    21: "dwudziesta pierwsza", 22: "dwudziesta druga", 23: "dwudziesta trzecia",
}

_HOURS_OBLIQUE = {
    0: "zerowej", 1: "pierwszej", 2: "drugiej", 3: "trzeciej", 4: "czwartej",
    5: "piątej", 6: "szóstej", 7: "siódmej", 8: "ósmej", 9: "dziewiątej",
    10: "dziesiątej", 11: "jedenastej", 12: "dwunastej", 13: "trzynastej",
    14: "czternastej", 15: "piętnastej", 16: "szesnastej", 17: "siedemnastej",
    18: "osiemnastej", 19: "dziewiętnastej", 20: "dwudziestej",
    21: "dwudziestej pierwszej", 22: "dwudziestej drugiej", 23: "dwudziestej trzeciej",
}

_MONTHS_GEN = (
    "stycznia|lutego|marca|kwietnia|maja|czerwca|"
    "lipca|sierpnia|września|października|listopada|grudnia"
)

# Polskie czasowniki w didaskaliach dialogowych po myślniku ("— Kowalska — powiedział..." -> "Kowalska. Powiedział...")
# Zapobiega to podwójnemu przecinkowi wokół wołacza, który podbija intonację jak w pytaniu.
_DIALOGUE_TAG_RE = re.compile(
    r"(?<![.!?])\s+[—–]\s+("
    r"(?:po|od|wy|za|do|prze|przy|u|w|z|roz|na|o)?(?:wiedział|parł|pytał|dawał|dał|krzyknął|"
    r"szepnął|mruknął|stwierdził|trącił|ciągnął|jaśnił|uważył|stchnął|rychnął|arknął|wołał|"
    r"rzucił|śmiechnął|kiwał|kwitował|mówił|pierał|łumaczył|jęknął|burknął|syknął|rzekł|"
    r"znajmił|znał|pewnił|strzegł|prosił|kazał|ponował|gerował|skazał|skinął|machnął|"
    r"spojrzał|wrócił|śmiał|arsknął|hrząknął|myślał|krzyczał|szeptał|mamrotał|dopytywał|"
    r"dodał|odrzekł|odparł|odpowiedział|wyjaśnił|zauważył|westchnął|warknął|prychnął|"
    r"rzucił|mruknął|bąknął|warknęł|szepnęł|krzyknęł|uśmiechnęł|westchnęł|skinęł|machnęł|"
    r"upierał|tłumaczył|ciągnął|wtrącił|dorzucił|przerwał|dokończył|skwitował|stwierdził)"
    r"[aąeioyłu]*)\b",
    re.IGNORECASE,
)

# Wzorce kontekstowe dla polskiej odmiany
_ROMAN_ORDINAL_PREFIX_RE = re.compile(
    r"\b(Rozdział|Część|Tom|Księga|Akt|Scena|Punkt|Paragraf)\s+([IVXL]{1,6})\b",
    re.IGNORECASE,
)

# Dekady: "lat 70.", "w latach 80.", "lat 90-tych", "lat 70. XX wieku"
_DECADE_RE = re.compile(
    r"\b(lat|latach)\s+(20|30|40|50|60|70|80|90)(?:[-–]?tych|\.(?=\s+(?:[IVXL]{1,5}\b|[a-ząćęłńóśźż])|\s*[,;:]|\s*$))",
    re.IGNORECASE,
)

# Wieki z opcjonalnym przyimkiem: "w XX wieku", "XX wieku", "XIX w.", "w XXI w."
_ROMAN_CENTURY_RE = re.compile(
    r"\b(?:(w|we|od|do|z|ze|od początku|pod koniec|na początku|w połowie)\s+)?([IVXL]{1,5})\s+(wiek(?:u|iem|owi|ów)?|w\.)(?=\s|$|[.,;:!?])",
    re.IGNORECASE,
)

# Daty: "15 maja 1946 roku" lub "15 maja"
_DATE_DAY_MONTH_RE = re.compile(
    rf"\b([1-9]|[12]\d|3[01])\.?\s+({_MONTHS_GEN})(?:\s+(\d{{4}})\s*(?:roku|r\.))?\b",
    re.IGNORECASE,
)

# Lata z przyimkiem i słowem "roku" / "r.": "w 1946 roku", "od 1989 r.", "do 2026 roku"
_YEAR_CONTEXT_RE = re.compile(
    r"\b(w|we|od|do|z|ze|sprzed|roku)\s+(\d{4})\s+(roku|r\.)",
    re.IGNORECASE,
)
# Samodzielne "1946 roku" / "1946 r." (bez przyimka przed) -> dopełniacz ("tysiąc dziewięćset czterdziestego szóstego roku")
_YEAR_STANDALONE_RE = re.compile(r"\b(1\d{3}|20\d{2})\s+(roku|r\.)", re.IGNORECASE)

# Godziny z przyimkiem: "o godzinie 14:30", "o 14:30", "do 16:00", "od 8:15"
_TIME_OBLIQUE_RE = re.compile(
    r"\b(o\s+godzinie|do\s+godziny|od\s+godziny|po\s+godzinie|przed\s+godziną|o|do|od|po|przed|koło|około)\s+([01]?\d|2[0-3]):([0-5]\d)\b",
    re.IGNORECASE,
)
_TIME_PLAIN_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")

_RANGE_RE = re.compile(r"\b(\d{1,6})\s*[–—-]\s*(\d{1,6})\b")
_PERCENT_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*%")
_DECIMAL_RE = re.compile(r"\b(\d+)[.,](\d+)\b")
_INT_RE = re.compile(r"\b\d{1,9}\b")


def _safe_num2words(n: int, to: str = "cardinal") -> str:
    try:
        return num2words(n, lang="pl", to=to)
    except Exception:
        return str(n)


def _ordinal_1_99(n: int, case: str = "gen") -> str:
    """Zwraca polski liczebnik porządkowy 1..99 w dopełniaczu ('gen') lub miejscowniku ('loc')."""
    idx = 0 if case == "gen" else 1
    if n in _ORD_UNITS:
        return _ORD_UNITS[n][idx]
    tens = (n // 10) * 10
    units = n % 10
    if tens in _ORD_TENS:
        if units == 0:
            return _ORD_TENS[tens][idx]
        return f"{_ORD_TENS[tens][idx]} {_ORD_UNITS[units][idx]}"
    return _safe_num2words(n, to="ordinal")


def _format_year_pl(year: int, case: str = "loc") -> str:
    """Odmienia rok (1000..2099) po polsku: 'loc' (w 1946 roku) lub 'gen' (z 1946 roku)."""
    idx = 0 if case == "gen" else 1
    if year == 2000:
        return "dwutysięcznego" if case == "gen" else "dwutysięcznym"
    if year == 1000:
        return "tysięcznego" if case == "gen" else "tysięcznym"

    thousands = (year // 1000) * 1000
    rem = year % 1000
    hundreds = (rem // 100) * 100
    last_two = rem % 100

    parts: list[str] = []
    if thousands == 1000:
        parts.append("tysiąc")
    elif thousands == 2000:
        parts.append("dwa tysiące")
    elif thousands > 0:
        parts.append(_safe_num2words(thousands))

    if last_two == 0 and hundreds in _HUNDREDS_ORD:
        parts.append(_HUNDREDS_ORD[hundreds][idx])
    else:
        if hundreds > 0:
            parts.append(_safe_num2words(hundreds))
        if last_two > 0:
            parts.append(_ordinal_1_99(last_two, case=case))

    return " ".join(parts)


def _replace_pronunciation(text: str, pronunciation_dict: dict[str, str]) -> str:
    if not pronunciation_dict:
        return text
    for src in sorted(pronunciation_dict.keys(), key=len, reverse=True):
        dst = pronunciation_dict[src]
        if not src:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(src)}(?!\w)")
        text = pattern.sub(dst, text)
    return text


def normalize_polish_text(
    text: str,
    pronunciation_dict: dict[str, str] | None = None,
) -> str:
    """Normalizuje polski tekst literacki pod syntezę IPA (F5-TTS multi3)."""
    if not text:
        return ""

    # 1. Czyste białe znaki i ujednolicenie cudzysłowów / myślników
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("…", "...")
    text = text.replace("„", "").replace("”", "").replace("“", "").replace("«", "").replace("»", "")
    text = text.replace('"', "")

    text = re.sub(r"^\s*[—–-]\s+", "", text)
    # Po kropce/wykrzykniku/pytajniku myślnik dialogowy (". — ") nie dodaje przecinka
    text = re.sub(r"([.!?])\s*[—–]\s*", r"\1 ", text)
    # Myślnik przed didaskaliami dialogowymi ("— powiedział", "— upierała się") zamykamy kropką
    # zamiast przecinka, co utrzymuje stanowczą, oznajmującą intonację na końcu wypowiedzi!
    text = _DIALOGUE_TAG_RE.sub(lambda m: f". {m.group(1)[0].upper()}{m.group(1)[1:]}", text)
    text = re.sub(r"\s+[—–]\s+", ", ", text)
    text = re.sub(r"[—–]", ", ", text)

    # 2. Słownik wymowy (np. audiobooka -> audiobuka, NVIDIA -> Enwidia, nazwy własne)
    if pronunciation_dict:
        text = _replace_pronunciation(text, pronunciation_dict)

    # 3. Polskie skróty (najpierw "o godz." -> "o godzinie" przed zamianą godzin!)
    for pat, repl in _ABBREVIATIONS:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)

    # 4. Dekady: "lat 70. XX wieku" -> "lat siedemdziesiątych XX wieku" (nie rozcina zdania kropką!)
    def _repl_decade(m: re.Match) -> str:
        prefix, dec_str = m.group(1), m.group(2)
        dec_word = _DECADES_PL.get(int(dec_str), dec_str)
        # Jeśli kropka była na końcu całego napisu, zachowaj kropkę kończącą zdanie
        trailing_dot = "." if m.end() == len(text) and m.group(0).endswith(".") else ""
        return f"{prefix} {dec_word}{trailing_dot}"

    text = _DECADE_RE.sub(_repl_decade, text)

    # 5. Liczby rzymskie w nagłówkach ("Rozdział IV" -> "Rozdział czwarty")
    def _repl_roman_prefix(m: re.Match) -> str:
        word, roman = m.group(1), m.group(2).upper()
        val = _ROMAN_MAP.get(roman)
        if val is None:
            return m.group(0)
        return f"{word} {_safe_num2words(val, to='ordinal')}"

    text = _ROMAN_ORDINAL_PREFIX_RE.sub(_repl_roman_prefix, text)

    # 6. Wieki rzymskie: "w XX wieku" -> "w dwudziestym wieku", "XX wieku" -> "dwudziestego wieku"
    def _repl_century(m: re.Match) -> str:
        prep = m.group(1)
        roman = m.group(2).upper()
        unit = m.group(3)
        val = _ROMAN_MAP.get(roman)
        if val is None:
            return m.group(0)
        if unit.lower() == "w.":
            unit = "wieku"
        case = "loc" if (prep and prep.lower() in ("w", "we")) else "gen"
        ord_word = _ordinal_1_99(val, case=case)
        return f"{prep} {ord_word} {unit}" if prep else f"{ord_word} {unit}"

    text = _ROMAN_CENTURY_RE.sub(_repl_century, text)

    # 7. Daty: "15 maja" -> "piętnastego maja", "15 maja 1946 roku" -> "... czterdziestego szóstego roku"
    def _repl_date(m: re.Match) -> str:
        day = int(m.group(1))
        month = m.group(2)
        year_s = m.group(3)
        day_word = _ordinal_1_99(day, case="gen")
        if year_s:
            year_word = _format_year_pl(int(year_s), case="gen")
            return f"{day_word} {month} {year_word} roku"
        return f"{day_word} {month}"

    text = _DATE_DAY_MONTH_RE.sub(_repl_date, text)

    # 8. Lata z przyimkiem: "w 1946 roku" -> "w tysiąc dziewięćset czterdziestym szóstym roku"
    def _repl_year_ctx(m: re.Match) -> str:
        prep, year_s = m.group(1), m.group(2)
        case = "loc" if prep.lower() in ("w", "we") else "gen"
        return f"{prep} {_format_year_pl(int(year_s), case=case)} roku"

    text = _YEAR_CONTEXT_RE.sub(_repl_year_ctx, text)

    # Samodzielne lata: "1946 roku" / "1946 r." -> dopełniacz
    text = _YEAR_STANDALONE_RE.sub(
        lambda m: f"{_format_year_pl(int(m.group(1)), case='gen')} roku",
        text,
    )

    # 9. Godziny z przyimkiem ("o godzinie 14:30" -> "o godzinie czternastej trzydzieści")
    def _repl_time_oblique(m: re.Match) -> str:
        prep = m.group(1)
        h, mins = int(m.group(2)), int(m.group(3))
        h_word = _HOURS_OBLIQUE.get(h, _safe_num2words(h))
        if mins == 0:
            return f"{prep} {h_word}"
        return f"{prep} {h_word} {_safe_num2words(mins)}"

    text = _TIME_OBLIQUE_RE.sub(_repl_time_oblique, text)

    # Pozostałe godziny bez przyimka ("14:30" -> "czternasta trzydzieści")
    def _repl_time_plain(m: re.Match) -> str:
        h, mins = int(m.group(1)), int(m.group(2))
        h_word = _HOURS_NOM.get(h, _safe_num2words(h))
        if mins == 0:
            return h_word
        return f"{h_word} {_safe_num2words(mins)}"

    text = _TIME_PLAIN_RE.sub(_repl_time_plain, text)

    # 10. Procenty
    def _repl_percent(m: re.Match) -> str:
        num_str = m.group(1).replace(".", ",")
        if "," in num_str:
            a, b = num_str.split(",", 1)
            spoken = f"{_safe_num2words(int(a))} przecinek {_safe_num2words(int(b))}"
        else:
            spoken = _safe_num2words(int(num_str))
        return f"{spoken} procent"

    text = _PERCENT_RE.sub(_repl_percent, text)

    # 11. Zakresy liczbowe ("10-15" -> "dziesięć do piętnaście")
    text = _RANGE_RE.sub(
        lambda m: f"{_safe_num2words(int(m.group(1)))} do {_safe_num2words(int(m.group(2)))}",
        text,
    )

    # 12. Ułamki dziesiętne ("3,5" -> "trzy przecinek pięć")
    text = _DECIMAL_RE.sub(
        lambda m: f"{_safe_num2words(int(m.group(1)))} przecinek {_safe_num2words(int(m.group(2)))}",
        text,
    )

    # 13. Pozostałe liczby całkowite
    text = _INT_RE.sub(lambda m: _safe_num2words(int(m.group(0))), text)

    # 14. Kosmetyka interpunkcji
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"([.!?]),+", r"\1", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[,;:\s]+", "", text).strip()

    return text
