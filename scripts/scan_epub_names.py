"""Skaner EPUB wykrywający obce imiona, nazwiska, nazwy własne, skrótowce i słowa z apostrofami."""

import re
import sys
from collections import Counter
from pathlib import Path

from f5_audiobook.text.epub_parser import parse_epub

# Polskie pospolite słowa często zaczynające zdanie (do odfiltrowania)
_COMMON_PL_STARTS = {
    "Nie", "Ale", "Gdy", "Kiedy", "Jednak", "Potem", "Wtedy", "Teraz", "Przez", "Przed",
    "Jego", "Jej", "Ich", "Może", "Musi", "Tylko", "Nawet", "Jeśli", "Choć", "Chociaż",
    "Wiedział", "Wiedziała", "Spojrzał", "Spojrzała", "Zawsze", "Nigdy", "Wszystko",
    "Wszyscy", "Nikt", "Nic", "Coś", "Ktoś", "Gdzie", "Gdzieś", "Tam", "Tutaj", "Tutaj",
    "Tak", "Czy", "Jak", "Jako", "Który", "Która", "Które", "Którzy", "Był", "Była", "Było",
    "Były", "Byli", "Miał", "Miała", "Miało", "Mieli", "Jest", "Są", "Będzie", "Będą",
    "Silos", "Silosu", "Silosie", "Silosem", "Silosów", "Silosy", "Dział", "Działu",
    "Dziale", "Poziom", "Poziomu", "Poziomie", "Mechaniczny", "Mechanicznego",
    "Szeryf", "Szeryfa", "Burmistrz", "Zastępca", "Czyszczenie", "Czyszczenia",
    "Rozdział", "Część", "Prolog", "Epilog", "Pakt", "Paktu", "Pakcie",
    "Kombinezon", "Kombinezonu", "Kombinezonie", "Klatka", "Klatki", "Klatce",
    "Schody", "Schodów", "Ekran", "Ekrany", "Ekranie", "Biuro", "Biura", "Biurze",
    "Więc", "Zatem", "Oczywiście", "Nagle", "Nareszcie", "Wreszcie", "Wciąż", "Ciągle",
    "Dziś", "Dzisiaj", "Wczoraj", "Jutro", "Rano", "Wieczorem", "Nocą", "Dzień",
    "Boże", "Panie", "Pani", "Pan", "Dobrze", "Dobra", "Przepraszam", "Dziękuję",
    "Cześć", "Halo", "Chodź", "Patrz", "Słuchaj", "Czekaj", "Poczekaj", "Przestań",
    "Dlaczego", "Dlatego", "Ponieważ", "Skoro", "Zanim", "Dopóki", "Dopiero", "Wkrótce",
    "Zaraz", "Nadal", "Jeszcze", "Już", "Zbyt", "Bardzo", "Zupełnie", "Całkiem", "Prawie",
    "Niemal", "Przecież", "Właśnie", "Właściwie", "Jedynie", "Poza", "Pod", "Nad", "Przy",
    "Bez", "Dla", "Od", "Do", "Ze", "We", "Na", "Po", "Za", "Ona", "Oni", "One", "Ono",
    "Ten", "Ta", "To", "Te", "Ci", "Tego", "Tej", "Tym", "Tych", "Tymi", "Tę", "Tą",
    "Mój", "Moja", "Moje", "Twój", "Twoja", "Twoje", "Nasz", "Nasza", "Nasze", "Swój",
    "Każdy", "Każda", "Każde", "Inny", "Inna", "Inne", "Inni", "Sam", "Sama", "Samo",
    "Sami", "Same", "Taki", "Taka", "Takie", "Tacy", "Ile", "Ilu", "Kilka", "Kilku",
    "Wiele", "Wielu", "Dużo", "Mało", "Więcej", "Mniej", "Pierwszy", "Pierwsza", "Drugi",
    "Druga", "Trzeci", "Czwarty", "Piąty", "Ostatni", "Ostatnia", "Nowy", "Nowa", "Stary",
    "Stara", "Wielki", "Wielka", "Mały", "Mała", "Czarny", "Biały", "Czerwony", "Zielony",
    "Woda", "Wody", "Ziemia", "Ziemi", "Niebo", "Nieba", "Świat", "Świata", "Świecie",
    "Ludzie", "Ludzi", "Człowiek", "Kobieta", "Mężczyzna", "Dziewczyna", "Chłopak",
    "Dziecko", "Dzieci", "Matka", "Ojciec", "Żona", "Mąż", "Syn", "Córka", "Brat", "Siostra",
    "Głowa", "Ręka", "Ręce", "Oczy", "Twarz", "Głos", "Słowa", "Myśli", "Serce", "Ciało",
    "Czas", "Czasu", "Chwilę", "Chwila", "Minutę", "Godzinę", "Lata", "Rok", "Roku",
    "Drzwi", "Okno", "Ściana", "Podłoga", "Sufit", "Stół", "Krzesło", "Łóżko", "Pokój",
    "Dom", "Domu", "Miasto", "Miasta", "Światło", "Ciemność", "Powietrze", "Wiatr", "Pył",
}


def main() -> None:
    epub_path = Path(sys.argv[1])
    book = parse_epub(epub_path)
    print(f"TITLE: {book.title} | AUTHOR: {book.author} | CHAPTERS: {len(book.chapters)}")
    for c in book.chapters:
        print(f"  ch{c.index:02d} [{'X' if c.selected else ' '}] ({c.char_count:6d} ch): {c.title}")

    full_text = "\n\n".join(c.text for c in book.chapters)

    # 1. Skrótowce WIELKIMI LITERAMI (2-6 liter)
    acronyms = Counter(re.findall(r"\b[A-Z]{2,6}(?:-[a-ząćęłńóśźż]+)?\b", full_text))
    print("\n=== SKRÓTOWCE (ALL CAPS, >= 2 wyst.) ===")
    for k, v in acronyms.most_common(60):
        if v >= 2:
            print(f"  {k}: {v}")

    # 2. Słowa z apostrofem (prawie zawsze obce nazwiska w polskiej odmianie: George'a, Solo'a, McLain'a itp.)
    apost = Counter(re.findall(r"\b[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+['’][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+\b", full_text))
    print("\n=== SŁOWA Z APOSTROFEM ===")
    for k, v in apost.most_common(80):
        print(f"  {k}: {v}")

    # 3. Nazwy własne pojawiające się w ŚRODKU zdania (nie po kropce/wykrzykniku/pytajniku/myślniku/nowej linii)
    mid_caps = Counter(
        re.findall(r"(?<=[a-ząćęłńóśźż,;:])\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż'’-]{2,})\b", full_text)
    )
    print("\n=== NAZWY WŁASNE W ŚRODKU ZDANIA (>= 2 wyst.) ===")
    for k, v in mid_caps.most_common(250):
        if v >= 2 and k not in _COMMON_PL_STARTS:
            print(f"  {k}: {v}")

    # 4. Słowa z wyraźnie obcymi literami (v, q, x) lub podwójnymi spółgłoskami/samogłoskami nietypowymi dla PL
    foreign_pat = re.compile(
        r"\b[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]*(?:[vqxVQX]|sh|th|gh|ph|ck|wh|ee|oo|ey\b|ay\b|ew\b|ow\b|tt|ll|bb|dd|ff|gg|pp|rr|mc)[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]*\b",
        re.IGNORECASE,
    )
    fwords = Counter(foreign_pat.findall(full_text))
    print("\n=== SŁOWA O OBCEJ PISOWNI (v/q/x/sh/th/ll/tt/ee/oo/ck itd., >= 2 wyst.) ===")
    for k, v in fwords.most_common(200):
        if v >= 2:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
