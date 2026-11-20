"""Interfejs wiersza poleceń (CLI) dla F5-Audiobook."""

from __future__ import annotations

import argparse
from pathlib import Path

from f5_audiobook.audio.exporter import export_audiobook
from f5_audiobook.config import ensure_dirs, load_pronunciation_dict
from f5_audiobook.engine.f5_ipa import F5IpaEngine
from f5_audiobook.pipeline.batch_runner import BatchOptions, BatchRunner
from f5_audiobook.pipeline.project_db import ProjectDB
from f5_audiobook.text.epub_parser import parse_book
from f5_audiobook.voices import load_voice_profile


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="F5-Audiobook CLI — konwersja EPUB/TXT na polski audiobook.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Parsuj książkę, wygeneruj audio i wyeksportuj M4B/MP3.")
    p_build.add_argument("book", type=Path, help="Ścieżka do pliku .epub lub .txt")
    p_build.add_argument("--voice", type=str, default="moj_lektor", help="Nazwa profilu lektora w voices/")
    p_build.add_argument("--no-qg", action="store_true", help="Wyłącz Quality Guard (Whisper)")
    p_build.add_argument("--no-mp3", action="store_true", help="Pomiń eksport MP3")
    p_build.add_argument("--no-m4b", action="store_true", help="Pomiń eksport M4B")
    p_build.add_argument("--resume", type=Path, default=None, help="Wznów istniejący folder projektu w work/")

    args = parser.parse_args()

    if args.cmd == "build":
        voice = load_voice_profile(args.voice)
        if args.resume:
            db = ProjectDB(args.resume)
        else:
            book = parse_book(args.book)
            pron = load_pronunciation_dict()
            db = ProjectDB.create_from_book(book, pronunciation_dict=pron, base_seed=voice.seed)
            print(f"Utworzono projekt: {db.project_dir} ({len(book.chapters)} rozdziałów)")

        engine = F5IpaEngine()
        runner = BatchRunner(engine)
        opts = BatchOptions(use_quality_guard=not args.no_qg)
        prog = runner.run(db, voice, opts, log_cb=print)
        print(prog.message)

        if prog.done > 0 and (not args.no_mp3 or not args.no_m4b):
            res = export_audiobook(
                db,
                export_mp3=not args.no_mp3,
                export_m4b=not args.no_m4b,
                log_cb=print,
            )
            print(f"Gotowe! Folder: {res.output_dir} | M4B: {res.m4b_file} | MP3: {len(res.mp3_files)}")


if __name__ == "__main__":
    main()
