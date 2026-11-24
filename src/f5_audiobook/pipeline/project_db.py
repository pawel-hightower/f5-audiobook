"""Baza SQLite przechowująca stan generowania audiobooka (rozdziały, chunki, seedy, oceny QG)."""

from __future__ import annotations

import re
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from f5_audiobook.config import WORK_DIR
from f5_audiobook.text.chunker import chunk_paragraphs
from f5_audiobook.text.epub_parser import BookData


@dataclass
class ChapterRow:
    id: int
    chapter_index: int
    title: str
    selected: bool
    total_chunks: int = 0
    done_chunks: int = 0


@dataclass
class ChunkRow:
    id: int
    chapter_id: int
    chapter_index: int
    chapter_title: str
    chunk_index: int
    raw_text: str
    normalized_text: str
    pause_after_ms: int
    status: str  # pending | done | error
    audio_path: str | None
    duration_sec: float
    seed: int
    qg_score: float | None
    qg_note: str
    heard_text: str
    error_msg: str


def slugify_project_name(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ_-]+", "_", title.strip()).strip("_")
    return slug[:80] or "audiobook"


class ProjectDB:
    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir = self.project_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.project_dir / "project.db"
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_index INTEGER NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    selected INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    raw_text TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    pause_after_ms INTEGER NOT NULL DEFAULT 380,
                    status TEXT NOT NULL DEFAULT 'pending',
                    audio_path TEXT,
                    duration_sec REAL NOT NULL DEFAULT 0.0,
                    seed INTEGER NOT NULL DEFAULT 101,
                    qg_score REAL,
                    qg_note TEXT NOT NULL DEFAULT '',
                    heard_text TEXT NOT NULL DEFAULT '',
                    error_msg TEXT NOT NULL DEFAULT '',
                    UNIQUE(chapter_id, chunk_index)
                );
                """
            )

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_meta(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else default

    def get_all_meta(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
            return {r["key"]: r["value"] for r in rows}

    @classmethod
    def create_from_book(
        cls,
        book: BookData,
        pronunciation_dict: dict[str, str] | None = None,
        work_root: Path = WORK_DIR,
        base_seed: int = 101,
        max_chars: int = 165,
    ) -> "ProjectDB":
        slug = slugify_project_name(book.title)
        proj_dir = work_root / slug
        db = cls(proj_dir)

        # Zapisz okładkę jeśli istnieje
        cover_path_str = ""
        if book.cover_bytes:
            cover_file = proj_dir / f"cover.{book.cover_ext}"
            cover_file.write_bytes(book.cover_bytes)
            cover_path_str = str(cover_file)

        db.set_meta("title", book.title)
        db.set_meta("author", book.author)
        db.set_meta("source_path", str(book.source_path))
        db.set_meta("cover_path", cover_path_str)

        with db._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM chapters")

            for ch in book.chapters:
                cur = conn.execute(
                    "INSERT INTO chapters(chapter_index, title, selected) VALUES (?, ?, ?)",
                    (ch.index, ch.title, 1 if ch.selected else 0),
                )
                chapter_id = cur.lastrowid
                chunks = chunk_paragraphs(
                    ch.paragraphs,
                    pronunciation_dict=pronunciation_dict,
                    max_chars=max_chars,
                    chapter_title=ch.title,
                    include_title=True,
                )
                for c in chunks:
                    conn.execute(
                        """
                        INSERT INTO chunks(
                            chapter_id, chunk_index, raw_text, normalized_text,
                            pause_after_ms, status, seed
                        ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
                        """,
                        (
                            chapter_id,
                            c.index,
                            c.raw_text,
                            c.normalized_text,
                            c.pause_after_ms,
                            base_seed + ch.index * 1000 + c.index,
                        ),
                    )
        return db

    def list_chapters(self) -> list[ChapterRow]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    ch.id, ch.chapter_index, ch.title, ch.selected,
                    COUNT(c.id) AS total_chunks,
                    SUM(CASE WHEN c.status = 'done' THEN 1 ELSE 0 END) AS done_chunks
                FROM chapters ch
                LEFT JOIN chunks c ON c.chapter_id = ch.id
                GROUP BY ch.id
                ORDER BY ch.chapter_index
                """
            ).fetchall()
        return [
            ChapterRow(
                id=r["id"],
                chapter_index=r["chapter_index"],
                title=r["title"],
                selected=bool(r["selected"]),
                total_chunks=r["total_chunks"] or 0,
                done_chunks=r["done_chunks"] or 0,
            )
            for r in rows
        ]

    def set_chapter_selected(self, chapter_id: int, selected: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chapters SET selected=? WHERE id=?",
                (1 if selected else 0, chapter_id),
            )

    def _row_to_chunk(self, r: sqlite3.Row) -> ChunkRow:
        return ChunkRow(
            id=r["id"],
            chapter_id=r["chapter_id"],
            chapter_index=r["chapter_index"],
            chapter_title=r["chapter_title"],
            chunk_index=r["chunk_index"],
            raw_text=r["raw_text"],
            normalized_text=r["normalized_text"],
            pause_after_ms=r["pause_after_ms"],
            status=r["status"],
            audio_path=r["audio_path"],
            duration_sec=float(r["duration_sec"] or 0.0),
            seed=int(r["seed"]),
            qg_score=float(r["qg_score"]) if r["qg_score"] is not None else None,
            qg_note=r["qg_note"] or "",
            heard_text=r["heard_text"] or "",
            error_msg=r["error_msg"] or "",
        )

    def list_chunks(self, chapter_id: int | None = None, only_selected: bool = False) -> list[ChunkRow]:
        where = []
        params: list = []
        if chapter_id is not None:
            where.append("c.chapter_id = ?")
            params.append(chapter_id)
        if only_selected:
            where.append("ch.selected = 1")
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.*, ch.chapter_index, ch.title AS chapter_title
                FROM chunks c
                JOIN chapters ch ON ch.id = c.chapter_id
                {where_sql}
                ORDER BY ch.chapter_index, c.chunk_index
                """,
                params,
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_chunk(self, chunk_id: int) -> ChunkRow:
        with self._connect() as conn:
            r = conn.execute(
                """
                SELECT c.*, ch.chapter_index, ch.title AS chapter_title
                FROM chunks c
                JOIN chapters ch ON ch.id = c.chapter_id
                WHERE c.id = ?
                """,
                (chunk_id,),
            ).fetchone()
        if r is None:
            raise KeyError(f"Nie znaleziono chunka id={chunk_id}")
        return self._row_to_chunk(r)

    def get_pending_chunks(self) -> list[ChunkRow]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.*, ch.chapter_index, ch.title AS chapter_title
                FROM chunks c
                JOIN chapters ch ON ch.id = c.chapter_id
                WHERE ch.selected = 1 AND c.status != 'done'
                ORDER BY ch.chapter_index, c.chunk_index
                """
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_progress(self) -> tuple[int, int, int]:
        """Zwraca (done, error, total) dla zaznaczonych rozdziałów."""
        with self._connect() as conn:
            r = conn.execute(
                """
                SELECT
                    COUNT(c.id) AS total,
                    SUM(CASE WHEN c.status = 'done' THEN 1 ELSE 0 END) AS done,
                    SUM(CASE WHEN c.status = 'error' THEN 1 ELSE 0 END) AS err
                FROM chunks c
                JOIN chapters ch ON ch.id = c.chapter_id
                WHERE ch.selected = 1
                """
            ).fetchone()
        return int(r["done"] or 0), int(r["err"] or 0), int(r["total"] or 0)

    def mark_chunk_done(
        self,
        chunk_id: int,
        audio_path: Path,
        duration_sec: float,
        seed: int,
        qg_score: float | None = None,
        qg_note: str = "",
        heard_text: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chunks
                SET status='done', audio_path=?, duration_sec=?, seed=?,
                    qg_score=?, qg_note=?, heard_text=?, error_msg=''
                WHERE id=?
                """,
                (str(audio_path), duration_sec, seed, qg_score, qg_note, heard_text, chunk_id),
            )

    def mark_chunk_error(self, chunk_id: int, error_msg: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chunks SET status='error', error_msg=? WHERE id=?",
                (error_msg, chunk_id),
            )

    def update_chunk_text(self, chunk_id: int, new_normalized_text: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chunks SET normalized_text=?, status='pending' WHERE id=?",
                (new_normalized_text.strip(), chunk_id),
            )

    def reapply_pronunciation_to_pending(self, pronunciation_dict: dict[str, str]) -> int:
        """Stosuje aktualny słownik wymowy do wszystkich jeszcze niewygenerowanych (pending) chunków."""
        from f5_audiobook.text.normalizer_pl import normalize_polish_text

        updated = 0
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, normalized_text FROM chunks WHERE status = 'pending'"
            ).fetchall()
            for r in rows:
                old_txt = r["normalized_text"]
                new_txt = normalize_polish_text(old_txt, pronunciation_dict)
                if new_txt and new_txt != old_txt:
                    conn.execute(
                        "UPDATE chunks SET normalized_text=? WHERE id=?",
                        (new_txt, r["id"]),
                    )
                    updated += 1
        return updated

    def chunk_audio_path(self, chunk: ChunkRow) -> Path:
        return self.chunks_dir / f"ch{chunk.chapter_index:03d}_c{chunk.chunk_index:05d}.wav"

    def resolve_chunk_audio_path(self, chunk: ChunkRow) -> Path | None:
        """Zwraca istniejącą ścieżkę pliku WAV chunka (odporna na przeniesienie folderu projektu)."""
        if chunk.audio_path:
            p = Path(chunk.audio_path)
            if p.exists():
                return p
            relocated = self.chunks_dir / p.name
            if relocated.exists():
                return relocated
        fallback = self.chunk_audio_path(chunk)
        return fallback if fallback.exists() else None

    def update_book_metadata(self, title: str, author: str) -> None:
        """Aktualizuje tytuł i autora książki w metadanych projektu."""
        clean_title = title.strip()
        clean_author = author.strip()
        if not clean_title:
            raise ValueError("Tytuł książki nie może być pusty.")
        self.set_meta("title", clean_title)
        self.set_meta("author", clean_author or "Nieznany autor")

    def set_cover_image(self, src_image: Path) -> Path:
        """Kopiuje nowy obraz okładki do folderu projektu i zapisuje ścieżkę w metadanych."""
        src = Path(src_image)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Nie znaleziono pliku okładki: {src}")
        ext = src.suffix.lower().lstrip(".") or "jpg"
        if ext == "jpeg":
            ext = "jpg"
        dest = self.project_dir / f"cover.{ext}"
        # Usuń stare pliki cover.* z innym rozszerzeniem, żeby nie śmiecić
        for old_cover in self.project_dir.glob("cover.*"):
            if old_cover.resolve() != dest.resolve():
                old_cover.unlink(missing_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        self.set_meta("cover_path", str(dest))
        return dest

    def rename_project(self, new_name: str) -> Path:
        """Bezpiecznie zmienia nazwę folderu projektu w work/ i aktualizuje wszystkie ścieżki w SQLite."""
        new_slug = slugify_project_name(new_name)
        if not new_slug:
            raise ValueError("Nazwa projektu nie może być pusta.")

        old_dir = self.project_dir.resolve()
        new_dir = old_dir.parent / new_slug
        if new_dir.name == old_dir.name:
            return self.project_dir

        case_only_rename = new_dir.name.lower() == old_dir.name.lower()
        if not case_only_rename and new_dir.exists():
            raise FileExistsError(f"Projekt o nazwie „{new_slug}” już istnieje:\n{new_dir}")

        # Zrób pełny checkpoint WAL i zamknij pliki -wal/-shm przed zmianą nazwy folderu (ważne na Windows)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.execute("PRAGMA journal_mode=DELETE;")
            conn.commit()
        finally:
            conn.close()

        if case_only_rename:
            tmp_dir = old_dir.parent / f"{old_dir.name}__renaming_tmp"
            old_dir.rename(tmp_dir)
            tmp_dir.rename(new_dir)
        else:
            old_dir.rename(new_dir)

        self.project_dir = new_dir
        self.chunks_dir = new_dir / "chunks"
        self.db_path = new_dir / "project.db"

        # Zaktualizuj ścieżki wewnątrz project.db (cover_path i wszystkie chunks.audio_path)
        old_prefix = str(old_dir).rstrip("\\/").lower()
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='cover_path'").fetchone()
            if row and row["value"]:
                old_cover = Path(row["value"])
                new_cover = new_dir / old_cover.name
                if new_cover.exists() or str(old_cover).lower().startswith(old_prefix):
                    conn.execute(
                        "UPDATE meta SET value=? WHERE key='cover_path'",
                        (str(new_cover),),
                    )

            chunk_rows = conn.execute(
                "SELECT id, audio_path FROM chunks WHERE audio_path IS NOT NULL AND audio_path != ''"
            ).fetchall()
            for r in chunk_rows:
                old_ap = Path(r["audio_path"])
                new_ap = self.chunks_dir / old_ap.name
                if new_ap.exists() or str(old_ap).lower().startswith(old_prefix):
                    conn.execute(
                        "UPDATE chunks SET audio_path=? WHERE id=?",
                        (str(new_ap), r["id"]),
                    )

        return self.project_dir


def list_saved_projects(work_root: Path = WORK_DIR) -> list[Path]:
    if not work_root.exists():
        return []
    return sorted(
        [d for d in work_root.iterdir() if d.is_dir() and (d / "project.db").exists()],
        key=lambda p: (p / "project.db").stat().st_mtime,
        reverse=True,
    )
