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
    multi_voice: bool = False
    dialogue_tagged: int = 0
    dialogue_untagged: int = 0


@dataclass
class RechunkMultiVoiceResult:
    converted_chapters: int
    skipped_done_chapters: int
    new_chunks: int
    dialogue_chunks: int


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
    voice_role: str = "narrator"  # narrator | male | female | dialogue


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
                    selected INTEGER NOT NULL DEFAULT 1,
                    multi_voice INTEGER NOT NULL DEFAULT 0
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
                    voice_role TEXT NOT NULL DEFAULT 'narrator',
                    UNIQUE(chapter_id, chunk_index)
                );
                """
            )
            # Automatyczna migracja starszych projektów (np. work/Silos/project.db)
            ch_cols = {r["name"] for r in conn.execute("PRAGMA table_info(chapters)").fetchall()}
            if "multi_voice" not in ch_cols:
                conn.execute("ALTER TABLE chapters ADD COLUMN multi_voice INTEGER NOT NULL DEFAULT 0")
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
            if "voice_role" not in cols:
                conn.execute("ALTER TABLE chunks ADD COLUMN voice_role TEXT NOT NULL DEFAULT 'narrator'")

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
        multi_voice: bool = False,
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
        db.set_meta("multi_voice", "1" if multi_voice else "0")

        with db._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM chapters")

            for ch in book.chapters:
                cur = conn.execute(
                    "INSERT INTO chapters(chapter_index, title, selected, multi_voice) VALUES (?, ?, ?, ?)",
                    (ch.index, ch.title, 1 if ch.selected else 0, 1 if multi_voice else 0),
                )
                chapter_id = cur.lastrowid
                chunks = chunk_paragraphs(
                    ch.paragraphs,
                    pronunciation_dict=pronunciation_dict,
                    max_chars=max_chars,
                    chapter_title=ch.title,
                    include_title=True,
                    multi_voice=multi_voice,
                )
                for c in chunks:
                    conn.execute(
                        """
                        INSERT INTO chunks(
                            chapter_id, chunk_index, raw_text, normalized_text,
                            pause_after_ms, status, seed, voice_role
                        ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                        """,
                        (
                            chapter_id,
                            c.index,
                            c.raw_text,
                            c.normalized_text,
                            c.pause_after_ms,
                            base_seed + ch.index * 1000 + c.index,
                            c.voice_role,
                        ),
                    )
        return db

    def is_multi_voice(self) -> bool:
        if self.get_meta("multi_voice", "0") == "1":
            return True
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM chapters WHERE multi_voice = 1 LIMIT 1").fetchone()
            return row is not None

    def rechunk_selected_chapters_multi_voice(
        self,
        pronunciation_dict: dict[str, str] | None = None,
        source_path: Path | None = None,
        base_seed: int = 101,
        max_chars: int = 165,
        skip_with_done_audio: bool = True,
    ) -> RechunkMultiVoiceResult:
        """Dzieli na nowo zaznaczone rozdziały istniejącego projektu w trybie 3 głosów (dialogi + narrator).

        Domyślnie bezpiecznie pomija rozdziały, które mają już wygenerowane pliki WAV (done_chunks > 0).
        """
        from f5_audiobook.text.epub_parser import parse_book

        src = Path(source_path) if source_path is not None else Path(self.get_meta("source_path", ""))
        if not str(src) or not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Nie znaleziono pliku źródłowego książki: {src}")

        if source_path is not None:
            self.set_meta("source_path", str(src))

        book = parse_book(src)
        by_index = {ch.index: ch for ch in book.chapters}
        by_title = {ch.title.strip().lower(): ch for ch in book.chapters}

        chapters = [ch for ch in self.list_chapters() if ch.selected]
        converted = 0
        skipped_done = 0
        new_chunks_total = 0
        dialogue_chunks_total = 0

        with self._connect() as conn:
            for ch in chapters:
                if ch.done_chunks > 0 and skip_with_done_audio:
                    skipped_done += 1
                    continue

                book_ch = by_index.get(ch.chapter_index) or by_title.get(ch.title.strip().lower())
                if book_ch is None:
                    continue

                chunks = chunk_paragraphs(
                    book_ch.paragraphs,
                    pronunciation_dict=pronunciation_dict,
                    max_chars=max_chars,
                    chapter_title=ch.title,
                    include_title=True,
                    multi_voice=True,
                )
                conn.execute("DELETE FROM chunks WHERE chapter_id = ?", (ch.id,))
                for c in chunks:
                    conn.execute(
                        """
                        INSERT INTO chunks(
                            chapter_id, chunk_index, raw_text, normalized_text,
                            pause_after_ms, status, seed, voice_role
                        ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                        """,
                        (
                            ch.id,
                            c.index,
                            c.raw_text,
                            c.normalized_text,
                            c.pause_after_ms,
                            base_seed + ch.chapter_index * 1000 + c.index,
                            c.voice_role,
                        ),
                    )
                    if c.voice_role in ("male", "female", "dialogue"):
                        dialogue_chunks_total += 1
                conn.execute("UPDATE chapters SET multi_voice = 1 WHERE id = ?", (ch.id,))
                converted += 1
                new_chunks_total += len(chunks)

        if converted > 0:
            self.set_meta("multi_voice", "1")

        return RechunkMultiVoiceResult(
            converted_chapters=converted,
            skipped_done_chapters=skipped_done,
            new_chunks=new_chunks_total,
            dialogue_chunks=dialogue_chunks_total,
        )

    def list_chapters(self) -> list[ChapterRow]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    ch.id, ch.chapter_index, ch.title, ch.selected, ch.multi_voice,
                    COUNT(c.id) AS total_chunks,
                    SUM(CASE WHEN c.status = 'done' THEN 1 ELSE 0 END) AS done_chunks,
                    SUM(CASE WHEN c.voice_role IN ('male', 'female') THEN 1 ELSE 0 END) AS dialogue_tagged,
                    SUM(CASE WHEN c.voice_role = 'dialogue' THEN 1 ELSE 0 END) AS dialogue_untagged
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
                multi_voice=bool(r["multi_voice"]),
                dialogue_tagged=int(r["dialogue_tagged"] or 0),
                dialogue_untagged=int(r["dialogue_untagged"] or 0),
            )
            for r in rows
        ]

    def set_chapter_selected(self, chapter_id: int, selected: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chapters SET selected=? WHERE id=?",
                (1 if selected else 0, chapter_id),
            )

    def set_chapters_selected_bulk(self, chapter_ids: list[int], selected: bool) -> None:
        """Zaznacza/odznacza wiele rozdziałów w jednej transakcji SQLite (błyskawiczne dla całych części)."""
        if not chapter_ids:
            return
        val = 1 if selected else 0
        with self._connect() as conn:
            conn.executemany(
                "UPDATE chapters SET selected=? WHERE id=?",
                [(val, int(cid)) for cid in chapter_ids],
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
            voice_role=(r["voice_role"] if "voice_role" in r.keys() and r["voice_role"] else "narrator"),
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

    def set_chunk_voice_role(self, chunk_id: int, voice_role: str, reset_to_pending: bool = True) -> None:
        role = voice_role if voice_role in ("narrator", "male", "female", "dialogue") else "narrator"
        with self._connect() as conn:
            if reset_to_pending:
                conn.execute(
                    "UPDATE chunks SET voice_role=?, status='pending' WHERE id=?",
                    (role, chunk_id),
                )
            else:
                conn.execute(
                    "UPDATE chunks SET voice_role=? WHERE id=?",
                    (role, chunk_id),
                )

    def set_chunks_voice_roles_bulk(
        self, role_map: dict[int, str], reset_changed_to_pending: bool = True
    ) -> int:
        """Aktualizuje role głosowe wielu chunków w jednej transakcji SQLite. Zwraca liczbę zmienionych."""
        if not role_map:
            return 0
        valid = {"narrator", "male", "female", "dialogue"}
        changed = 0
        with self._connect() as conn:
            for cid, raw_role in role_map.items():
                role = raw_role if raw_role in valid else "narrator"
                row = conn.execute("SELECT voice_role FROM chunks WHERE id=?", (int(cid),)).fetchone()
                if row is None:
                    continue
                if row["voice_role"] != role:
                    if reset_changed_to_pending:
                        conn.execute(
                            "UPDATE chunks SET voice_role=?, status='pending' WHERE id=?",
                            (role, int(cid)),
                        )
                    else:
                        conn.execute(
                            "UPDATE chunks SET voice_role=? WHERE id=?",
                            (role, int(cid)),
                        )
                    changed += 1
        return changed

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
