"""SQLite-backed memory index with FTS5 for relevance-based retrieval.

Memory files (MEMORY.md, daily notes) remain the source of truth.  This
module maintains a lightweight search index alongside them so the agent
only injects the most relevant chunks into each prompt instead of the
entire memory corpus.

Indexing is mtime-gated: files are only re-parsed when they change.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _split_into_chunks(text: str) -> list[str]:
    """Split markdown text into paragraph-level chunks for indexing."""
    raw = re.split(r"\n\s*\n+", text.strip())
    chunks: list[str] = []
    for part in raw:
        p = part.strip()
        if not p or len(p) < 12:
            continue
        # Cap individual chunk size for retrieval quality
        if len(p) > 1200:
            p = p[:1200]
        chunks.append(p)
    return chunks


def _fts_query(text: str) -> str:
    """Build an FTS5 OR query from free-form text, avoiding syntax injection."""
    terms = re.findall(r"[A-Za-z0-9_]{2,}", text)
    if not terms:
        return ""
    # Deduplicate, lowercase, cap at 16 terms
    seen: set[str] = set()
    unique: list[str] = []
    for t in terms:
        low = t.lower()
        if low not in seen:
            seen.add(low)
            unique.append(low)
        if len(unique) >= 16:
            break
    return " OR ".join(unique)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryHit:
    """A single search result from the memory index."""
    source_key: str
    content: str


# ---------------------------------------------------------------------------
# MemoryDB
# ---------------------------------------------------------------------------

class MemoryDB:
    """SQLite + FTS5 index over the workspace memory directory.

    Usage::

        db = MemoryDB(workspace / "memory" / "memory.sqlite3")
        db.index_file("MEMORY.md", workspace / "memory" / "MEMORY.md")
        hits = db.search("docker configuration", limit=8)
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._has_fts: bool | None = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=3.0)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA busy_timeout=3000;")
        return con

    def _ensure_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_sources (
                    source_key  TEXT PRIMARY KEY,
                    mtime_ns    INTEGER NOT NULL,
                    updated_at  TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id           INTEGER PRIMARY KEY,
                    source_key   TEXT NOT NULL,
                    content      TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at   TEXT NOT NULL,
                    UNIQUE (source_key, content_hash)
                )
                """
            )

            # Try to create FTS5 virtual table.  Some Python builds lack it.
            try:
                con.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                    USING fts5(
                        content,
                        source_key,
                        content='memory_entries',
                        content_rowid='id'
                    )
                    """
                )
                con.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memory_entries BEGIN
                        INSERT INTO memory_fts(rowid, content, source_key)
                        VALUES (new.id, new.content, new.source_key);
                    END;
                    """
                )
                con.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memory_entries BEGIN
                        INSERT INTO memory_fts(memory_fts, rowid, content, source_key)
                        VALUES ('delete', old.id, old.content, old.source_key);
                    END;
                    """
                )
                con.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE ON memory_entries BEGIN
                        INSERT INTO memory_fts(memory_fts, rowid, content, source_key)
                        VALUES ('delete', old.id, old.content, old.source_key);
                        INSERT INTO memory_fts(rowid, content, source_key)
                        VALUES (new.id, new.content, new.source_key);
                    END;
                    """
                )
                self._has_fts = True
            except sqlite3.OperationalError:
                self._has_fts = False
                logger.debug("FTS5 not available in this Python build; falling back to LIKE")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @staticmethod
    def _get_mtime_ns(path: Path) -> int:
        try:
            return path.stat().st_mtime_ns if path.exists() else 0
        except Exception:
            return 0

    def index_file(self, source_key: str, path: Path) -> None:
        """Index (or re-index) a file.  Skips if mtime hasn't changed."""
        mtime_ns = self._get_mtime_ns(path)
        now = _utc_now_iso()

        with self._connect() as con:
            row = con.execute(
                "SELECT mtime_ns FROM memory_sources WHERE source_key = ?",
                (source_key,),
            ).fetchone()
            if row and int(row[0]) == int(mtime_ns):
                return  # unchanged

            # Wipe old entries for this source
            con.execute(
                "DELETE FROM memory_entries WHERE source_key = ?",
                (source_key,),
            )

            text = ""
            try:
                if path.exists() and path.is_file():
                    text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = ""

            for chunk in _split_into_chunks(text) if text else []:
                con.execute(
                    """
                    INSERT OR IGNORE INTO memory_entries
                        (source_key, content, content_hash, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (source_key, chunk, _hash_text(chunk), now),
                )

            con.execute(
                """
                INSERT INTO memory_sources (source_key, mtime_ns, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(source_key)
                DO UPDATE SET mtime_ns = excluded.mtime_ns,
                              updated_at = excluded.updated_at
                """,
                (source_key, int(mtime_ns), now),
            )

        logger.debug(f"Indexed memory file {source_key}")

    def index_directory(self, memory_dir: Path) -> None:
        """Index all .md files in the memory directory."""
        if not memory_dir.is_dir():
            return
        for md_file in memory_dir.glob("*.md"):
            self.index_file(md_file.name, md_file)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        *,
        limit: int = 8,
        exclude_sources: set[str] | None = None,
    ) -> list[MemoryHit]:
        """Search indexed memory for chunks relevant to *query_text*.

        Args:
            query_text: Free-form query (e.g. the user's message).
            limit: Maximum results to return.
            exclude_sources: Source keys to skip (e.g. today's note which
                is always injected separately).

        Returns:
            Ranked list of ``MemoryHit`` results.
        """
        q = _fts_query(query_text)
        if not q:
            return []
        limit = max(int(limit), 0)
        if limit <= 0:
            return []

        exclude = exclude_sources or set()

        with self._connect() as con:
            if self._has_fts:
                try:
                    rows = con.execute(
                        """
                        SELECT me.source_key, me.content
                        FROM memory_fts
                        JOIN memory_entries me ON memory_fts.rowid = me.id
                        WHERE memory_fts MATCH ?
                        ORDER BY bm25(memory_fts)
                        LIMIT ?
                        """,
                        (q, limit + len(exclude)),
                    ).fetchall()
                    hits = [
                        MemoryHit(source_key=r[0], content=r[1])
                        for r in rows
                        if r[0] not in exclude
                    ]
                    return hits[:limit]
                except sqlite3.OperationalError:
                    pass  # fall through to LIKE

            # Fallback: LIKE search
            like = "%" + query_text.strip()[:200] + "%"
            rows = con.execute(
                """
                SELECT source_key, content
                FROM memory_entries
                WHERE content LIKE ?
                LIMIT ?
                """,
                (like, limit + len(exclude)),
            ).fetchall()
            hits = [
                MemoryHit(source_key=r[0], content=r[1])
                for r in rows
                if r[0] not in exclude
            ]
            return hits[:limit]
