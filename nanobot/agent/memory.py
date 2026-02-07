"""Memory system for persistent agent memory.

Supports daily notes (memory/YYYY-MM-DD.md) and long-term memory (MEMORY.md).
Uses an SQLite FTS5 index for relevance-based retrieval so only the most
useful chunks are injected into each prompt.
"""

from pathlib import Path
from datetime import datetime

from loguru import logger

from nanobot.agent.memory_db import MemoryDB
from nanobot.utils.helpers import ensure_dir, today_date


# Max chars of FTS-retrieved context to inject into the prompt
_MAX_RETRIEVED_CHARS = 6000


class MemoryStore:
    """
    Memory system for the agent.

    Stores markdown files in ``workspace/memory/`` and maintains an FTS5
    search index alongside them for efficient retrieval.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self._db = MemoryDB(self.memory_dir / "memory.sqlite3")

    # ------------------------------------------------------------------
    # File I/O (unchanged public API)
    # ------------------------------------------------------------------

    def get_today_file(self) -> Path:
        """Get path to today's memory file."""
        return self.memory_dir / f"{today_date()}.md"

    def read_today(self) -> str:
        """Read today's memory notes."""
        today_file = self.get_today_file()
        if today_file.exists():
            return today_file.read_text(encoding="utf-8")
        return ""

    def append_today(self, content: str) -> None:
        """Append content to today's memory notes."""
        today_file = self.get_today_file()

        if today_file.exists():
            existing = today_file.read_text(encoding="utf-8")
            content = existing + "\n" + content
        else:
            header = f"# {today_date()}\n\n"
            content = header + content

        today_file.write_text(content, encoding="utf-8")

    def read_long_term(self) -> str:
        """Read long-term memory (MEMORY.md)."""
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        """Write to long-term memory (MEMORY.md)."""
        self.memory_file.write_text(content, encoding="utf-8")

    def get_recent_memories(self, days: int = 7) -> str:
        """Get memories from the last N days."""
        from datetime import timedelta

        memories = []
        today = datetime.now().date()

        for i in range(days):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            file_path = self.memory_dir / f"{date_str}.md"

            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                memories.append(content)

        return "\n\n---\n\n".join(memories)

    def list_memory_files(self) -> list[Path]:
        """List all memory files sorted by date (newest first)."""
        if not self.memory_dir.exists():
            return []

        files = list(self.memory_dir.glob("????-??-??.md"))
        return sorted(files, reverse=True)

    # ------------------------------------------------------------------
    # FTS-powered context retrieval
    # ------------------------------------------------------------------

    def _ensure_indexed(self) -> None:
        """(Re-)index any changed memory files into the FTS database."""
        self._db.index_directory(self.memory_dir)

    def get_memory_context(self, query: str | None = None) -> str:
        """Build memory context for the agent prompt.

        If *query* is provided, uses FTS to retrieve only the most
        relevant chunks across all memory files.  Today's notes are
        always included in full (they're current context).

        Falls back to the legacy behaviour (inject full MEMORY.md +
        today's notes) when no query is given or the index returns
        nothing.

        Args:
            query: The user's current message (used for relevance search).

        Returns:
            Formatted memory context string.
        """
        # Always re-index changed files (mtime-gated, nearly free)
        self._ensure_indexed()

        parts: list[str] = []
        today_filename = f"{today_date()}.md"

        # --- Today's notes (always included in full) ---
        today = self.read_today()
        if today:
            parts.append(f"## Today's Notes\n{today}")

        # --- Relevant memory via FTS ---
        if query:
            hits = self._db.search(
                query,
                limit=8,
                exclude_sources={today_filename},  # already included above
            )
            if hits:
                retrieved: list[str] = []
                total_chars = 0
                for hit in hits:
                    if total_chars + len(hit.content) > _MAX_RETRIEVED_CHARS:
                        break
                    retrieved.append(f"[{hit.source_key}] {hit.content}")
                    total_chars += len(hit.content)

                if retrieved:
                    parts.append(
                        "## Relevant Memory\n" + "\n\n".join(retrieved)
                    )
                    return "\n\n".join(parts)

        # --- Fallback: inject full MEMORY.md (legacy behaviour) ---
        long_term = self.read_long_term()
        if long_term:
            parts.insert(0, f"## Long-term Memory\n{long_term}")

        return "\n\n".join(parts) if parts else ""
