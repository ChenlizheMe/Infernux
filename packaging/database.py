import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import logging


class ProjectDatabase:

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            home_dir = Path.home() / ".infernux"
            home_dir.mkdir(parents=True, exist_ok=True)
            db_path = home_dir / "projects.db"

        self._conn = sqlite3.connect(db_path)
        self._create_table()

    def all_projects(self) -> List[Tuple[str, str, str]]:
        cur = self._conn.execute(
            "SELECT name, created_at, path FROM projects ORDER BY created_at DESC;"
        )
        return cur.fetchall()
    
    def get_project_path(self, name: str) -> str:
        cur = self._conn.execute("SELECT path FROM projects WHERE name = ?;", (name,))
        row = cur.fetchone()
        return row[0] if row else ""

    def add_project(self, name: str, path: str) -> bool:
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO projects (name, created_at, path) VALUES (?, ?, ?);",
                    (name, datetime.now().isoformat(timespec="seconds"), path),
                )
            return True
        except Exception as _exc:
            logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
            return False

    def _create_table(self) -> None:
        with self._conn:
            # Include the path column when creating the table.
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    name       TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    path       TEXT NOT NULL
                );
                """
            )
            # Backfill the path column for legacy databases.
            cur = self._conn.execute("PRAGMA table_info(projects);")
            columns = {row[1] for row in cur.fetchall()}
            if "path" not in columns:
                self._conn.execute("ALTER TABLE projects ADD COLUMN path TEXT NOT NULL DEFAULT '';")

            # Key-value settings table
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    # ── Settings helpers ─────────────────────────────────────────────

    def get_setting(self, key: str, default: str = "") -> str:
        cur = self._conn.execute("SELECT value FROM settings WHERE key = ?;", (key,))
        row = cur.fetchone()
        return row[0] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?);",
                (key, value),
            )

    def delete_project(self, name: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM projects WHERE name = ?;", (name,))
            return self._conn.total_changes > 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
