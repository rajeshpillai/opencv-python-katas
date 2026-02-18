"""
SQLite database initialization and helpers.
Tables are created on startup; katas are seeded from JSON files.
"""

import sqlite3
import json
import os
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "opencv-katas.db"
KATAS_DIR = Path(__file__).parent.parent / "data" / "katas"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            email       TEXT    UNIQUE NOT NULL,
            password_hash TEXT  NOT NULL,
            created_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS katas (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            slug        TEXT    UNIQUE NOT NULL,
            title       TEXT    NOT NULL,
            level       TEXT    NOT NULL,
            concepts    TEXT    NOT NULL,
            content_json TEXT   NOT NULL
        );

        CREATE TABLE IF NOT EXISTS user_progress (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            kata_id     INTEGER NOT NULL REFERENCES katas(id),
            completed_at TEXT   DEFAULT (datetime('now')),
            UNIQUE(user_id, kata_id)
        );

        CREATE TABLE IF NOT EXISTS user_code_versions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            kata_id     INTEGER NOT NULL REFERENCES katas(id),
            code        TEXT    NOT NULL,
            saved_at    TEXT    DEFAULT (datetime('now'))
        );
    """)

    conn.commit()
    _seed_katas(conn)
    conn.close()


def _seed_katas(conn: sqlite3.Connection) -> None:
    """Load kata JSON files into the katas table (upsert by slug)."""
    if not KATAS_DIR.exists():
        return

    cur = conn.cursor()
    for json_file in sorted(KATAS_DIR.glob("*.json")):
        with open(json_file, "r") as f:
            data = json.load(f)

        slug = data["slug"]
        title = data["title"]
        level = data["level"]
        concepts = json.dumps(data.get("concepts", []))
        content_json = json.dumps(data)

        cur.execute("""
            INSERT INTO katas (slug, title, level, concepts, content_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                title        = excluded.title,
                level        = excluded.level,
                concepts     = excluded.concepts,
                content_json = excluded.content_json
        """, (slug, title, level, concepts, content_json))

    conn.commit()
