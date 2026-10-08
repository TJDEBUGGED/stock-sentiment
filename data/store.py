"""
store.py — Tiny SQLite store for scored headlines.

Yahoo only returns the most recent handful of headlines per request. By saving
every scored headline (de-duplicated by uuid), the sentiment history grows each
time you use the app, which is what makes a trend line possible over time.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "sentiment.db"


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS headlines (
            uuid         TEXT PRIMARY KEY,
            ticker       TEXT NOT NULL,
            title        TEXT NOT NULL,
            publisher    TEXT,
            published_at TEXT,
            link         TEXT,
            score        REAL NOT NULL,
            label        TEXT NOT NULL
        )
        """
    )
    return conn


def save_headlines(ticker: str, scored: list[dict], db_path: Path = DB_PATH) -> None:
    # `with conn` commits but does not close; closing() releases the file (matters on Windows).
    with closing(_connect(db_path)) as conn, conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO headlines
                (uuid, ticker, title, publisher, published_at, link, score, label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    a["uuid"], ticker, a["title"], a.get("publisher"),
                    a.get("published_at"), a.get("link"), a["score"], a["label"],
                )
                for a in scored
            ],
        )


def load_headlines(ticker: str, db_path: Path = DB_PATH) -> list[dict]:
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM headlines WHERE ticker = ? ORDER BY published_at DESC",
            (ticker,),
        ).fetchall()
    return [dict(r) for r in rows]
