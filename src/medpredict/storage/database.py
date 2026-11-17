"""SQLite results store."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

TABLES = ("meta", "model_results", "calibration_curves", "subgroups", "shift_results", "predictions", "explanations")


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value_json TEXT, updated_at TEXT)")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def reset(self) -> None:
        with self.connect() as c:
            for t in TABLES:
                if t != "meta":
                    c.execute(f"DROP TABLE IF EXISTS {t}")
            c.execute("DELETE FROM meta")

    def put_meta(self, key: str, value: Any) -> None:
        with self.connect() as c:
            c.execute("INSERT OR REPLACE INTO meta VALUES (?, ?, ?)", (key, json.dumps(value, default=str), utcnow()))

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self.connect() as c:
            r = c.execute("SELECT value_json FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def write_df(self, table: str, df: pd.DataFrame) -> None:
        with self.connect() as c:
            df.to_sql(table, c, if_exists="append", index=False)

    def query_df(self, sql: str, params: tuple | list = ()) -> pd.DataFrame:
        with self.connect() as c:
            return pd.read_sql_query(sql, c, params=params)
