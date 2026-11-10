"""Copy the existing SmartTinnitus SQLite database into PostgreSQL.

This script is deliberately separate from the Streamlit app. It reads the
actual tables/columns in smarttinnitus.db, creates matching PostgreSQL tables,
and copies the existing rows.

It does NOT change the Streamlit application to PostgreSQL. That activation is
performed only after the copied database has been verified.

Usage (PowerShell):
    $env:DATABASE_URL="postgresql://..."
    python migrate_sqlite_to_postgres.py
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

try:
    import psycopg
    from psycopg import sql
except ImportError as exc:
    raise SystemExit(
        "PostgreSQL driver missing. Run: python -m pip install -r requirements-postgres.txt"
    ) from exc


ROOT = Path(__file__).resolve().parent
SQLITE_PATH = ROOT / "smarttinnitus.db"
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()


def sqlite_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [row[0] for row in rows]


def sqlite_columns(conn: sqlite3.Connection, table: str):
    escaped = table.replace('"', '""')
    return conn.execute(f'PRAGMA table_info("{escaped}")').fetchall()


def map_type(sqlite_type: str) -> str:
    t = (sqlite_type or "").upper()
    if "INT" in t:
        return "BIGINT"
    if any(token in t for token in ("REAL", "FLOA", "DOUB", "NUM")):
        return "DOUBLE PRECISION"
    if "BLOB" in t:
        return "BYTEA"
    if "BOOL" in t:
        return "BOOLEAN"
    # Keep dates/timestamps as text for the first migration so the existing
    # application's stored values are preserved exactly.
    return "TEXT"


def create_table(pg, table: str, columns):
    definitions = []
    pk_columns = []

    for _, name, col_type, not_null, default_value, pk_order in columns:
        parts = [sql.Identifier(name), sql.SQL(map_type(col_type))]
        if not_null:
            parts.append(sql.SQL("NOT NULL"))
        definitions.append(sql.SQL(" ").join(parts))
        if pk_order:
            pk_columns.append((pk_order, name))

    if pk_columns:
        ordered = [name for _, name in sorted(pk_columns)]
        definitions.append(
            sql.SQL("PRIMARY KEY ({})").format(
                sql.SQL(", ").join(sql.Identifier(name) for name in ordered)
            )
        )

    statement = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
        sql.Identifier(table),
        sql.SQL(", ").join(definitions),
    )
    pg.execute(statement)


def copy_table(sqlite_conn, pg, table: str, columns):
    names = [col[1] for col in columns]
    escaped = table.replace('"', '""')
    rows = sqlite_conn.execute(f'SELECT * FROM "{escaped}"').fetchall()

    if not rows:
        return 0

    insert = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING").format(
        sql.Identifier(table),
        sql.SQL(", ").join(sql.Identifier(name) for name in names),
        sql.SQL(", ").join(sql.Placeholder() for _ in names),
    )

    with pg.cursor() as cur:
        cur.executemany(insert, rows)

    return len(rows)


def main():
    if not SQLITE_PATH.exists():
        raise SystemExit(f"SQLite database not found: {SQLITE_PATH}")

    if not DATABASE_URL:
        raise SystemExit(
            "DATABASE_URL is empty. Set the PostgreSQL connection string first."
        )

    print(f"Source SQLite database: {SQLITE_PATH}")
    print("Connecting to PostgreSQL...")

    sqlite_conn = sqlite3.connect(SQLITE_PATH)

    try:
        tables = sqlite_tables(sqlite_conn)
        if not tables:
            raise SystemExit("No application tables were found in smarttinnitus.db")

        with psycopg.connect(DATABASE_URL) as pg:
            for table in tables:
                columns = sqlite_columns(sqlite_conn, table)
                create_table(pg, table, columns)
                copied = copy_table(sqlite_conn, pg, table, columns)
                print(f"{table}: {copied} row(s) copied")

            pg.commit()

        print("\nMigration copy completed successfully.")
        print("IMPORTANT: Do not delete smarttinnitus.db yet.")
        print("Next step: verify PostgreSQL, then activate it in the app.")

    finally:
        sqlite_conn.close()


if __name__ == "__main__":
    main()
