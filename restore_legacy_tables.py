"""Restore archived SmartTinnitus tables from legacy back to public.

Use only if testing after archive reveals a dependency on an old table.

Run:
    python restore_legacy_tables.py
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import psycopg
    from psycopg import sql
except ImportError as exc:
    raise SystemExit(
        "psycopg is missing. Run: python -m pip install -r requirements-postgres.txt"
    ) from exc


ROOT = Path(__file__).resolve().parent

LEGACY_TABLES = [
    "adaptive_session_trials",
    "adaptive_sessions",
    "audiologist_decisions",
    "personalization_reviews",
    "sessions",
]


def load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def table_exists(cur, schema_name, table_name):
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
        )
        """,
        (schema_name, table_name),
    )
    return bool(cur.fetchone()[0])


def main():
    load_env()

    database_url = os.getenv("DATABASE_URL", "").strip()

    if not database_url:
        raise SystemExit("DATABASE_URL is empty.")

    print("Connecting to PostgreSQL...")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for table in LEGACY_TABLES:
                if not table_exists(cur, "legacy", table):
                    print(f"- {table}: not present in legacy")
                    continue

                if table_exists(cur, "public", table):
                    print(f"- {table}: already exists in public; skipped")
                    continue

                cur.execute(
                    sql.SQL("ALTER TABLE legacy.{} SET SCHEMA public").format(
                        sql.Identifier(table)
                    )
                )
                print(f"- {table}: restored legacy → public")

        conn.commit()

    print("\nRestore completed.")


if __name__ == "__main__":
    main()
