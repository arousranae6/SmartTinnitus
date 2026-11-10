"""Archive obsolete SmartTinnitus PostgreSQL tables into a separate legacy schema.

This is safer than dropping them immediately:
- final application tables remain in public
- old data is preserved in legacy
- the operation is reversible with restore_legacy_tables.py

Run:
    python archive_legacy_tables.py
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
            cur.execute("CREATE SCHEMA IF NOT EXISTS legacy")

            print("\nArchiving legacy tables:")

            for table in LEGACY_TABLES:
                if table_exists(cur, "legacy", table):
                    print(f"- {table}: already archived")
                    continue

                if not table_exists(cur, "public", table):
                    print(f"- {table}: not found in public")
                    continue

                cur.execute(
                    sql.SQL("ALTER TABLE public.{} SET SCHEMA legacy").format(
                        sql.Identifier(table)
                    )
                )

                print(f"- {table}: moved public → legacy")

        conn.commit()

    print("\nLegacy archive completed successfully.")
    print("No data was deleted.")
    print("Your final application tables remain in the public schema.")


if __name__ == "__main__":
    main()
