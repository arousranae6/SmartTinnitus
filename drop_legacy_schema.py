"""Permanently delete the archived SmartTinnitus legacy schema.

DO NOT run this until the PostgreSQL app has been fully tested after archiving.

Run intentionally with:
    python drop_legacy_schema.py --confirm
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import psycopg
except ImportError as exc:
    raise SystemExit(
        "psycopg is missing. Run: python -m pip install -r requirements-postgres.txt"
    ) from exc


ROOT = Path(__file__).resolve().parent


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


def main():
    if "--confirm" not in sys.argv:
        raise SystemExit(
            "Nothing deleted. When you are absolutely sure, run:\n"
            "python drop_legacy_schema.py --confirm"
        )

    load_env()

    database_url = os.getenv("DATABASE_URL", "").strip()

    if not database_url:
        raise SystemExit("DATABASE_URL is empty.")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS legacy CASCADE")
        conn.commit()

    print("legacy schema permanently deleted.")


if __name__ == "__main__":
    main()
