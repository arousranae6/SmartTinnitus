"""Prepare the final SmartTinnitus PostgreSQL schema safely.

This script:
- creates clinical_reports
- creates useful indexes
- reports legacy table row counts
- DOES NOT drop legacy tables

Run:
    python prepare_final_schema.py
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import psycopg
except ImportError as exc:
    raise SystemExit(
        "psycopg is missing. Run: python -m pip install -r requirements-postgres.txt"
    ) from exc


ROOT = Path(__file__).resolve().parent

LEGACY_TABLES = [
    "adaptive_sessions",
    "adaptive_session_trials",
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


def main():
    load_env()

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("DATABASE_URL is empty.")

    print("Connecting to PostgreSQL...")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS clinical_reports (
                    id BIGSERIAL PRIMARY KEY,
                    patient_id BIGINT NOT NULL,
                    generated_by_user_id BIGINT,
                    generated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    report_version VARCHAR(32) NOT NULL DEFAULT '1.0',
                    filename TEXT NOT NULL,
                    storage_path TEXT,
                    status VARCHAR(32) NOT NULL DEFAULT 'generated',
                    file_hash VARCHAR(64)
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_clinical_reports_patient_id
                ON clinical_reports (patient_id)
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_clinical_reports_generated_at
                ON clinical_reports (generated_at DESC)
                """
            )

            print("clinical_reports: ready")

            print("\nLegacy table audit (nothing will be deleted):")

            for table in LEGACY_TABLES:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                          AND table_name = %s
                    )
                    """,
                    (table,),
                )
                exists = bool(cur.fetchone()[0])

                if not exists:
                    print(f"- {table}: not present")
                    continue

                cur.execute(f'SELECT COUNT(*) FROM "{table}"')
                count = int(cur.fetchone()[0])
                print(f"- {table}: {count} row(s) — KEEP FOR NOW")

        conn.commit()

    print("\nFinal schema preparation completed.")
    print("No legacy tables were deleted.")
    print(
        "Reason: the current app.py still contains dormant references to "
        "sessions, audiologist_decisions and personalization_reviews."
    )


if __name__ == "__main__":
    main()
