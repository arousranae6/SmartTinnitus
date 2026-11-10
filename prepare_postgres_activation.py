from __future__ import annotations

import os
from pathlib import Path

try:
    import psycopg
    from psycopg import sql
except ImportError as exc:
    raise SystemExit(
        "PostgreSQL driver missing. Run: python -m pip install -r requirements-postgres.txt"
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
    load_env()

    database_url = os.getenv("DATABASE_URL", "").strip()

    if not database_url:
        raise SystemExit(
            "DATABASE_URL is empty. Put your Supabase PostgreSQL URL in .env "
            "or set it in PowerShell."
        )

    print("Connecting to PostgreSQL...")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:

            # Find all public tables with an integer/bigint id column.
            cur.execute(
                """
                SELECT
                    c.table_name,
                    c.data_type
                FROM information_schema.columns c
                WHERE c.table_schema = 'public'
                  AND c.column_name = 'id'
                  AND c.data_type IN ('integer', 'bigint', 'smallint')
                ORDER BY c.table_name
                """
            )

            tables = cur.fetchall()

            if not tables:
                print("No tables with an integer id column were found.")
                return

            for table_name, _data_type in tables:
                seq_name = f"{table_name}_id_seq"

                # CREATE SEQUENCE IF NOT EXISTS
                cur.execute(
                    sql.SQL("CREATE SEQUENCE IF NOT EXISTS {}").format(
                        sql.Identifier(seq_name)
                    )
                )

                # Set id default to nextval('sequence_name'::regclass)
                cur.execute(
                    sql.SQL(
                        "ALTER TABLE {} ALTER COLUMN id "
                        "SET DEFAULT nextval({}::regclass)"
                    ).format(
                        sql.Identifier(table_name),
                        sql.Literal(seq_name),
                    )
                )

                # Sequence belongs to table.id
                cur.execute(
                    sql.SQL("ALTER SEQUENCE {} OWNED BY {}.id").format(
                        sql.Identifier(seq_name),
                        sql.Identifier(table_name),
                    )
                )

                # Find current maximum id
                cur.execute(
                    sql.SQL("SELECT COALESCE(MAX(id), 0) FROM {}").format(
                        sql.Identifier(table_name)
                    )
                )

                max_id = int(cur.fetchone()[0] or 0)

                # Advance sequence so next inserted row gets a new id
                if max_id > 0:
                    cur.execute(
                        sql.SQL(
                            "SELECT setval({}::regclass, {}, true)"
                        ).format(
                            sql.Literal(seq_name),
                            sql.Literal(max_id),
                        )
                    )
                else:
                    cur.execute(
                        sql.SQL(
                            "SELECT setval({}::regclass, 1, false)"
                        ).format(
                            sql.Literal(seq_name)
                        )
                    )

                print(f"{table_name}: next id prepared after {max_id}")

            # Uniqueness expected by SmartTinnitus
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ux_patients_patient_code
                ON patients (patient_code)
                """
            )

            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ux_users_username_lower
                ON users (LOWER(username))
                """
            )

        conn.commit()

    print()
    print("PostgreSQL activation preparation completed successfully.")
    print("Your migrated data was preserved.")
    print("Next step: start SmartTinnitus and test login.")


if __name__ == "__main__":
    main()
