"""Central database gateway for SmartTinnitus.

Supports:
- SQLite fallback for local development when DATABASE_URL is not set.
- PostgreSQL/Supabase when DATABASE_URL is set.

The PostgreSQL adapter keeps the current SmartTinnitus service/UI code working
while the project is progressively moved away from SQLite-specific SQL.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from database import get_connection as _sqlite_get_connection
from database import initialize_database as _sqlite_initialize_database


ROOT = Path(__file__).resolve().parents[1]


def _load_local_env():
    """Load DATABASE_URL from .env without requiring python-dotenv."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # Never prevent SQLite fallback because of a malformed local env file.
        pass


_load_local_env()


def get_database_url() -> str | None:
    value = os.getenv("DATABASE_URL", "").strip()
    return value or None


def get_database_provider() -> str:
    return "postgresql" if get_database_url() else "sqlite"


def is_postgresql_enabled() -> bool:
    return get_database_provider() == "postgresql"


class HybridRow:
    """Tuple-like row that also supports row['column'] access."""

    __slots__ = ("_values", "_names", "_mapping")

    def __init__(self, values, names):
        self._values = tuple(values)
        self._names = tuple(names)
        self._mapping = dict(zip(self._names, self._values))

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._mapping[key]
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def keys(self):
        return self._mapping.keys()

    def items(self):
        return self._mapping.items()

    def values(self):
        return self._mapping.values()

    def __repr__(self):
        return repr(self._values)


def _replace_qmark_placeholders(query: str) -> str:
    """Convert SQLite ? placeholders to PostgreSQL %s outside string literals."""
    out = []
    in_single = False
    in_double = False
    i = 0

    while i < len(query):
        ch = query[i]

        if ch == "'" and not in_double:
            # SQL escaped single quote: ''
            if in_single and i + 1 < len(query) and query[i + 1] == "'":
                out.extend(["'", "'"])
                i += 2
                continue
            in_single = not in_single
            out.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
        elif ch == "?" and not in_single and not in_double:
            out.append("%s")
        else:
            out.append(ch)
        i += 1

    return "".join(out)


def _normalize_postgres_sql(query: str) -> str:
    q = query

    # Common SQLite syntax that PostgreSQL does not accept.
    q = re.sub(r"\bCOLLATE\s+NOCASE\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\bdatetime\s*\(\s*'now'\s*\)", "CURRENT_TIMESTAMP", q, flags=re.IGNORECASE)

    # SQLite often uses datetime(column_name) for sorting/comparison.
    # PostgreSQL has no datetime() function; migrated timestamp values may
    # still be stored as text, so cast them explicitly.
    q = re.sub(
        r"\bdatetime\s*\(\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*\)",
        r"CAST(\1 AS TIMESTAMP)",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(
        r"\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b",
        "BIGSERIAL PRIMARY KEY",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"\bAUTOINCREMENT\b", "", q, flags=re.IGNORECASE)

    # INSERT OR IGNORE -> INSERT ... ON CONFLICT DO NOTHING
    if re.search(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", q, flags=re.IGNORECASE):
        q = re.sub(
            r"\bINSERT\s+OR\s+IGNORE\s+INTO\b",
            "INSERT INTO",
            q,
            flags=re.IGNORECASE,
        )
        stripped = q.rstrip().rstrip(";")
        if not re.search(r"\bON\s+CONFLICT\b", stripped, flags=re.IGNORECASE):
            stripped += " ON CONFLICT DO NOTHING"
        q = stripped

    return _replace_qmark_placeholders(q)


def _insert_table_name(query: str) -> str | None:
    match = re.match(
        r"\s*INSERT\s+(?:OR\s+IGNORE\s+)?INTO\s+([A-Za-z_][A-Za-z0-9_]*)",
        query,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1) if match else None


class PostgresCursorAdapter:
    def __init__(self, connection, cursor=None):
        self._connection = connection
        self._cursor = cursor or connection._raw.cursor()
        self._synthetic_rows = None
        self._synthetic_description = None
        self._synthetic_index = 0
        self._lastrowid = None

    @property
    def description(self):
        if self._synthetic_description is not None:
            return self._synthetic_description
        return self._cursor.description

    @property
    def rowcount(self):
        if self._synthetic_rows is not None:
            return len(self._synthetic_rows)
        return self._cursor.rowcount

    @property
    def lastrowid(self):
        return self._lastrowid

    def _reset_synthetic(self):
        self._synthetic_rows = None
        self._synthetic_description = None
        self._synthetic_index = 0

    def _handle_pragma_table_info(self, query: str):
        match = re.match(
            r'\s*PRAGMA\s+table_info\s*\(\s*["\']?([A-Za-z_][A-Za-z0-9_]*)["\']?\s*\)\s*;?\s*$',
            query,
            flags=re.IGNORECASE,
        )
        if not match:
            return False

        table = match.group(1)
        sql = """
            SELECT
                c.ordinal_position - 1 AS cid,
                c.column_name AS name,
                c.data_type AS type,
                CASE WHEN c.is_nullable = 'NO' THEN 1 ELSE 0 END AS notnull,
                c.column_default AS dflt_value,
                CASE WHEN tc.constraint_type = 'PRIMARY KEY' THEN 1 ELSE 0 END AS pk
            FROM information_schema.columns c
            LEFT JOIN information_schema.key_column_usage kcu
              ON c.table_schema = kcu.table_schema
             AND c.table_name = kcu.table_name
             AND c.column_name = kcu.column_name
            LEFT JOIN information_schema.table_constraints tc
              ON kcu.constraint_name = tc.constraint_name
             AND kcu.table_schema = tc.table_schema
            WHERE c.table_schema = 'public'
              AND c.table_name = %s
            ORDER BY c.ordinal_position
        """
        self._cursor.execute(sql, (table,))
        rows = self._cursor.fetchall()
        names = [d.name if hasattr(d, "name") else d[0] for d in self._cursor.description]
        self._synthetic_rows = [HybridRow(row, names) for row in rows]
        self._synthetic_description = self._cursor.description
        return True

    def _handle_sqlite_master(self, query: str, params):
        if "sqlite_master" not in query.lower():
            return False

        # Current SmartTinnitus usage is table-existence checking.
        table_name = None
        if params:
            table_name = params[0]
        else:
            m = re.search(r"name\s*=\s*['\"]([^'\"]+)['\"]", query, flags=re.IGNORECASE)
            if m:
                table_name = m.group(1)

        sql = """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = %s
            LIMIT 1
        """
        self._cursor.execute(sql, (table_name,))
        rows = self._cursor.fetchall()
        self._synthetic_rows = [HybridRow(row, ["?column?"]) for row in rows]
        self._synthetic_description = self._cursor.description
        return True

    def execute(self, query, params=None):
        self._reset_synthetic()
        self._lastrowid = None

        query = str(query)
        params = tuple(params or ())

        if self._handle_pragma_table_info(query):
            return self

        if self._handle_sqlite_master(query, params):
            return self

        table_name = _insert_table_name(query)
        normalized = _normalize_postgres_sql(query)

        # Preserve SQLite cursor.lastrowid behavior for normal inserts.
        wants_lastrowid = bool(table_name) and not re.search(
            r"\bRETURNING\b", normalized, flags=re.IGNORECASE
        )

        if wants_lastrowid:
            normalized = normalized.rstrip().rstrip(";") + " RETURNING id"

        try:
            self._cursor.execute(normalized, params)
            if wants_lastrowid:
                returned = self._cursor.fetchone()
                if returned is not None:
                    self._lastrowid = int(returned[0])
        except Exception:
            # Some tables may not have an id column. Retry without RETURNING.
            if wants_lastrowid:
                self._connection._raw.rollback()
                self._cursor = self._connection._raw.cursor()
                normalized = re.sub(
                    r"\s+RETURNING\s+id\s*$",
                    "",
                    normalized,
                    flags=re.IGNORECASE,
                )
                self._cursor.execute(normalized, params)
            else:
                raise

        return self

    def executemany(self, query, param_list):
        self._reset_synthetic()
        normalized = _normalize_postgres_sql(str(query))
        self._cursor.executemany(normalized, param_list)
        return self

    def _column_names(self):
        desc = self.description
        if not desc:
            return []
        return [d.name if hasattr(d, "name") else d[0] for d in desc]

    def fetchone(self):
        if self._synthetic_rows is not None:
            if self._synthetic_index >= len(self._synthetic_rows):
                return None
            row = self._synthetic_rows[self._synthetic_index]
            self._synthetic_index += 1
            return row

        row = self._cursor.fetchone()
        if row is None:
            return None
        return HybridRow(row, self._column_names())

    def fetchall(self):
        if self._synthetic_rows is not None:
            remaining = self._synthetic_rows[self._synthetic_index :]
            self._synthetic_index = len(self._synthetic_rows)
            return remaining

        rows = self._cursor.fetchall()
        names = self._column_names()
        return [HybridRow(row, names) for row in rows]

    def fetchmany(self, size=None):
        rows = self._cursor.fetchmany(size) if size is not None else self._cursor.fetchmany()
        names = self._column_names()
        return [HybridRow(row, names) for row in rows]

    def close(self):
        try:
            self._cursor.close()
        except Exception:
            pass

    def __iter__(self):
        while True:
            row = self.fetchone()
            if row is None:
                break
            yield row


class PostgresConnectionAdapter:
    def __init__(self, raw_connection):
        self._raw = raw_connection

    def cursor(self):
        return PostgresCursorAdapter(self)

    def execute(self, query, params=None):
        cur = self.cursor()
        return cur.execute(query, params)

    def executemany(self, query, param_list):
        cur = self.cursor()
        return cur.executemany(query, param_list)

    def commit(self):
        return self._raw.commit()

    def rollback(self):
        return self._raw.rollback()

    def close(self):
        return self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.close()


def _postgres_connection():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL is configured but psycopg is not installed. "
            "Run: python -m pip install -r requirements-postgres.txt"
        ) from exc

    raw = psycopg.connect(get_database_url())
    return PostgresConnectionAdapter(raw)


def get_connection():
    if is_postgresql_enabled():
        return _postgres_connection()
    return _sqlite_get_connection()


def initialize_database():
    if not is_postgresql_enabled():
        return _sqlite_initialize_database()

    # PostgreSQL schema is created by the migration. At runtime, simply verify
    # that the migrated core table exists.
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = 'patients'
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            raise RuntimeError(
                "PostgreSQL is connected, but the SmartTinnitus schema was not found. "
                "Run migrate_sqlite_to_postgres.py first."
            )
        return True
    finally:
        conn.close()


@contextmanager
def connection() -> Iterator:
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction() -> Iterator:
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
