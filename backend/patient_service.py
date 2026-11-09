"""Patient-facing backend queries.

This is intentionally UI-agnostic. Streamlit pages consume dictionaries from
this service rather than owning new SQL. It currently uses the project's
existing database.get_connection() adapter, so the rest of the application
continues to work unchanged while the backend is migrated incrementally.
"""
from __future__ import annotations

from typing import Any

from database import get_connection


def _table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def list_patients() -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, patient_code, age, sex, created_at
            FROM patients
            ORDER BY patient_code COLLATE NOCASE
            """
        ).fetchall()
        return [
            {
                "id": row[0],
                "patient_code": row[1],
                "age": row[2],
                "sex": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_patient(patient_id: int) -> dict[str, Any] | None:
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT id, patient_code, age, sex, created_at
            FROM patients
            WHERE id=?
            LIMIT 1
            """,
            (patient_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "patient_code": row[1],
            "age": row[2],
            "sex": row[3],
            "created_at": row[4],
        }
    finally:
        conn.close()


def get_patient_id_by_code(patient_code: str) -> int | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM patients WHERE patient_code=? LIMIT 1",
            (patient_code,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _latest_review_status(conn, patient_id: int, pathway: str) -> str:
    if not _table_exists(conn, "personalization_reviews"):
        return "—"
    row = conn.execute(
        """
        SELECT status
        FROM personalization_reviews
        WHERE patient_id=? AND pathway=?
        ORDER BY id DESC
        LIMIT 1
        """,
        (patient_id, pathway),
    ).fetchone()
    return row[0] if row and row[0] else "—"


def get_patient_summary(patient_id: int) -> dict[str, Any] | None:
    """Return the compact patient record used by professional workspaces."""
    conn = get_connection()
    try:
        patient = conn.execute(
            "SELECT id, patient_code, age, sex, created_at FROM patients WHERE id=?",
            (patient_id,),
        ).fetchone()
        if patient is None:
            return None

        profile = conn.execute(
            """
            SELECT estimated_pitch_hz, loudness_relative, created_at
            FROM tinnitus_profiles
            WHERE patient_id=?
            ORDER BY id DESC LIMIT 1
            """,
            (patient_id,),
        ).fetchone()
        if profile is None:
            assessment_status = "Not started"
        elif profile[0] is not None and profile[1] is not None:
            assessment_status = "Complete"
        else:
            assessment_status = "Incomplete"

        audiogram = conn.execute(
            """
            SELECT COUNT(*), MAX(created_at)
            FROM audiograms
            WHERE patient_id=?
            """,
            (patient_id,),
        ).fetchone() if _table_exists(conn, "audiograms") else (0, None)
        audiogram_available = bool(audiogram and int(audiogram[0] or 0) > 0)

        followup = conn.execute(
            """
            SELECT session_date
            FROM sessions
            WHERE patient_id=?
            ORDER BY session_date DESC, id DESC LIMIT 1
            """,
            (patient_id,),
        ).fetchone()

        account = conn.execute(
            """
            SELECT username, is_active
            FROM users
            WHERE patient_id=? AND role='patient'
            ORDER BY id DESC LIMIT 1
            """,
            (patient_id,),
        ).fetchone()

        open_care = 0
        action_required = 0
        if _table_exists(conn, "care_requests"):
            open_care = conn.execute(
                """
                SELECT COUNT(*) FROM care_requests
                WHERE patient_id=? AND status NOT IN ('Completed','Cancelled')
                """,
                (patient_id,),
            ).fetchone()[0]
            action_required = conn.execute(
                """
                SELECT COUNT(*) FROM care_requests
                WHERE patient_id=?
                  AND direction='audiologist_to_patient'
                  AND status IN ('Pending','Needs Information','Needs Confirmation','Referred')
                """,
                (patient_id,),
            ).fetchone()[0]

        return {
            "id": patient[0],
            "patient_code": patient[1],
            "age": patient[2],
            "sex": patient[3],
            "created_at": patient[4],
            "account_username": account[0] if account else None,
            "account_active": bool(account and account[1]),
            "assessment_status": assessment_status,
            "audiogram_available": audiogram_available,
            "latest_audiogram": str(audiogram[1])[:10] if audiogram_available and audiogram[1] else None,
            "manual_status": _latest_review_status(conn, patient_id, "standard"),
            "ai_status": _latest_review_status(conn, patient_id, "ai"),
            "last_followup": followup[0] if followup else None,
            "open_care": int(open_care or 0),
            "action_required": int(action_required or 0),
        }
    finally:
        conn.close()


def list_patient_summaries() -> list[dict[str, Any]]:
    return [
        summary
        for patient in list_patients()
        if (summary := get_patient_summary(patient["id"])) is not None
    ]
