"""Patient service layer for SmartTinnitus.

The Streamlit UI should use this module for patient-directory operations instead
of issuing patient SQL directly. SQLite remains the storage engine for now; the
service boundary makes a later PostgreSQL/MySQL migration much easier.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
import sqlite3

from backend.db import get_connection
from patient_access import create_patient_login


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def list_patients() -> list[dict[str, Any]]:
    """Return all registered patients in a stable UI-friendly structure."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, patient_code, age, sex, created_at
            FROM patients
            ORDER BY patient_code COLLATE NOCASE ASC, id ASC
            """
        ).fetchall()
        return [
            {
                "id": int(row[0]),
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
            WHERE id = ?
            LIMIT 1
            """,
            (int(patient_id),),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row[0]),
            "patient_code": row[1],
            "age": row[2],
            "sex": row[3],
            "created_at": row[4],
        }
    finally:
        conn.close()


def get_patient_id_by_code(patient_code: str) -> int | None:
    code = str(patient_code or "").strip()
    if not code:
        return None

    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT id
            FROM patients
            WHERE patient_code = ?
            LIMIT 1
            """,
            (code,),
        ).fetchone()
        return int(row[0]) if row is not None else None
    finally:
        conn.close()


def _assessment_status(conn, patient_id: int) -> str:
    if not _table_exists(conn, "tinnitus_profiles"):
        return "Not started"

    row = conn.execute(
        "SELECT COUNT(*) FROM tinnitus_profiles WHERE patient_id = ?",
        (int(patient_id),),
    ).fetchone()
    return "Available" if row and int(row[0] or 0) > 0 else "Not started"


def _audiogram_available(conn, patient_id: int) -> bool:
    if not _table_exists(conn, "audiograms"):
        return False

    row = conn.execute(
        "SELECT COUNT(*) FROM audiograms WHERE patient_id = ?",
        (int(patient_id),),
    ).fetchone()
    return bool(row and int(row[0] or 0) > 0)


def list_patient_summaries() -> list[dict[str, Any]]:
    """Return compact patient-directory cards used by the Patients page."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, patient_code, age, sex, created_at
            FROM patients
            ORDER BY patient_code COLLATE NOCASE ASC, id ASC
            """
        ).fetchall()

        summaries: list[dict[str, Any]] = []
        for row in rows:
            patient_id = int(row[0])
            summaries.append(
                {
                    "id": patient_id,
                    "patient_code": row[1],
                    "age": row[2],
                    "sex": row[3],
                    "created_at": row[4],
                    "assessment_status": _assessment_status(conn, patient_id),
                    "audiogram_available": _audiogram_available(conn, patient_id),
                }
            )
        return summaries
    finally:
        conn.close()


def create_patient_with_login(
    patient_code: str,
    age: int | None,
    sex: str,
) -> dict[str, Any]:
    """Create a patient and its patient login as one application-level operation.

    If account creation fails, the patient row is rolled back/deleted so the UI
    cannot leave behind an unusable patient record.
    """
    code = str(patient_code or "").strip()
    sex_value = str(sex or "").strip()

    if not code:
        return {"success": False, "message": "Enter a Patient ID."}
    if not sex_value or sex_value == "Select":
        return {"success": False, "message": "Select sex."}

    conn = get_connection()
    patient_id: int | None = None

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO patients (patient_code, age, sex, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                code,
                int(age) if age is not None else None,
                sex_value,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        patient_id = int(cursor.lastrowid)
        conn.commit()

    except sqlite3.IntegrityError:
        conn.rollback()
        return {"success": False, "message": "Patient ID already exists."}
    except Exception as exc:
        conn.rollback()
        return {
            "success": False,
            "message": f"The patient could not be created: {exc}",
        }
    finally:
        conn.close()

    try:
        success, message, username, password = create_patient_login(
            patient_id=patient_id,
            patient_code=code,
        )
    except Exception as exc:
        success = False
        message = str(exc)
        username = None
        password = None

    if not success:
        cleanup_conn = get_connection()
        try:
            cleanup_conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
            cleanup_conn.commit()
        finally:
            cleanup_conn.close()

        return {
            "success": False,
            "message": message or "The patient account could not be created.",
        }

    return {
        "success": True,
        "message": "Patient and login account created successfully.",
        "patient_id": patient_id,
        "patient_code": code,
        "username": username,
        "password": password,
    }
