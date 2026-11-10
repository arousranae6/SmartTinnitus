"""Final Clinical Report service for SmartTinnitus.

The report mirrors the final application workflow only:
- Patient information
- Tinnitus profile
- Latest audiogram
- Acoustic personalization / assignments
- Patient acoustic feedback
- Patient Acoustic Digital Twin evidence
- Care coordination
- Clinical summary

Legacy workflow concepts such as adaptive sessions, old follow-up sessions,
audiologist Approve/Modify/Reject decisions, confidence/active learning and
AI model evaluation are intentionally excluded.
"""

from backend.db import get_connection

import hashlib
from datetime import datetime
from pathlib import Path
from backend.audiometry_service import get_latest_audiogram
from report import generate_patient_report
from optimization import calculate_response_score


def _row_to_dict(row, columns):
    if row is None:
        return None
    return dict(zip(columns, row))


def _safe_response_score(comfort, noticeability, listen_again):
    try:
        return float(
            calculate_response_score(
                comfort,
                noticeability,
                listen_again,
            )
        )
    except Exception:
        return None


def get_report_context(patient_id):
    patient_id = int(patient_id)
    conn = get_connection()

    try:
        patient = _row_to_dict(
            conn.execute(
                """
                SELECT patient_code, age, sex, created_at
                FROM patients
                WHERE id = ?
                LIMIT 1
                """,
                (patient_id,),
            ).fetchone(),
            ["patient_code", "age", "sex", "created_at"],
        )

        if patient is None:
            raise ValueError("Patient not found.")

        profile = _row_to_dict(
            conn.execute(
                """
                SELECT laterality, tinnitus_type, temporal_pattern, duration_months,
                       annoyance_score, sleep_score, concentration_score, stress_score,
                       estimated_pitch_hz, loudness_relative, created_at
                FROM tinnitus_profiles
                WHERE patient_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (patient_id,),
            ).fetchone(),
            [
                "laterality",
                "tinnitus_type",
                "temporal_pattern",
                "duration_months",
                "annoyance_score",
                "sleep_score",
                "concentration_score",
                "stress_score",
                "estimated_pitch_hz",
                "loudness_relative",
                "created_at",
            ],
        )

        # Final assignment workflow: manual and AI-assisted candidates that
        # were actually sent to the patient.
        assignment_rows = conn.execute(
            """
            SELECT id, pathway, status, stimulus_type, center_frequency_hz,
                   bandwidth_hz, relative_level, duration_seconds,
                   predicted_score, audiologist_note, sent_at, responded_at,
                   comfort_score, tinnitus_noticeability, listen_again,
                   patient_comment
            FROM acoustic_assignments
            WHERE patient_id = ?
            ORDER BY id DESC
            """,
            (patient_id,),
        ).fetchall()

        assignment_columns = [
            "id",
            "pathway",
            "status",
            "stimulus_type",
            "center_frequency_hz",
            "bandwidth_hz",
            "relative_level",
            "duration_seconds",
            "predicted_score",
            "audiologist_note",
            "sent_at",
            "responded_at",
            "comfort_score",
            "tinnitus_noticeability",
            "listen_again",
            "patient_comment",
        ]

        assignments = [
            dict(zip(assignment_columns, row))
            for row in assignment_rows
        ]

        feedback = []
        for item in assignments:
            if item.get("status") == "response_received":
                entry = dict(item)
                entry["response_score"] = _safe_response_score(
                    item.get("comfort_score"),
                    item.get("tinnitus_noticeability"),
                    item.get("listen_again"),
                )
                feedback.append(entry)

        # Historical observed acoustic evidence remains relevant to the
        # Patient Acoustic Digital Twin, but legacy workflow metadata does not.
        trial_rows = conn.execute(
            """
            SELECT id, source_type, stimulus_type, center_frequency_hz,
                   bandwidth_hz, relative_level, comfort_score,
                   tinnitus_noticeability, listen_again, tested_at, created_at
            FROM stimulus_trials
            WHERE patient_id = ?
            ORDER BY id ASC
            """,
            (patient_id,),
        ).fetchall()

        trial_columns = [
            "id",
            "source_type",
            "stimulus_type",
            "center_frequency_hz",
            "bandwidth_hz",
            "relative_level",
            "comfort_score",
            "tinnitus_noticeability",
            "listen_again",
            "tested_at",
            "created_at",
        ]

        evidence = []
        for row in trial_rows:
            item = dict(zip(trial_columns, row))
            item["response_score"] = _safe_response_score(
                item.get("comfort_score"),
                item.get("tinnitus_noticeability"),
                item.get("listen_again"),
            )
            evidence.append(item)

        care_rows = conn.execute(
            """
            SELECT id, direction, request_type, communication_mode, specialist,
                   title, preferred_datetime, scheduled_datetime, status,
                   patient_response, audiologist_note, created_at
            FROM care_requests
            WHERE patient_id = ?
            ORDER BY id DESC
            """,
            (patient_id,),
        ).fetchall()

        care_columns = [
            "id",
            "direction",
            "request_type",
            "communication_mode",
            "specialist",
            "title",
            "preferred_datetime",
            "scheduled_datetime",
            "status",
            "patient_response",
            "audiologist_note",
            "created_at",
        ]

        care_requests = [
            dict(zip(care_columns, row))
            for row in care_rows
        ]

    finally:
        conn.close()

    latest_audiogram = get_latest_audiogram(patient_id)

    audiogram = None
    if latest_audiogram:
        right = latest_audiogram.get("Right", {}) or {}
        left = latest_audiogram.get("Left", {}) or {}
        frequencies = sorted(set(right.keys()) | set(left.keys()))

        if frequencies:
            audiogram = {
                "created_at": latest_audiogram.get("created_at"),
                "frequencies": frequencies,
                "right_ear": [right.get(f) for f in frequencies],
                "left_ear": [left.get(f) for f in frequencies],
                "tinnitus_pitch": (
                    profile.get("estimated_pitch_hz")
                    if profile
                    else None
                ),
            }

    valid_evidence = [
        item for item in evidence
        if item.get("response_score") is not None
    ]

    best_evidence = (
        max(valid_evidence, key=lambda x: x["response_score"])
        if valid_evidence
        else None
    )

    latest_feedback = feedback[0] if feedback else None

    digital_twin = {
        "observed_evidence_count": len(evidence),
        "assignment_feedback_count": len(feedback),
        "best_observed_response": best_evidence,
        "latest_feedback": latest_feedback,
    }

    return {
        "patient": patient,
        "profile": profile,
        "audiogram": audiogram,
        "assignments": assignments,
        "feedback": feedback,
        "digital_twin": digital_twin,
        "care_requests": care_requests,
    }


def get_report_summary(patient_id):
    context = get_report_context(patient_id)

    return {
        "patient_code": context["patient"]["patient_code"],
        "acoustic_tests": len(context["assignments"]),
        "feedback_count": len(context["feedback"]),
        "has_audiogram": context["audiogram"] is not None,
    }



def ensure_clinical_reports_table():
    """Create the final report-history table if it does not exist."""
    conn = get_connection()
    try:
        conn.execute(
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
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS ix_clinical_reports_patient_id
            ON clinical_reports (patient_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS ix_clinical_reports_generated_at
            ON clinical_reports (generated_at DESC)
            """
        )
        conn.commit()
    finally:
        conn.close()


def get_report_history(patient_id, limit=10):
    ensure_clinical_reports_table()

    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, generated_at, report_version, filename,
                   storage_path, status, file_hash, generated_by_user_id
            FROM clinical_reports
            WHERE patient_id = ?
            ORDER BY generated_at DESC, id DESC
            LIMIT ?
            """,
            (int(patient_id), int(limit)),
        ).fetchall()
    finally:
        conn.close()

    columns = [
        "id",
        "generated_at",
        "report_version",
        "filename",
        "storage_path",
        "status",
        "file_hash",
        "generated_by_user_id",
    ]

    return [dict(zip(columns, row)) for row in rows]


def _sha256_file(filename):
    path = Path(filename)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_generated_report(
    patient_id,
    filename,
    generated_by_user_id=None,
    report_version="1.0",
):
    ensure_clinical_reports_table()

    file_hash = _sha256_file(filename)

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO clinical_reports (
                patient_id,
                generated_by_user_id,
                generated_at,
                report_version,
                filename,
                storage_path,
                status,
                file_hash
            )
            VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
            """,
            (
                int(patient_id),
                (
                    int(generated_by_user_id)
                    if generated_by_user_id is not None
                    else None
                ),
                str(report_version),
                str(Path(filename).name),
                str(Path(filename)),
                "generated",
                file_hash,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def generate_clinical_report(
    patient_id,
    filename=None,
    generated_by_user_id=None,
    report_version="1.0",
):
    context = get_report_context(patient_id)
    patient_code = context["patient"]["patient_code"]

    if filename is None:
        filename = f"SmartTinnitus_Clinical_Report_{patient_code}.pdf"

    generate_patient_report(
        filename=filename,
        patient=context["patient"],
        profile=context["profile"],
        audiogram=context["audiogram"],
        assignments=context["assignments"],
        feedback=context["feedback"],
        digital_twin=context["digital_twin"],
        care_requests=context["care_requests"],
    )

    _record_generated_report(
        patient_id=patient_id,
        filename=filename,
        generated_by_user_id=generated_by_user_id,
        report_version=report_version,
    )

    return filename
