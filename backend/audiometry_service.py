"""Audiometry data service for SmartTinnitus.

Keeps audiometry-related SQL outside the Streamlit UI. The service preserves
each saved audiogram as a dated set so clinicians can review the latest result
and longitudinal audiogram history.
"""

from datetime import datetime

from backend.db import get_connection


def get_latest_tinnitus_pitch(patient_id):
    """Return the latest stored tinnitus pitch in Hz, or ``None``."""
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT estimated_pitch_hz
            FROM tinnitus_profiles
            WHERE patient_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (int(patient_id),),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return float(row[0])
    finally:
        conn.close()


def _load_audiogram_set(conn, patient_id, created_at):
    rows = conn.execute(
        """
        SELECT ear, frequency_hz, threshold_db_hl
        FROM audiograms
        WHERE patient_id = ? AND created_at = ?
        ORDER BY frequency_hz ASC
        """,
        (int(patient_id), created_at),
    ).fetchall()

    result = {"Right": {}, "Left": {}, "created_at": created_at}
    for ear, frequency_hz, threshold_db_hl in rows:
        if ear not in ("Right", "Left"):
            continue
        result[ear][int(frequency_hz)] = float(threshold_db_hl)
    return result


def get_latest_audiogram(patient_id):
    """Return the patient's most recently saved complete audiogram set."""
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT created_at
            FROM audiograms
            WHERE patient_id = ?
            GROUP BY created_at
            ORDER BY datetime(created_at) DESC, created_at DESC
            LIMIT 1
            """,
            (int(patient_id),),
        ).fetchone()
        if row is None:
            return {"Right": {}, "Left": {}, "created_at": None}
        return _load_audiogram_set(conn, patient_id, row[0])
    finally:
        conn.close()


def get_audiogram_history(patient_id, limit=10):
    """Return saved audiogram sets newest first."""
    limit = max(1, min(int(limit), 100))
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT created_at
            FROM audiograms
            WHERE patient_id = ?
            GROUP BY created_at
            ORDER BY datetime(created_at) DESC, created_at DESC
            LIMIT ?
            """,
            (int(patient_id), limit),
        ).fetchall()
        return [_load_audiogram_set(conn, patient_id, row[0]) for row in rows]
    finally:
        conn.close()


def save_audiogram(patient_id, frequencies, right_thresholds, left_thresholds):
    """Save one complete audiogram set without deleting earlier sets."""
    frequencies = list(frequencies)
    right_thresholds = list(right_thresholds)
    left_thresholds = list(left_thresholds)

    if not frequencies:
        raise ValueError("Audiogram frequencies cannot be empty.")
    if len(frequencies) != len(right_thresholds) or len(frequencies) != len(left_thresholds):
        raise ValueError("Audiogram frequency and threshold lists must have the same length.")

    normalized = []
    for frequency, right, left in zip(frequencies, right_thresholds, left_thresholds):
        frequency = int(frequency)
        right = float(right)
        left = float(left)
        if right < -10 or right > 120 or left < -10 or left > 120:
            raise ValueError("Audiogram thresholds must be between -10 and 120 dB HL.")
        normalized.append((frequency, right, left))

    # Microseconds prevent two saves within the same second from being merged.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    conn = get_connection()
    try:
        conn.execute("BEGIN")
        right_rows = [
            (int(patient_id), "Right", frequency, right, timestamp)
            for frequency, right, _left in normalized
        ]
        left_rows = [
            (int(patient_id), "Left", frequency, left, timestamp)
            for frequency, _right, left in normalized
        ]
        conn.executemany(
            """
            INSERT INTO audiograms
            (patient_id, ear, frequency_hz, threshold_db_hl, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            right_rows + left_rows,
        )
        conn.commit()
        return timestamp
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
