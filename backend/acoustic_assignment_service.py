from datetime import datetime
from backend.db import get_connection


def _now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def ensure_acoustic_assignment_table(connection=None):
    own = connection is None
    conn = connection or get_connection()
    try:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS acoustic_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                profile_id INTEGER,
                pathway TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'assigned',
                stimulus_type TEXT NOT NULL,
                center_frequency_hz REAL,
                bandwidth_hz REAL,
                relative_level REAL NOT NULL,
                duration_seconds INTEGER NOT NULL DEFAULT 20,
                predicted_score REAL,
                model_version TEXT,
                model_trained_at TEXT,
                recommendation_created_at TEXT,
                audiologist_note TEXT,
                created_at TEXT NOT NULL,
                sent_at TEXT,
                responded_at TEXT,
                comfort_score INTEGER,
                tinnitus_noticeability INTEGER,
                listen_again TEXT,
                patient_comment TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(id),
                FOREIGN KEY (profile_id) REFERENCES tinnitus_profiles(id)
            )
            '''
        )
        conn.commit()
    finally:
        if own:
            conn.close()


def create_assignment(
    patient_id,
    profile_id,
    pathway,
    stimulus_type,
    center_frequency_hz,
    bandwidth_hz,
    relative_level,
    duration_seconds=20,
    predicted_score=None,
    model_version=None,
    model_trained_at=None,
    recommendation_created_at=None,
    audiologist_note=None,
):
    conn = get_connection()
    try:
        ensure_acoustic_assignment_table(conn)
        now = _now()
        cur = conn.execute(
            '''
            INSERT INTO acoustic_assignments (
                patient_id, profile_id, pathway, status,
                stimulus_type, center_frequency_hz, bandwidth_hz,
                relative_level, duration_seconds, predicted_score,
                model_version, model_trained_at, recommendation_created_at,
                audiologist_note, created_at, sent_at
            ) VALUES (?, ?, ?, 'assigned', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                int(patient_id), profile_id, pathway,
                stimulus_type,
                float(center_frequency_hz) if center_frequency_hz is not None else None,
                float(bandwidth_hz) if bandwidth_hz is not None else None,
                float(relative_level), int(duration_seconds),
                float(predicted_score) if predicted_score is not None else None,
                model_version, model_trained_at,
                recommendation_created_at, audiologist_note,
                now, now,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_pending_assignment(patient_id):
    conn = get_connection()
    try:
        ensure_acoustic_assignment_table(conn)
        row = conn.execute(
            '''
            SELECT id, patient_id, profile_id, pathway, status,
                   stimulus_type, center_frequency_hz, bandwidth_hz,
                   relative_level, duration_seconds, predicted_score,
                   audiologist_note, sent_at
            FROM acoustic_assignments
            WHERE patient_id = ? AND status = 'assigned'
            ORDER BY id DESC
            LIMIT 1
            ''',
            (int(patient_id),),
        ).fetchone()
        if row is None:
            return None
        keys = [
            'id','patient_id','profile_id','pathway','status','stimulus_type',
            'center_frequency_hz','bandwidth_hz','relative_level','duration_seconds',
            'predicted_score','audiologist_note','sent_at'
        ]
        return dict(zip(keys, row))
    finally:
        conn.close()


def submit_assignment_feedback(
    assignment_id,
    patient_id,
    comfort_score,
    tinnitus_noticeability,
    listen_again,
    patient_comment='',
):
    conn = get_connection()
    try:
        ensure_acoustic_assignment_table(conn)
        cur = conn.execute(
            '''
            UPDATE acoustic_assignments
            SET status = 'response_received',
                responded_at = ?,
                comfort_score = ?,
                tinnitus_noticeability = ?,
                listen_again = ?,
                patient_comment = ?
            WHERE id = ? AND patient_id = ? AND status = 'assigned'
            ''',
            (
                _now(), int(comfort_score), int(tinnitus_noticeability),
                listen_again, patient_comment.strip() or None,
                int(assignment_id), int(patient_id),
            ),
        )
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def list_assignment_feedback(patient_id, limit=20):
    conn = get_connection()
    try:
        ensure_acoustic_assignment_table(conn)
        rows = conn.execute(
            '''
            SELECT id, pathway, stimulus_type, center_frequency_hz,
                   bandwidth_hz, relative_level, duration_seconds,
                   predicted_score, sent_at, responded_at,
                   comfort_score, tinnitus_noticeability, listen_again,
                   patient_comment
            FROM acoustic_assignments
            WHERE patient_id = ? AND status = 'response_received'
            ORDER BY id DESC
            LIMIT ?
            ''',
            (int(patient_id), int(limit)),
        ).fetchall()
        keys = [
            'id','pathway','stimulus_type','center_frequency_hz','bandwidth_hz',
            'relative_level','duration_seconds','predicted_score','sent_at','responded_at',
            'comfort_score','tinnitus_noticeability','listen_again','patient_comment'
        ]
        return [dict(zip(keys, row)) for row in rows]
    finally:
        conn.close()
