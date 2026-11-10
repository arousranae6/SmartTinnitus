from datetime import datetime
from backend.db import get_connection


def _now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def ensure_care_coordination_table(connection=None):
    owns_connection = connection is None
    conn = connection or get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS care_requests
            (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                direction TEXT NOT NULL,
                request_type TEXT NOT NULL,
                communication_mode TEXT,
                specialist TEXT,
                title TEXT,
                message TEXT,
                preferred_datetime TEXT,
                scheduled_datetime TEXT,
                status TEXT NOT NULL DEFAULT 'Pending',
                patient_response TEXT,
                audiologist_note TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
            """
        )
        conn.commit()
    finally:
        if owns_connection:
            conn.close()


def create_care_request(patient_id, direction, request_type, title, message,
                        communication_mode=None, specialist=None,
                        preferred_datetime=None, scheduled_datetime=None,
                        status='Pending'):
    conn = get_connection()
    try:
        ensure_care_coordination_table(conn)
        cur = conn.execute(
            """
            INSERT INTO care_requests
            (patient_id, direction, request_type, communication_mode, specialist,
             title, message, preferred_datetime, scheduled_datetime, status,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (patient_id, direction, request_type, communication_mode, specialist,
             title, message, preferred_datetime, scheduled_datetime, status,
             _now(), _now()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_patient_care_summary(patient_id):
    if patient_id is None:
        return {'pending': 0, 'scheduled': 0, 'action_required': 0, 'latest': None}
    conn = get_connection()
    try:
        ensure_care_coordination_table(conn)
        pending = conn.execute(
            """SELECT COUNT(*) FROM care_requests
               WHERE patient_id=? AND status NOT IN ('Completed','Cancelled')""",
            (patient_id,),
        ).fetchone()[0]
        scheduled = conn.execute(
            """SELECT COUNT(*) FROM care_requests
               WHERE patient_id=? AND status IN ('Scheduled','Accepted')""",
            (patient_id,),
        ).fetchone()[0]
        action_required = conn.execute(
            """SELECT COUNT(*) FROM care_requests
               WHERE patient_id=? AND direction='audiologist_to_patient'
               AND status IN ('Pending','Needs Information','Needs Confirmation','Referred')""",
            (patient_id,),
        ).fetchone()[0]
        latest = conn.execute(
            """SELECT request_type, title, status, scheduled_datetime, created_at
               FROM care_requests WHERE patient_id=? ORDER BY id DESC LIMIT 1""",
            (patient_id,),
        ).fetchone()
        return {
            'pending': int(pending or 0),
            'scheduled': int(scheduled or 0),
            'action_required': int(action_required or 0),
            'latest': latest,
        }
    finally:
        conn.close()


def list_incoming_requests(patient_id):
    conn = get_connection()
    try:
        ensure_care_coordination_table(conn)
        return conn.execute(
            """SELECT id, request_type, communication_mode, specialist, title, message,
                      preferred_datetime, scheduled_datetime, status, patient_response,
                      audiologist_note, created_at
               FROM care_requests
               WHERE patient_id=? AND direction='audiologist_to_patient'
               ORDER BY id DESC""",
            (patient_id,),
        ).fetchall()
    finally:
        conn.close()


def list_patient_outgoing_requests(patient_id):
    conn = get_connection()
    try:
        ensure_care_coordination_table(conn)
        return conn.execute(
            """SELECT id, title, message, preferred_datetime, scheduled_datetime, status, created_at
               FROM care_requests
               WHERE patient_id=? AND direction='patient_to_audiologist'
               ORDER BY id DESC""",
            (patient_id,),
        ).fetchall()
    finally:
        conn.close()


def list_all_care_requests(patient_id):
    conn = get_connection()
    try:
        ensure_care_coordination_table(conn)
        return conn.execute(
            """SELECT id, direction, request_type, communication_mode, specialist, title, message,
                      preferred_datetime, scheduled_datetime, status, patient_response,
                      audiologist_note, created_at
               FROM care_requests WHERE patient_id=? ORDER BY id DESC""",
            (patient_id,),
        ).fetchall()
    finally:
        conn.close()


def submit_patient_response(request_id, response):
    conn = get_connection()
    try:
        ensure_care_coordination_table(conn)
        conn.execute(
            """UPDATE care_requests
               SET patient_response=?, status='Responded', updated_at=?
               WHERE id=?""",
            ((response or '').strip(), _now(), request_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_request_status(request_id, status):
    conn = get_connection()
    try:
        ensure_care_coordination_table(conn)
        conn.execute(
            "UPDATE care_requests SET status=?, updated_at=? WHERE id=?",
            (status, _now(), request_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_care_request(request_id, status, scheduled_datetime=None, audiologist_note=None):
    conn = get_connection()
    try:
        ensure_care_coordination_table(conn)
        conn.execute(
            """UPDATE care_requests
               SET status=?, scheduled_datetime=?, audiologist_note=?, updated_at=?
               WHERE id=?""",
            (status,
             (scheduled_datetime or '').strip() or None,
             (audiologist_note or '').strip() or None,
             _now(), request_id),
        )
        conn.commit()
    finally:
        conn.close()
