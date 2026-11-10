"""Authentication service for SmartTinnitus.

Password hashing and user-account database operations live here rather than in
Streamlit UI code.  The session/UI functions remain in ``login.py``.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime

from backend.db import get_connection


PBKDF2_ITERATIONS = 310_000


def hash_password(password, salt=None):
    if salt is None:
        salt_bytes = os.urandom(16)
    else:
        salt_bytes = bytes.fromhex(salt)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        str(password).encode("utf-8"),
        salt_bytes,
        PBKDF2_ITERATIONS,
    ).hex()

    return password_hash, salt_bytes.hex()


def verify_password(password, stored_hash, stored_salt):
    calculated_hash, _ = hash_password(password, stored_salt)
    return hmac.compare_digest(calculated_hash, stored_hash)


def username_exists(username):
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT id
            FROM users
            WHERE LOWER(username) = LOWER(?)
            LIMIT 1
            """,
            (str(username).strip(),),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def count_users(role=None):
    conn = get_connection()
    try:
        if role is None:
            row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM users WHERE role = ?",
                (str(role).strip().lower(),),
            ).fetchone()
        return int(row[0] or 0)
    finally:
        conn.close()


def create_user_account(
    username,
    password,
    role,
    patient_id=None,
    display_name=None,
):
    username = str(username).strip()
    password = str(password)
    role = str(role).strip().lower()

    if not username:
        return False, "Username is required."
    if len(username) < 3:
        return False, "Username must contain at least 3 characters."
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."
    if role not in ["patient", "audiologist"]:
        return False, "Invalid account role."
    if role == "patient" and patient_id is None:
        return False, "A patient account must be linked to a patient record."
    if username_exists(username):
        return False, "This username already exists."

    password_hash, password_salt = hash_password(password)

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO users
            (
                username,
                password_hash,
                password_salt,
                role,
                patient_id,
                display_name,
                is_active,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                password_hash,
                password_salt,
                role,
                patient_id,
                str(display_name).strip() if display_name else None,
                1,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    except Exception as error:
        conn.rollback()
        return False, f"Could not create account: {error}"
    finally:
        conn.close()

    return True, "Account created successfully."


def authenticate_user(username, password, role):
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT
                id,
                username,
                password_hash,
                password_salt,
                role,
                patient_id,
                display_name,
                is_active
            FROM users
            WHERE LOWER(username) = LOWER(?)
              AND role = ?
            LIMIT 1
            """,
            (
                str(username).strip(),
                str(role).strip().lower(),
            ),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return False, None, "Incorrect username or password."

    (
        user_id,
        stored_username,
        password_hash,
        password_salt,
        stored_role,
        patient_id,
        display_name,
        is_active,
    ) = row

    if not bool(is_active):
        return False, None, "This account is inactive."

    if not verify_password(password, password_hash, password_salt):
        return False, None, "Incorrect username or password."

    user = {
        "user_id": user_id,
        "username": stored_username,
        "role": stored_role,
        "patient_id": patient_id,
        "display_name": display_name or stored_username,
    }

    return True, user, None
