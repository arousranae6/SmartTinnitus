import sqlite3


# =====================================================
# DATABASE CONNECTION
# =====================================================

def _ensure_runtime_schema(
    conn
):
    """
    Defensive schema check.

    This runs whenever SmartTinnitus opens the database so an older
    smarttinnitus.db file is automatically upgraded before another
    module queries the new provenance fields.
    """

    cursor = conn.cursor()

    # -------------------------------------------------
    # stimulus_trials provenance columns
    # -------------------------------------------------

    cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name = 'stimulus_trials'
        """
    )

    if cursor.fetchone() is not None:

        cursor.execute(
            """
            PRAGMA table_info(stimulus_trials)
            """
        )

        trial_columns = {
            row[1]
            for row
            in cursor.fetchall()
        }

        required_trial_columns = {
            "session_id":
                "INTEGER",

            "decision_id":
                "INTEGER",

            "adaptive_session_id":
                "INTEGER",

            "profile_id":
                "INTEGER",

            "source_type":
                "TEXT",

            "model_version":
                "TEXT",

            "model_trained_at":
                "TEXT",

            "model_predicted_score":
                "REAL",

            "recommendation_created_at":
                "TEXT",

            "tested_at":
                "TEXT"
        }

        for (
            column_name,
            column_type
        ) in required_trial_columns.items():

            if column_name not in trial_columns:

                cursor.execute(
                    f"""
                    ALTER TABLE stimulus_trials
                    ADD COLUMN {column_name} {column_type}
                    """
                )

    # -------------------------------------------------
    # audiologist_decisions provenance columns
    # -------------------------------------------------

    cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name = 'audiologist_decisions'
        """
    )

    if cursor.fetchone() is not None:

        cursor.execute(
            """
            PRAGMA table_info(audiologist_decisions)
            """
        )

        decision_columns = {
            row[1]
            for row
            in cursor.fetchall()
        }

        required_decision_columns = {
            "profile_id":
                "INTEGER",

            "model_version":
                "TEXT",

            "model_trained_at":
                "TEXT",

            "recommendation_created_at":
                "TEXT",

            "final_predicted_score":
                "REAL"
        }

        for (
            column_name,
            column_type
        ) in required_decision_columns.items():

            if column_name not in decision_columns:

                cursor.execute(
                    f"""
                    ALTER TABLE audiologist_decisions
                    ADD COLUMN {column_name} {column_type}
                    """
                )

    # -------------------------------------------------
    # Adaptive-session tables
    # -------------------------------------------------

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS adaptive_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        profile_id INTEGER,
        status TEXT,
        max_trials INTEGER,
        initial_center_frequency_hz REAL,
        initial_relative_level REAL,
        best_adaptive_trial_id INTEGER,
        best_response_score REAL,
        promising_frequency_min_hz REAL,
        promising_frequency_max_hz REAL,
        summary_text TEXT,
        started_at TEXT,
        completed_at TEXT,
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS adaptive_session_trials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        adaptive_session_id INTEGER,
        patient_id INTEGER,
        profile_id INTEGER,
        linked_stimulus_trial_id INTEGER,
        trial_number INTEGER,
        phase TEXT,
        rationale TEXT,
        stimulus_type TEXT,
        center_frequency_hz REAL,
        bandwidth_hz REAL,
        relative_level REAL,
        comfort_score INTEGER,
        tinnitus_noticeability INTEGER,
        listen_again TEXT,
        response_score REAL,
        tested_at TEXT,
        created_at TEXT
    )
    """)

    conn.commit()


def get_connection():

    conn = sqlite3.connect(
        "smarttinnitus.db",
        check_same_thread=False
    )

    _ensure_runtime_schema(
        conn
    )

    return conn


# =====================================================
# INITIALIZE DATABASE
# =====================================================

def initialize_database():

    conn = get_connection()
    cursor = conn.cursor()

    # =================================================
    # PATIENTS
    # =================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_code TEXT UNIQUE,
        age INTEGER,
        sex TEXT,
        created_at TEXT
    )
    """)

    # =================================================
    # APPLICATION USERS
    # =================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        password_salt TEXT NOT NULL,
        role TEXT NOT NULL,
        patient_id INTEGER,
        display_name TEXT,
        is_active INTEGER DEFAULT 1,
        created_at TEXT,
        FOREIGN KEY (patient_id)
            REFERENCES patients(id)
    )
    """)

    # =================================================
    # TINNITUS PROFILES
    # =================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tinnitus_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        laterality TEXT,
        tinnitus_type TEXT,
        temporal_pattern TEXT,
        duration_months INTEGER,
        annoyance_score INTEGER,
        sleep_score INTEGER,
        concentration_score INTEGER,
        stress_score INTEGER,
        estimated_pitch_hz REAL,
        loudness_relative REAL,
        created_at TEXT,
        FOREIGN KEY (patient_id)
            REFERENCES patients(id)
    )
    """)

    # =================================================
    # FOLLOW-UP SESSIONS
    # =================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        session_date TEXT,
        annoyance_before INTEGER,
        annoyance_after INTEGER,
        tinnitus_noticeability_before INTEGER,
        tinnitus_noticeability_after INTEGER,
        notes TEXT,
        created_at TEXT,
        FOREIGN KEY (patient_id)
            REFERENCES patients(id)
    )
    """)

    # =================================================
    # STIMULUS TRIALS
    # =================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stimulus_trials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        session_id INTEGER,
        decision_id INTEGER,
        profile_id INTEGER,
        adaptive_session_id INTEGER,

        source_type TEXT,
        model_version TEXT,
        model_trained_at TEXT,
        model_predicted_score REAL,
        recommendation_created_at TEXT,
        tested_at TEXT,

        stimulus_type TEXT,
        center_frequency_hz REAL,
        bandwidth_hz REAL,
        relative_level REAL,
        comfort_score INTEGER,
        tinnitus_noticeability INTEGER,
        listen_again TEXT,
        created_at TEXT,

        FOREIGN KEY (patient_id)
            REFERENCES patients(id),

        FOREIGN KEY (session_id)
            REFERENCES sessions(id),

        FOREIGN KEY (decision_id)
            REFERENCES audiologist_decisions(id),

        FOREIGN KEY (profile_id)
            REFERENCES tinnitus_profiles(id)
    )
    """)


    # =================================================
    # CLOSED-LOOP ADAPTIVE SESSIONS
    # =================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS adaptive_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        profile_id INTEGER,
        status TEXT,
        max_trials INTEGER,
        initial_center_frequency_hz REAL,
        initial_relative_level REAL,
        best_adaptive_trial_id INTEGER,
        best_response_score REAL,
        promising_frequency_min_hz REAL,
        promising_frequency_max_hz REAL,
        summary_text TEXT,
        started_at TEXT,
        completed_at TEXT,
        created_at TEXT,

        FOREIGN KEY (patient_id)
            REFERENCES patients(id),

        FOREIGN KEY (profile_id)
            REFERENCES tinnitus_profiles(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS adaptive_session_trials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        adaptive_session_id INTEGER,
        patient_id INTEGER,
        profile_id INTEGER,
        linked_stimulus_trial_id INTEGER,
        trial_number INTEGER,
        phase TEXT,
        rationale TEXT,
        stimulus_type TEXT,
        center_frequency_hz REAL,
        bandwidth_hz REAL,
        relative_level REAL,
        comfort_score INTEGER,
        tinnitus_noticeability INTEGER,
        listen_again TEXT,
        response_score REAL,
        tested_at TEXT,
        created_at TEXT,

        FOREIGN KEY (adaptive_session_id)
            REFERENCES adaptive_sessions(id),

        FOREIGN KEY (patient_id)
            REFERENCES patients(id),

        FOREIGN KEY (profile_id)
            REFERENCES tinnitus_profiles(id),

        FOREIGN KEY (linked_stimulus_trial_id)
            REFERENCES stimulus_trials(id)
    )
    """)

    # =================================================
    # AUDIOGRAMS
    # =================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audiograms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        ear TEXT,
        frequency_hz INTEGER,
        threshold_db_hl REAL,
        created_at TEXT,

        FOREIGN KEY (patient_id)
            REFERENCES patients(id)
    )
    """)

    # =================================================
    # AUDIOLOGIST DECISIONS
    # =================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audiologist_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        patient_id INTEGER,
        session_id INTEGER,
        profile_id INTEGER,

        model_version TEXT,
        model_trained_at TEXT,
        recommendation_created_at TEXT,

        ai_stimulus_type TEXT,
        ai_frequency_hz REAL,
        ai_bandwidth_hz REAL,
        ai_relative_level REAL,
        ai_predicted_score REAL,

        decision TEXT,

        final_stimulus_type TEXT,
        final_frequency_hz REAL,
        final_bandwidth_hz REAL,
        final_relative_level REAL,
        final_predicted_score REAL,

        clinician_notes TEXT,

        created_at TEXT,

        FOREIGN KEY (patient_id)
            REFERENCES patients(id),

        FOREIGN KEY (session_id)
            REFERENCES sessions(id),

        FOREIGN KEY (profile_id)
            REFERENCES tinnitus_profiles(id)
    )
    """)

    # =================================================
    # SAFE MIGRATION FOR OLD DATABASES
    # =================================================

    cursor.execute("""
    PRAGMA table_info(stimulus_trials)
    """)

    existing_columns = [
        row[1]
        for row
        in cursor.fetchall()
    ]

    if "session_id" not in existing_columns:

        cursor.execute("""
        ALTER TABLE stimulus_trials
        ADD COLUMN session_id INTEGER
        """)

    if "decision_id" not in existing_columns:

        cursor.execute("""
        ALTER TABLE stimulus_trials
        ADD COLUMN decision_id INTEGER
        """)

    stimulus_trial_provenance_columns = {
        "profile_id":
            "INTEGER",

        "source_type":
            "TEXT",

        "model_version":
            "TEXT",

        "model_trained_at":
            "TEXT",

        "model_predicted_score":
            "REAL",

        "recommendation_created_at":
            "TEXT",

        "tested_at":
            "TEXT"
    }

    for (
        column_name,
        column_type
    ) in stimulus_trial_provenance_columns.items():

        if column_name not in existing_columns:

            cursor.execute(
                f"""
                ALTER TABLE stimulus_trials
                ADD COLUMN {column_name} {column_type}
                """
            )

    # -------------------------------------------------
    # Safe migration for audiologist decision records
    # -------------------------------------------------

    cursor.execute("""
    PRAGMA table_info(audiologist_decisions)
    """)

    decision_columns = [
        row[1]
        for row
        in cursor.fetchall()
    ]

    if "final_predicted_score" not in decision_columns:

        cursor.execute("""
        ALTER TABLE audiologist_decisions
        ADD COLUMN final_predicted_score REAL
        """)

    decision_provenance_columns = {
        "profile_id":
            "INTEGER",

        "model_version":
            "TEXT",

        "model_trained_at":
            "TEXT",

        "recommendation_created_at":
            "TEXT"
    }

    for (
        column_name,
        column_type
    ) in decision_provenance_columns.items():

        if column_name not in decision_columns:

            cursor.execute(
                f"""
                ALTER TABLE audiologist_decisions
                ADD COLUMN {column_name} {column_type}
                """
            )

    # =================================================
    # INDEXES
    # =================================================

    cursor.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS
    idx_users_username
    ON users(username)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS
    idx_users_role
    ON users(role)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS
    idx_users_patient
    ON users(patient_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS
    idx_tinnitus_profiles_patient
    ON tinnitus_profiles(patient_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS
    idx_sessions_patient
    ON sessions(patient_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS
    idx_trials_patient
    ON stimulus_trials(patient_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS
    idx_adaptive_sessions_patient
    ON adaptive_sessions(patient_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS
    idx_adaptive_trials_session
    ON adaptive_session_trials(adaptive_session_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS
    idx_trials_profile
    ON stimulus_trials(profile_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS
    idx_decisions_profile
    ON audiologist_decisions(profile_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS
    idx_trials_session
    ON stimulus_trials(session_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS
    idx_trials_decision
    ON stimulus_trials(decision_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS
    idx_audiograms_patient
    ON audiograms(patient_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS
    idx_audiologist_decisions_patient
    ON audiologist_decisions(patient_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS
    idx_audiologist_decisions_session
    ON audiologist_decisions(session_id)
    """)

    # One audiologist decision should generate at most one
    # linked patient stimulus-trial record.
    cursor.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS
    idx_trials_decision_unique
    ON stimulus_trials(decision_id)
    WHERE decision_id IS NOT NULL
    """)

    conn.commit()
    conn.close()
