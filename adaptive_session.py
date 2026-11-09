import math
from datetime import datetime

import pandas as pd
import streamlit as st

from audio_utils import generate_stimulus
from database import get_connection
from optimization import calculate_response_score
from provenance import get_latest_profile_id, now_timestamp


# ============================================================
# SMARTTINNITUS CLOSED-LOOP ADAPTIVE ACOUSTIC SESSION
# ============================================================
#
# Prototype sequential optimization workflow:
#
# 1. Exploration
# 2. Local exploitation
# 3. Stimulus-type comparison
# 4. Fine tuning
#
# The next acoustic stimulus depends on the patient's recorded
# responses during the same session.
#
# IMPORTANT:
# - This is an engineering / research prototype.
# - It does not diagnose tinnitus.
# - It does not establish treatment effectiveness.
# - Relative level is digital amplitude, not calibrated dB SPL.
# ============================================================


MAX_TRIALS = 6

MIN_FREQUENCY = 250
MAX_FREQUENCY = 10000

MIN_LEVEL = 0.01
MAX_LEVEL = 0.05


# ============================================================
# HELPERS
# ============================================================


def _clip_frequency(value):
    return float(
        max(
            MIN_FREQUENCY,
            min(
                MAX_FREQUENCY,
                float(value)
            )
        )
    )


def _clip_level(value):
    return float(
        max(
            MIN_LEVEL,
            min(
                MAX_LEVEL,
                float(value)
            )
        )
    )


def _round_frequency(
    value,
    step=50
):
    return float(
        round(
            float(value)
            /
            step
        )
        *
        step
    )


def _load_patients():
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            id,
            patient_code
        FROM patients
        ORDER BY patient_code ASC, id ASC
        """,
        conn
    )

    conn.close()

    return df


def _get_latest_profile(
    patient_id
):
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            *
        FROM tinnitus_profiles
        WHERE patient_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        conn,
        params=(
            patient_id,
        )
    )

    conn.close()

    if df.empty:
        return None

    return df.iloc[
        0
    ].to_dict()


def _get_patient_history(
    patient_id
):
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            stimulus_type,
            center_frequency_hz,
            bandwidth_hz,
            relative_level,
            comfort_score,
            tinnitus_noticeability,
            listen_again
        FROM stimulus_trials
        WHERE patient_id = ?
        ORDER BY id ASC
        """,
        conn,
        params=(
            patient_id,
        )
    )

    conn.close()

    if df.empty:
        return df

    scores = []

    for _, row in df.iterrows():
        try:
            score = calculate_response_score(
                row[
                    "comfort_score"
                ],
                row[
                    "tinnitus_noticeability"
                ],
                row[
                    "listen_again"
                ]
            )

        except Exception:
            score = None

        scores.append(
            score
        )

    df[
        "response_score"
    ] = scores

    return df


def _initial_center_frequency(
    profile,
    history_df
):
    tinnitus_pitch = profile.get(
        "estimated_pitch_hz"
    )

    if (
        tinnitus_pitch is None
        or
        pd.isna(
            tinnitus_pitch
        )
    ):
        tinnitus_pitch = 3000.0

    tinnitus_pitch = float(
        tinnitus_pitch
    )

    usable = history_df.copy()

    if usable.empty:
        return _round_frequency(
            tinnitus_pitch
        )

    if (
        "response_score"
        not in usable.columns
    ):
        return _round_frequency(
            tinnitus_pitch
        )

    usable = usable[
        usable[
            "stimulus_type"
        ].isin(
            [
                "Pure Tone",
                "Narrow-Band Noise"
            ]
        )
        &
        usable[
            "center_frequency_hz"
        ].notna()
        &
        usable[
            "response_score"
        ].notna()
    ].copy()

    if len(
        usable
    ) < 3:
        return _round_frequency(
            tinnitus_pitch
        )

    best = usable.sort_values(
        "response_score",
        ascending=False
    ).head(
        min(
            5,
            len(
                usable
            )
        )
    )

    weights = (
        best[
            "response_score"
        ].astype(float)
        -
        float(
            best[
                "response_score"
            ].min()
        )
        +
        1.0
    )

    center = (
        (
            best[
                "center_frequency_hz"
            ].astype(float)
            *
            weights
        ).sum()
        /
        weights.sum()
    )

    return _round_frequency(
        center
    )


def _preferred_level(
    history_df
):
    if history_df.empty:
        return 0.03

    if (
        "response_score"
        not in history_df.columns
    ):
        return 0.03

    valid = history_df[
        history_df[
            "relative_level"
        ].notna()
        &
        history_df[
            "response_score"
        ].notna()
    ].copy()

    if len(
        valid
    ) < 3:
        return 0.03

    top = valid.sort_values(
        "response_score",
        ascending=False
    ).head(
        min(
            5,
            len(
                valid
            )
        )
    )

    return _clip_level(
        round(
            float(
                top[
                    "relative_level"
                ].median()
            ),
            2
        )
    )


def _get_active_session(
    patient_id
):
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            *
        FROM adaptive_sessions
        WHERE patient_id = ?
          AND status = 'active'
        ORDER BY id DESC
        LIMIT 1
        """,
        conn,
        params=(
            patient_id,
        )
    )

    conn.close()

    if df.empty:
        return None

    return df.iloc[
        0
    ].to_dict()


def _get_adaptive_trials(
    adaptive_session_id
):
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            *
        FROM adaptive_session_trials
        WHERE adaptive_session_id = ?
        ORDER BY trial_number ASC
        """,
        conn,
        params=(
            adaptive_session_id,
        )
    )

    conn.close()

    return df


def _start_session(
    patient_id,
    profile,
    history_df
):
    center = _initial_center_frequency(
        profile,
        history_df
    )

    level = _preferred_level(
        history_df
    )

    profile_id = profile.get(
        "id"
    )

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO adaptive_sessions
        (
            patient_id,
            profile_id,
            status,
            max_trials,
            initial_center_frequency_hz,
            initial_relative_level,
            started_at,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            profile_id,
            "active",
            MAX_TRIALS,
            center,
            level,
            now_timestamp(),
            now_timestamp()
        )
    )

    session_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return int(
        session_id
    )


def _best_trial(
    trials_df
):
    if trials_df.empty:
        return None

    valid = trials_df[
        trials_df[
            "response_score"
        ].notna()
    ].copy()

    if valid.empty:
        return None

    return valid.sort_values(
        "response_score",
        ascending=False
    ).iloc[
        0
    ].to_dict()


def _tested_signature_set(
    trials_df
):
    signatures = set()

    if trials_df.empty:
        return signatures

    for _, row in trials_df.iterrows():
        signatures.add(
            (
                str(
                    row[
                        "stimulus_type"
                    ]
                ),
                round(
                    float(
                        row[
                            "center_frequency_hz"
                        ]
                    ),
                    1
                ),
                round(
                    float(
                        row[
                            "bandwidth_hz"
                        ]
                        or
                        0
                    ),
                    1
                ),
                round(
                    float(
                        row[
                            "relative_level"
                        ]
                    ),
                    3
                )
            )
        )

    return signatures


def _candidate_signature(
    candidate
):
    return (
        str(
            candidate[
                "stimulus_type"
            ]
        ),
        round(
            float(
                candidate[
                    "center_frequency_hz"
                ]
            ),
            1
        ),
        round(
            float(
                candidate.get(
                    "bandwidth_hz",
                    0
                )
                or
                0
            ),
            1
        ),
        round(
            float(
                candidate[
                    "relative_level"
                ]
            ),
            3
        )
    )


# ============================================================
# SEQUENTIAL NEXT-STIMULUS LOGIC
# ============================================================


def choose_next_stimulus(
    adaptive_session,
    trials_df
):
    """
    Explainable closed-loop search.

    Trial 1:
        start at personalized center frequency.

    Trial 2:
        explore +500 Hz.

    Trial 3:
        use the first two responses to choose direction:
        - if +500 improved response -> move another +250
        - otherwise explore -500 from initial center.

    Trial 4:
        test Narrow-Band Noise at the best frequency so far.

    Trial 5:
        test White Noise to compare broadband response.

    Trial 6:
        fine-tune the current best configuration:
        - frequency-specific stimuli: ±125 Hz around best region
        - white noise: adapt relative level.
    """

    initial_center = float(
        adaptive_session[
            "initial_center_frequency_hz"
        ]
    )

    initial_level = float(
        adaptive_session[
            "initial_relative_level"
        ]
    )

    completed = len(
        trials_df
    )

    tested = _tested_signature_set(
        trials_df
    )

    # --------------------------------------------------------
    # Trial 1 — baseline center
    # --------------------------------------------------------

    if completed == 0:
        candidate = {
            "phase":
                "Exploration",

            "rationale":
                (
                    "Start at the individualized search center derived "
                    "from the tinnitus pitch and previous response history."
                ),

            "stimulus_type":
                "Pure Tone",

            "center_frequency_hz":
                _clip_frequency(
                    initial_center
                ),

            "bandwidth_hz":
                0.0,

            "relative_level":
                _clip_level(
                    initial_level
                )
        }

        return candidate

    # --------------------------------------------------------
    # Trial 2 — broad exploration above center
    # --------------------------------------------------------

    if completed == 1:
        candidate = {
            "phase":
                "Exploration",

            "rationale":
                (
                    "Explore a higher frequency to determine whether "
                    "the response improves above the initial center."
                ),

            "stimulus_type":
                "Pure Tone",

            "center_frequency_hz":
                _round_frequency(
                    _clip_frequency(
                        initial_center
                        +
                        500
                    )
                ),

            "bandwidth_hz":
                0.0,

            "relative_level":
                _clip_level(
                    initial_level
                )
        }

        return candidate

    # --------------------------------------------------------
    # Trial 3 — response-directed frequency move
    # --------------------------------------------------------

    if completed == 2:
        trial_1 = trials_df.iloc[
            0
        ]

        trial_2 = trials_df.iloc[
            1
        ]

        score_1 = float(
            trial_1[
                "response_score"
            ]
        )

        score_2 = float(
            trial_2[
                "response_score"
            ]
        )

        if score_2 > score_1:
            next_frequency = (
                float(
                    trial_2[
                        "center_frequency_hz"
                    ]
                )
                +
                250
            )

            rationale = (
                "The higher-frequency trial produced the stronger "
                "response, so the search continues in that direction "
                "with a smaller step."
            )

        else:
            next_frequency = (
                initial_center
                -
                500
            )

            rationale = (
                "The higher-frequency trial did not improve the response, "
                "so the system explores below the initial center."
            )

        candidate = {
            "phase":
                "Directed Exploration",

            "rationale":
                rationale,

            "stimulus_type":
                "Pure Tone",

            "center_frequency_hz":
                _round_frequency(
                    _clip_frequency(
                        next_frequency
                    )
                ),

            "bandwidth_hz":
                0.0,

            "relative_level":
                _clip_level(
                    initial_level
                )
        }

        return candidate

    # --------------------------------------------------------
    # Trial 4 — compare NBN at best frequency
    # --------------------------------------------------------

    if completed == 3:
        best = _best_trial(
            trials_df
        )

        best_frequency = float(
            best[
                "center_frequency_hz"
            ]
        )

        candidate = {
            "phase":
                "Stimulus Comparison",

            "rationale":
                (
                    "The strongest frequency found so far is now tested "
                    "with Narrow-Band Noise to compare stimulus type while "
                    "preserving the promising frequency region."
                ),

            "stimulus_type":
                "Narrow-Band Noise",

            "center_frequency_hz":
                _round_frequency(
                    best_frequency
                ),

            "bandwidth_hz":
                500.0,

            "relative_level":
                _clip_level(
                    float(
                        best[
                            "relative_level"
                        ]
                    )
                )
        }

        return candidate

    # --------------------------------------------------------
    # Trial 5 — broadband comparison
    # --------------------------------------------------------

    if completed == 4:
        best = _best_trial(
            trials_df
        )

        candidate = {
            "phase":
                "Stimulus Comparison",

            "rationale":
                (
                    "A broadband White Noise trial is introduced to compare "
                    "the patient's response with the frequency-specific "
                    "stimuli tested earlier."
                ),

            "stimulus_type":
                "White Noise",

            # Internal model/audio placeholder only.
            "center_frequency_hz":
                _round_frequency(
                    float(
                        best[
                            "center_frequency_hz"
                        ]
                    )
                ),

            "bandwidth_hz":
                0.0,

            "relative_level":
                _clip_level(
                    float(
                        best[
                            "relative_level"
                        ]
                    )
                )
        }

        return candidate

    # --------------------------------------------------------
    # Trial 6 — fine tuning around current best
    # --------------------------------------------------------

    best = _best_trial(
        trials_df
    )

    best_type = str(
        best[
            "stimulus_type"
        ]
    )

    best_frequency = float(
        best[
            "center_frequency_hz"
        ]
    )

    best_level = float(
        best[
            "relative_level"
        ]
    )

    comfort = float(
        best[
            "comfort_score"
        ]
    )

    noticeability = float(
        best[
            "tinnitus_noticeability"
        ]
    )

    if best_type == "White Noise":
        if comfort < 6:
            next_level = _clip_level(
                best_level
                -
                0.01
            )

            rationale = (
                "White Noise currently performs best, but comfort is limited. "
                "The final trial lowers the digital level to improve tolerability."
            )

        elif noticeability > 5:
            next_level = _clip_level(
                best_level
                +
                0.01
            )

            rationale = (
                "White Noise currently performs best and comfort is acceptable, "
                "but tinnitus noticeability remains high. The final trial "
                "slightly increases the digital level."
            )

        else:
            next_level = _clip_level(
                best_level
            )

            rationale = (
                "White Noise currently performs best with acceptable comfort "
                "and noticeability, so the final trial confirms the same region."
            )

        candidate = {
            "phase":
                "Fine Tuning",

            "rationale":
                rationale,

            "stimulus_type":
                "White Noise",

            "center_frequency_hz":
                _round_frequency(
                    best_frequency
                ),

            "bandwidth_hz":
                0.0,

            "relative_level":
                next_level
        }

    else:
        frequencies = [
            best_frequency
            -
            125,
            best_frequency
            +
            125
        ]

        candidate_frequency = None

        for frequency in frequencies:
            candidate = {
                "stimulus_type":
                    best_type,

                "center_frequency_hz":
                    _round_frequency(
                        _clip_frequency(
                            frequency
                        ),
                        step=25
                    ),

                "bandwidth_hz":
                    (
                        float(
                            best[
                                "bandwidth_hz"
                            ]
                        )
                        if best_type
                        ==
                        "Narrow-Band Noise"
                        else 0.0
                    ),

                "relative_level":
                    _clip_level(
                        best_level
                    )
            }

            if (
                _candidate_signature(
                    candidate
                )
                not in tested
            ):
                candidate_frequency = (
                    candidate[
                        "center_frequency_hz"
                    ]
                )
                break

        if candidate_frequency is None:
            candidate_frequency = (
                _round_frequency(
                    _clip_frequency(
                        best_frequency
                        +
                        125
                    ),
                    step=25
                )
            )

        candidate = {
            "phase":
                "Fine Tuning",

            "rationale":
                (
                    "The final trial performs a small local frequency adjustment "
                    "around the strongest configuration found during this session."
                ),

            "stimulus_type":
                best_type,

            "center_frequency_hz":
                candidate_frequency,

            "bandwidth_hz":
                (
                    float(
                        best[
                            "bandwidth_hz"
                        ]
                    )
                    if best_type
                    ==
                    "Narrow-Band Noise"
                    else 0.0
                ),

            "relative_level":
                _clip_level(
                    best_level
                )
        }

    return candidate


# ============================================================
# SAVE TRIAL + MIRROR INTO stimulus_trials
# ============================================================


def save_adaptive_trial(
    adaptive_session,
    candidate,
    comfort,
    noticeability,
    listen_again
):
    patient_id = int(
        adaptive_session[
            "patient_id"
        ]
    )

    adaptive_session_id = int(
        adaptive_session[
            "id"
        ]
    )

    profile_id = adaptive_session.get(
        "profile_id"
    )

    trials_df = _get_adaptive_trials(
        adaptive_session_id
    )

    trial_number = (
        len(
            trials_df
        )
        +
        1
    )

    score = calculate_response_score(
        comfort,
        noticeability,
        listen_again
    )

    timestamp = now_timestamp()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO adaptive_session_trials
        (
            adaptive_session_id,
            patient_id,
            profile_id,
            trial_number,
            phase,
            rationale,
            stimulus_type,
            center_frequency_hz,
            bandwidth_hz,
            relative_level,
            comfort_score,
            tinnitus_noticeability,
            listen_again,
            response_score,
            tested_at,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            adaptive_session_id,
            patient_id,
            profile_id,
            trial_number,
            candidate[
                "phase"
            ],
            candidate[
                "rationale"
            ],
            candidate[
                "stimulus_type"
            ],
            candidate[
                "center_frequency_hz"
            ],
            candidate[
                "bandwidth_hz"
            ],
            candidate[
                "relative_level"
            ],
            comfort,
            noticeability,
            listen_again,
            score,
            timestamp,
            timestamp
        )
    )

    adaptive_trial_id = (
        cursor.lastrowid
    )

    cursor.execute(
        """
        INSERT INTO stimulus_trials
        (
            patient_id,
            session_id,
            decision_id,
            profile_id,
            adaptive_session_id,
            source_type,
            tested_at,
            stimulus_type,
            center_frequency_hz,
            bandwidth_hz,
            relative_level,
            comfort_score,
            tinnitus_noticeability,
            listen_again,
            created_at
        )
        VALUES
        (?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            profile_id,
            adaptive_session_id,
            "adaptive_session",
            timestamp,
            candidate[
                "stimulus_type"
            ],
            candidate[
                "center_frequency_hz"
            ],
            candidate[
                "bandwidth_hz"
            ],
            candidate[
                "relative_level"
            ],
            comfort,
            noticeability,
            listen_again,
            timestamp
        )
    )

    linked_stimulus_trial_id = (
        cursor.lastrowid
    )

    cursor.execute(
        """
        UPDATE adaptive_session_trials
        SET linked_stimulus_trial_id = ?
        WHERE id = ?
        """,
        (
            linked_stimulus_trial_id,
            adaptive_trial_id
        )
    )

    conn.commit()
    conn.close()

    return float(
        score
    )


# ============================================================
# COMPLETE SESSION
# ============================================================


def complete_adaptive_session(
    adaptive_session_id
):
    trials_df = _get_adaptive_trials(
        adaptive_session_id
    )

    if trials_df.empty:
        return None

    best = _best_trial(
        trials_df
    )

    frequency_trials = trials_df[
        trials_df[
            "stimulus_type"
        ].isin(
            [
                "Pure Tone",
                "Narrow-Band Noise"
            ]
        )
    ].copy()

    if frequency_trials.empty:
        promising_min = None
        promising_max = None

    else:
        top = frequency_trials.sort_values(
            "response_score",
            ascending=False
        ).head(
            min(
                3,
                len(
                    frequency_trials
                )
            )
        )

        center = float(
            top[
                "center_frequency_hz"
            ].median()
        )

        promising_min = _round_frequency(
            _clip_frequency(
                center
                -
                250
            )
        )

        promising_max = _round_frequency(
            _clip_frequency(
                center
                +
                250
            )
        )

    summary = (
        f"Best observed configuration: "
        f"{best['stimulus_type']} | "
        f"{float(best['center_frequency_hz']):.0f} Hz | "
        f"level {float(best['relative_level']):.2f} | "
        f"response {float(best['response_score']):.1f}/100"
    )

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE adaptive_sessions
        SET
            status = 'completed',
            completed_at = ?,
            best_adaptive_trial_id = ?,
            best_response_score = ?,
            promising_frequency_min_hz = ?,
            promising_frequency_max_hz = ?,
            summary_text = ?
        WHERE id = ?
        """,
        (
            now_timestamp(),
            int(
                best[
                    "id"
                ]
            ),
            float(
                best[
                    "response_score"
                ]
            ),
            promising_min,
            promising_max,
            summary,
            adaptive_session_id
        )
    )

    conn.commit()
    conn.close()

    return {
        "best":
            best,

        "promising_min":
            promising_min,

        "promising_max":
            promising_max,

        "summary":
            summary
    }


# ============================================================
# SESSION HISTORY
# ============================================================


def _load_session_history(
    patient_id
):
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            id,
            status,
            started_at,
            completed_at,
            best_response_score,
            promising_frequency_min_hz,
            promising_frequency_max_hz,
            summary_text
        FROM adaptive_sessions
        WHERE patient_id = ?
        ORDER BY id DESC
        """,
        conn,
        params=(
            patient_id,
        )
    )

    conn.close()

    return df


# ============================================================
# UI
# ============================================================


def show_adaptive_session_page(
    current_role,
    auth_patient_id=None
):
    st.title(
        "Closed-Loop Adaptive Acoustic Session"
    )

    st.caption(
        "Sequential prototype optimization: each next sound is selected "
        "using responses collected earlier in the same session."
    )

    st.warning(
        "Research prototype only. The session optimizes a prototype "
        "response score and does not establish treatment effectiveness."
    )

    # --------------------------------------------------------
    # Resolve patient
    # --------------------------------------------------------

    if current_role == "patient":
        if auth_patient_id is None:
            st.error(
                "This login is not linked to a patient record."
            )

            return

        patient_id = int(
            auth_patient_id
        )

        patients_df = _load_patients()

        row = patients_df[
            patients_df[
                "id"
            ]
            ==
            patient_id
        ]

        if row.empty:
            st.error(
                "The linked patient record could not be found."
            )

            return

        patient_code = str(
            row.iloc[
                0
            ][
                "patient_code"
            ]
        )

        st.caption(
            f"Logged in as Patient {patient_code}"
        )

    else:
        patients_df = _load_patients()

        if patients_df.empty:
            st.info(
                "No patients are available."
            )

            return

        options = {
            f"{row['patient_code']} (ID {int(row['id'])})":
                int(
                    row[
                        "id"
                    ]
                )
            for _, row in patients_df.iterrows()
        }

        selected = st.selectbox(
            "Patient",
            list(
                options.keys()
            ),
            key="adaptive_session_patient"
        )

        patient_id = options[
            selected
        ]

        patient_code = selected.split(
            " (ID "
        )[0]

    # --------------------------------------------------------
    # Profile requirement
    # --------------------------------------------------------

    profile = _get_latest_profile(
        patient_id
    )

    if profile is None:
        st.info(
            "Complete the tinnitus assessment before starting "
            "an adaptive acoustic session."
        )

        return

    pitch = profile.get(
        "estimated_pitch_hz"
    )

    if (
        pitch is None
        or
        pd.isna(
            pitch
        )
    ):
        st.info(
            "An estimated tinnitus pitch is required before starting "
            "the adaptive session."
        )

        return

    history_df = _get_patient_history(
        patient_id
    )

    active_session = _get_active_session(
        patient_id
    )

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    if active_session is None:
        st.markdown(
            "## Start a New Adaptive Session"
        )

        center = _initial_center_frequency(
            profile,
            history_df
        )

        level = _preferred_level(
            history_df
        )

        c1, c2, c3 = st.columns(
            3
        )

        c1.metric(
            "Estimated Tinnitus Pitch",
            f"{float(pitch):.0f} Hz"
        )

        c2.metric(
            "Starting Search Center",
            f"{float(center):.0f} Hz"
        )

        c3.metric(
            "Starting Relative Level",
            f"{float(level):.2f}"
        )

        st.write(
            "The session contains up to "
            f"**{MAX_TRIALS} sequential trials**. "
            "The next trial changes according to the patient's previous responses."
        )

        if st.button(
            "▶ Start Adaptive Session",
            use_container_width=True,
            key="start_adaptive_session"
        ):
            _start_session(
                patient_id,
                profile,
                history_df
            )

            st.rerun()

    # --------------------------------------------------------
    # ACTIVE SESSION
    # --------------------------------------------------------

    else:
        adaptive_session_id = int(
            active_session[
                "id"
            ]
        )

        trials_df = _get_adaptive_trials(
            adaptive_session_id
        )

        completed = len(
            trials_df
        )

        st.markdown(
            "## Session in Progress"
        )

        p1, p2, p3 = st.columns(
            3
        )

        p1.metric(
            "Session ID",
            adaptive_session_id
        )

        p2.metric(
            "Completed Trials",
            f"{completed} / {MAX_TRIALS}"
        )

        best = _best_trial(
            trials_df
        )

        p3.metric(
            "Best Response So Far",
            (
                f"{float(best['response_score']):.1f} / 100"
                if best is not None
                else "—"
            )
        )

        progress = min(
            1.0,
            completed
            /
            MAX_TRIALS
        )

        st.progress(
            progress
        )

        # ----------------------------------------------------
        # If complete, finalize
        # ----------------------------------------------------

        if completed >= MAX_TRIALS:
            result = complete_adaptive_session(
                adaptive_session_id
            )

            if result is not None:
                st.success(
                    "Adaptive session completed."
                )

            st.rerun()

        # ----------------------------------------------------
        # Next stimulus
        # ----------------------------------------------------

        candidate = choose_next_stimulus(
            active_session,
            trials_df
        )

        st.markdown(
            f"### Trial {completed + 1}: {candidate['phase']}"
        )

        st.info(
            candidate[
                "rationale"
            ]
        )

        c1, c2, c3, c4 = st.columns(
            4
        )

        c1.metric(
            "Stimulus",
            candidate[
                "stimulus_type"
            ]
        )

        if (
            candidate[
                "stimulus_type"
            ]
            ==
            "White Noise"
        ):
            frequency_display = (
                "Broadband"
            )

        else:
            frequency_display = (
                f"{float(candidate['center_frequency_hz']):.0f} Hz"
            )

        c2.metric(
            "Frequency",
            frequency_display
        )

        if (
            candidate[
                "stimulus_type"
            ]
            ==
            "Narrow-Band Noise"
        ):
            bandwidth_display = (
                f"{float(candidate['bandwidth_hz']):.0f} Hz"
            )

        else:
            bandwidth_display = (
                "N/A"
            )

        c3.metric(
            "Bandwidth",
            bandwidth_display
        )

        c4.metric(
            "Relative Level",
            f"{float(candidate['relative_level']):.2f}"
        )

        if st.button(
            "▶ Listen to Current Stimulus",
            use_container_width=True,
            key=f"adaptive_play_{adaptive_session_id}_{completed + 1}"
        ):
            try:
                audio = generate_stimulus(
                    stimulus_type=(
                        candidate[
                            "stimulus_type"
                        ]
                    ),
                    frequency=float(
                        candidate[
                            "center_frequency_hz"
                        ]
                    ),
                    duration=2,
                    volume=float(
                        candidate[
                            "relative_level"
                        ]
                    ),
                    bandwidth=(
                        float(
                            candidate[
                                "bandwidth_hz"
                            ]
                        )
                        if candidate[
                            "stimulus_type"
                        ]
                        ==
                        "Narrow-Band Noise"
                        else 500
                    )
                )

                st.audio(
                    audio,
                    format="audio/wav"
                )

            except Exception as error:
                st.error(
                    f"Could not generate stimulus: {error}"
                )

        st.markdown(
            "### Patient Feedback"
        )

        with st.form(
            f"adaptive_feedback_{adaptive_session_id}_{completed + 1}"
        ):
            f1, f2 = st.columns(
                2
            )

            comfort = f1.slider(
                "Comfort",
                0,
                10,
                5,
                key=f"adaptive_comfort_{adaptive_session_id}_{completed + 1}"
            )

            noticeability = f2.slider(
                "Tinnitus Noticeability",
                0,
                10,
                5,
                key=f"adaptive_notice_{adaptive_session_id}_{completed + 1}"
            )

            listen_again = st.radio(
                "Would you listen to this stimulus again?",
                [
                    "Yes",
                    "No"
                ],
                horizontal=True,
                key=f"adaptive_again_{adaptive_session_id}_{completed + 1}"
            )

            submit = st.form_submit_button(
                "Submit Feedback & Generate Next Trial",
                use_container_width=True
            )

        if submit:
            score = save_adaptive_trial(
                active_session,
                candidate,
                comfort,
                noticeability,
                listen_again
            )

            st.success(
                f"Trial saved. Prototype response score: {score:.1f} / 100"
            )

            st.rerun()

        # ----------------------------------------------------
        # Session progress table
        # ----------------------------------------------------

        if not trials_df.empty:
            st.markdown(
                "### Session Progress"
            )

            progress_df = trials_df[
                [
                    "trial_number",
                    "phase",
                    "stimulus_type",
                    "center_frequency_hz",
                    "bandwidth_hz",
                    "relative_level",
                    "comfort_score",
                    "tinnitus_noticeability",
                    "response_score"
                ]
            ].copy()

            progress_df.columns = [
                "Trial",
                "Phase",
                "Stimulus",
                "Frequency (Hz)",
                "Bandwidth (Hz)",
                "Level",
                "Comfort",
                "Noticeability",
                "Response /100"
            ]

            progress_df[
                "Response /100"
            ] = progress_df[
                "Response /100"
            ].round(
                1
            )

            st.dataframe(
                progress_df,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # SESSION HISTORY
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown(
        "## Adaptive Session History"
    )

    history = _load_session_history(
        patient_id
    )

    if history.empty:
        st.info(
            "No adaptive sessions have been recorded yet."
        )

    else:
        display = history.copy()

        display.columns = [
            "Session ID",
            "Status",
            "Started",
            "Completed",
            "Best Response /100",
            "Promising Min Hz",
            "Promising Max Hz",
            "Summary"
        ]

        display[
            "Best Response /100"
        ] = pd.to_numeric(
            display[
                "Best Response /100"
            ],
            errors="coerce"
        ).round(
            1
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        completed_history = history[
            history[
                "status"
            ]
            ==
            "completed"
        ]

        if not completed_history.empty:
            latest = completed_history.iloc[
                0
            ]

            st.markdown(
                "### Latest Completed Session"
            )

            c1, c2, c3 = st.columns(
                3
            )

            c1.metric(
                "Best Response",
                (
                    f"{float(latest['best_response_score']):.1f} / 100"
                    if pd.notna(
                        latest[
                            "best_response_score"
                        ]
                    )
                    else "—"
                )
            )

            c2.metric(
                "Promising Region",
                (
                    f"{float(latest['promising_frequency_min_hz']):.0f}–"
                    f"{float(latest['promising_frequency_max_hz']):.0f} Hz"
                    if
                    pd.notna(
                        latest[
                            "promising_frequency_min_hz"
                        ]
                    )
                    and
                    pd.notna(
                        latest[
                            "promising_frequency_max_hz"
                        ]
                    )
                    else "N/A"
                )
            )

            c3.metric(
                "Status",
                "Completed"
            )

            st.caption(
                _safe_text(
                    latest[
                        "summary_text"
                    ]
                )
            )


def _safe_text(
    value
):
    if value is None:
        return "No summary available."

    try:
        if pd.isna(
            value
        ):
            return "No summary available."
    except Exception:
        pass

    return str(
        value
    )
