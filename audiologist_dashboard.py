import os
from datetime import datetime

import pandas as pd
import streamlit as st

from database import get_connection

from adaptive_personalization import (
    generate_adaptive_candidate_stimuli,
    get_adaptive_personalization_summary
)

from provenance import (
    get_latest_profile_id,
    get_model_provenance,
    now_timestamp
)

from optimization import (
    calculate_response_score
)

from audiogram import (
    AUDIOGRAM_FREQUENCIES,
    plot_audiogram
)

from report import (
    generate_patient_report
)

from ml_model import (
    MODEL_FILE,
    load_model,
    model_exists,
    get_saved_model_metrics,
    get_ml_dataset_summary,
    predict_responses_batch
)


# ============================================================
# CACHE CURRENT AI MODEL
# ============================================================


@st.cache_resource
def get_audiologist_model(
    model_modification_time
):

    return load_model()


# ============================================================
# HELPERS
# ============================================================


def safe_value(
    value,
    default="Not available"
):

    if value is None:

        return default

    return value


def safe_integer(
    value
):

    if value is None:

        return "Not measured"

    try:

        return str(
            int(
                round(
                    float(value)
                )
            )
        )

    except Exception:

        return str(value)


def safe_float_display(
    value,
    decimals=2
):

    if value is None:

        return "Not measured"

    try:

        return f"{float(value):.{decimals}f}"

    except Exception:

        return str(value)


# ============================================================
# DATABASE READ FUNCTIONS
# ============================================================


def get_patients(
    conn
):

    query = """
    SELECT
        id,
        patient_code,
        age,
        sex,
        created_at
    FROM patients
    ORDER BY patient_code
    """

    return pd.read_sql_query(
        query,
        conn
    )


def get_latest_profile(
    conn,
    patient_id
):

    query = """
    SELECT
        id,
        patient_id,
        laterality,
        tinnitus_type,
        temporal_pattern,
        duration_months,
        annoyance_score,
        sleep_score,
        concentration_score,
        stress_score,
        estimated_pitch_hz,
        loudness_relative,
        created_at
    FROM tinnitus_profiles
    WHERE patient_id = ?
    ORDER BY id DESC
    LIMIT 1
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(
            patient_id,
        )
    )

    if df.empty:

        return None

    return df.iloc[
        0
    ].to_dict()


def get_trials(
    conn,
    patient_id
):

    query = """
    SELECT
        stimulus_trials.id
            AS trial_id,

        stimulus_trials.session_id,
        stimulus_trials.decision_id,
        stimulus_trials.profile_id,
        stimulus_trials.source_type,
        stimulus_trials.model_version,
        stimulus_trials.model_trained_at,
        stimulus_trials.model_predicted_score,
        stimulus_trials.recommendation_created_at,
        stimulus_trials.tested_at,

        stimulus_trials.stimulus_type,
        stimulus_trials.center_frequency_hz,
        stimulus_trials.bandwidth_hz,
        stimulus_trials.relative_level,
        stimulus_trials.comfort_score,
        stimulus_trials.tinnitus_noticeability,
        stimulus_trials.listen_again,
        stimulus_trials.created_at,

        sessions.session_date

    FROM stimulus_trials

    LEFT JOIN sessions
        ON stimulus_trials.session_id =
           sessions.id

    WHERE stimulus_trials.patient_id = ?

    ORDER BY stimulus_trials.id DESC
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(
            patient_id,
        )
    )

    if df.empty:

        return df

    scores = []

    for _, row in df.iterrows():

        score = (
            calculate_response_score(
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
        )

        scores.append(
            score
        )

    df[
        "response_score"
    ] = scores

    return df


def get_sessions(
    conn,
    patient_id
):

    query = """
    SELECT
        id AS session_id,
        session_date,
        annoyance_before,
        annoyance_after,
        tinnitus_noticeability_before,
        tinnitus_noticeability_after,
        notes,
        created_at
    FROM sessions
    WHERE patient_id = ?
    ORDER BY session_date ASC, id ASC
    """

    return pd.read_sql_query(
        query,
        conn,
        params=(
            patient_id,
        )
    )


def get_audiogram(
    conn,
    patient_id
):

    query = """
    SELECT
        ear,
        frequency_hz,
        threshold_db_hl
    FROM audiograms
    WHERE patient_id = ?
    ORDER BY frequency_hz
    """

    return pd.read_sql_query(
        query,
        conn,
        params=(
            patient_id,
        )
    )


def get_audiologist_decisions(
    conn,
    patient_id
):

    query = """
    SELECT
        audiologist_decisions.id
            AS decision_id,

        audiologist_decisions.session_id,
        audiologist_decisions.profile_id,
        audiologist_decisions.model_version,
        audiologist_decisions.model_trained_at,
        audiologist_decisions.recommendation_created_at,

        sessions.session_date,

        audiologist_decisions.ai_stimulus_type,
        audiologist_decisions.ai_frequency_hz,
        audiologist_decisions.ai_bandwidth_hz,
        audiologist_decisions.ai_relative_level,
        audiologist_decisions.ai_predicted_score,

        audiologist_decisions.decision,

        audiologist_decisions.final_stimulus_type,
        audiologist_decisions.final_frequency_hz,
        audiologist_decisions.final_bandwidth_hz,
        audiologist_decisions.final_relative_level,
        audiologist_decisions.final_predicted_score,

        audiologist_decisions.clinician_notes,

        audiologist_decisions.created_at

    FROM audiologist_decisions

    LEFT JOIN sessions
        ON audiologist_decisions.session_id =
           sessions.id

    WHERE audiologist_decisions.patient_id = ?

    ORDER BY audiologist_decisions.id DESC
    """

    return pd.read_sql_query(
        query,
        conn,
        params=(
            patient_id,
        )
    )


# ============================================================
# AUDIOGRAM
# ============================================================


def prepare_audiogram_values(
    audiogram_df
):

    if audiogram_df.empty:

        return (
            None,
            None
        )

    right = []
    left = []

    for frequency in AUDIOGRAM_FREQUENCIES:

        right_match = (
            audiogram_df[
                (
                    audiogram_df[
                        "ear"
                    ]
                    ==
                    "Right"
                )
                &
                (
                    audiogram_df[
                        "frequency_hz"
                    ]
                    ==
                    frequency
                )
            ]
        )

        left_match = (
            audiogram_df[
                (
                    audiogram_df[
                        "ear"
                    ]
                    ==
                    "Left"
                )
                &
                (
                    audiogram_df[
                        "frequency_hz"
                    ]
                    ==
                    frequency
                )
            ]
        )

        if right_match.empty:

            right.append(
                None
            )

        else:

            right.append(
                float(
                    right_match.iloc[
                        0
                    ][
                        "threshold_db_hl"
                    ]
                )
            )

        if left_match.empty:

            left.append(
                None
            )

        else:

            left.append(
                float(
                    left_match.iloc[
                        0
                    ][
                        "threshold_db_hl"
                    ]
                )
            )

    return (
        right,
        left
    )


# ============================================================
# AI PATIENT FEATURES
# ============================================================


def create_patient_features(
    patient,
    profile
):

    return {
        "age":
            patient.get(
                "age"
            ),

        "sex":
            patient.get(
                "sex"
            ),

        "laterality":
            profile.get(
                "laterality"
            ),

        "tinnitus_type":
            profile.get(
                "tinnitus_type"
            ),

        "temporal_pattern":
            profile.get(
                "temporal_pattern"
            ),

        "duration_months":
            profile.get(
                "duration_months"
            ),

        "annoyance_score":
            profile.get(
                "annoyance_score"
            ),

        "sleep_score":
            profile.get(
                "sleep_score"
            ),

        "concentration_score":
            profile.get(
                "concentration_score"
            ),

        "stress_score":
            profile.get(
                "stress_score"
            ),

        "estimated_pitch_hz":
            profile.get(
                "estimated_pitch_hz"
            ),

        "loudness_relative":
            profile.get(
                "loudness_relative"
            )
    }


# ============================================================
# AI CANDIDATE GENERATION
# ============================================================


def generate_candidate_stimuli(
    tinnitus_pitch,
    trials_df=None
):

    (
        candidates,
        metadata
    ) = generate_adaptive_candidate_stimuli(
        tinnitus_pitch=tinnitus_pitch,
        trials_df=trials_df
    )

    return (
        candidates,
        metadata
    )


# ============================================================
# AI RECOMMENDATIONS
# ============================================================


def get_ai_recommendations(
    patient_features,
    tinnitus_pitch,
    trials_df=None
):

    if not model_exists():

        return None

    model_modification_time = (
        os.path.getmtime(
            MODEL_FILE
        )
    )

    model = (
        get_audiologist_model(
            model_modification_time
        )
    )

    (
        candidates,
        adaptive_metadata
    ) = (
        generate_candidate_stimuli(
            tinnitus_pitch,
            trials_df=trials_df
        )
    )

    predictions = (
        predict_responses_batch(
            patient_features,
            candidates,
            model=model
        )
    )

    rows = []

    for candidate, prediction in zip(
        candidates,
        predictions
    ):

        stimulus_type = (
            candidate[
                "stimulus_type"
            ]
        )

        if (
            stimulus_type
            ==
            "White Noise"
        ):

            frequency_display = (
                "Broadband"
            )

            bandwidth_display = (
                None
            )

        else:

            frequency_display = int(
                round(
                    candidate[
                        "center_frequency_hz"
                    ]
                )
            )

            bandwidth_display = (
                int(
                    round(
                        candidate[
                            "bandwidth_hz"
                        ]
                    )
                )
                if stimulus_type
                ==
                "Narrow-Band Noise"
                else
                None
            )

        rows.append(
            {
                "Stimulus":
                    stimulus_type,

                "Frequency (Hz)":
                    frequency_display,

                "Bandwidth (Hz)":
                    bandwidth_display,

                "Relative Level":
                    candidate[
                        "relative_level"
                    ],

                "Predicted Score":
                    float(
                        prediction
                    ),

                "_center_frequency_hz":
                    candidate[
                        "center_frequency_hz"
                    ],

                "_bandwidth_hz":
                    candidate[
                        "bandwidth_hz"
                    ]
            }
        )

    df = pd.DataFrame(
        rows
    )

    df = (
        df.sort_values(
            "Predicted Score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return (
        df,
        adaptive_metadata
    )


# ============================================================
# PREDICT EXACT CLINICIAN-REVIEWED CONFIGURATION
# ============================================================


def predict_final_configuration(
    patient_features,
    stimulus_type,
    frequency_hz,
    bandwidth_hz,
    relative_level
):

    if (
        stimulus_type is None
        or
        frequency_hz is None
        or
        relative_level is None
    ):

        return None

    if not model_exists():

        return None

    if (
        stimulus_type
        ==
        "Narrow-Band Noise"
    ):

        model_bandwidth = (
            float(
                bandwidth_hz
            )
            if bandwidth_hz is not None
            else 500.0
        )

    else:

        # The model schema requires a numeric bandwidth.
        # It is not acoustically interpreted for Pure Tone
        # or White Noise.
        model_bandwidth = 0.0

    candidate = {
        "stimulus_type":
            stimulus_type,

        "center_frequency_hz":
            float(
                frequency_hz
            ),

        "bandwidth_hz":
            float(
                model_bandwidth
            ),

        "relative_level":
            float(
                relative_level
            )
    }

    model_modification_time = (
        os.path.getmtime(
            MODEL_FILE
        )
    )

    model = (
        get_audiologist_model(
            model_modification_time
        )
    )

    prediction = (
        predict_responses_batch(
            patient_features,
            [
                candidate
            ],
            model=model
        )
    )

    if len(
        prediction
    ) == 0:

        return None

    return float(
        prediction[
            0
        ]
    )


# ============================================================
# SAVE AUDIOLOGIST DECISION
# ============================================================


def save_audiologist_decision(
    conn,
    patient_id,
    session_id,
    ai_recommendation,
    decision,
    final_stimulus_type,
    final_frequency_hz,
    final_bandwidth_hz,
    final_relative_level,
    final_predicted_score,
    clinician_notes
):

    cursor = conn.cursor()

    ai_bandwidth = (
        ai_recommendation[
            "_bandwidth_hz"
        ]
    )

    if (
        ai_recommendation[
            "Stimulus"
        ]
        !=
        "Narrow-Band Noise"
    ):

        ai_bandwidth = (
            None
        )

    if pd.isna(
        ai_bandwidth
    ):

        ai_bandwidth = None

    profile_id = get_latest_profile_id(
        conn,
        patient_id
    )

    model_provenance = get_model_provenance(
        MODEL_FILE,
        get_saved_model_metrics()
    )

    recommendation_created_at = (
        now_timestamp()
    )

    decision_created_at = (
        now_timestamp()
    )

    cursor.execute(
        """
        INSERT INTO audiologist_decisions
        (
            patient_id,
            session_id,
            profile_id,
            model_version,
            model_trained_at,
            recommendation_created_at,

            ai_stimulus_type,
            ai_frequency_hz,
            ai_bandwidth_hz,
            ai_relative_level,
            ai_predicted_score,

            decision,

            final_stimulus_type,
            final_frequency_hz,
            final_bandwidth_hz,
            final_relative_level,
            final_predicted_score,

            clinician_notes,

            created_at
        )

        VALUES
        (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?,
            ?, ?, ?, ?, ?,
            ?,
            ?
        )
        """,
        (
            patient_id,
            session_id,
            profile_id,
            model_provenance[
                "model_version"
            ],
            model_provenance[
                "model_trained_at"
            ],
            recommendation_created_at,

            ai_recommendation[
                "Stimulus"
            ],

            float(
                ai_recommendation[
                    "_center_frequency_hz"
                ]
            ),

            ai_bandwidth,

            float(
                ai_recommendation[
                    "Relative Level"
                ]
            ),

            float(
                ai_recommendation[
                    "Predicted Score"
                ]
            ),

            decision,

            final_stimulus_type,
            final_frequency_hz,
            final_bandwidth_hz,
            final_relative_level,
            final_predicted_score,

            clinician_notes,

            decision_created_at
        )
    )

    conn.commit()



# ============================================================
# BUILD ACOUSTIC FINGERPRINT
# ============================================================
#
# This is a compact descriptive summary of the patient's
# tinnitus characteristics and recorded acoustic-response
# history.
#
# It is NOT a diagnostic fingerprint and NOT a validated
# clinical biomarker.
# ============================================================


def build_acoustic_fingerprint(
    profile,
    trials_df,
    decisions_df
):

    fingerprint = {
        "tinnitus_pitch":
            None,

        "relative_loudness":
            None,

        "tinnitus_type":
            None,

        "laterality":
            None,

        "best_stimulus":
            None,

        "best_frequency":
            None,

        "best_bandwidth":
            None,

        "best_level":
            None,

        "average_comfort":
            None,

        "average_noticeability":
            None,

        "best_response_score":
            None,

        "latest_clinician_decision":
            None,

        "latest_clinician_stimulus":
            None,

        "latest_clinician_frequency":
            None,

        "latest_clinician_bandwidth":
            None,

        "latest_clinician_level":
            None
    }

    # --------------------------------------------------------
    # Tinnitus profile
    # --------------------------------------------------------

    if profile is not None:

        fingerprint[
            "tinnitus_pitch"
        ] = profile.get(
            "estimated_pitch_hz"
        )

        fingerprint[
            "relative_loudness"
        ] = profile.get(
            "loudness_relative"
        )

        fingerprint[
            "tinnitus_type"
        ] = profile.get(
            "tinnitus_type"
        )

        fingerprint[
            "laterality"
        ] = profile.get(
            "laterality"
        )

    # --------------------------------------------------------
    # Recorded acoustic responses
    # --------------------------------------------------------

    if (
        trials_df is not None
        and
        not trials_df.empty
    ):

        best_trial = (
            trials_df.loc[
                trials_df[
                    "response_score"
                ]
                .idxmax()
            ]
        )

        best_stimulus = (
            best_trial[
                "stimulus_type"
            ]
        )

        fingerprint[
            "best_stimulus"
        ] = best_stimulus

        # Frequency is not meaningful for broadband white noise.
        if (
            best_stimulus
            !=
            "White Noise"
        ):

            fingerprint[
                "best_frequency"
            ] = best_trial[
                "center_frequency_hz"
            ]

        # Bandwidth is only meaningful for narrow-band noise.
        if (
            best_stimulus
            ==
            "Narrow-Band Noise"
        ):

            bandwidth = (
                best_trial[
                    "bandwidth_hz"
                ]
            )

            if not pd.isna(
                bandwidth
            ):

                fingerprint[
                    "best_bandwidth"
                ] = bandwidth

        fingerprint[
            "best_level"
        ] = best_trial[
            "relative_level"
        ]

        fingerprint[
            "average_comfort"
        ] = float(
            trials_df[
                "comfort_score"
            ]
            .mean()
        )

        fingerprint[
            "average_noticeability"
        ] = float(
            trials_df[
                "tinnitus_noticeability"
            ]
            .mean()
        )

        fingerprint[
            "best_response_score"
        ] = float(
            best_trial[
                "response_score"
            ]
        )

    # --------------------------------------------------------
    # Latest clinician decision
    # --------------------------------------------------------

    if (
        decisions_df is not None
        and
        not decisions_df.empty
    ):

        latest_decision = (
            decisions_df.iloc[
                0
            ]
        )

        fingerprint[
            "latest_clinician_decision"
        ] = latest_decision[
            "decision"
        ]

        fingerprint[
            "latest_clinician_stimulus"
        ] = latest_decision[
            "final_stimulus_type"
        ]

        fingerprint[
            "latest_clinician_frequency"
        ] = latest_decision[
            "final_frequency_hz"
        ]

        fingerprint[
            "latest_clinician_bandwidth"
        ] = latest_decision[
            "final_bandwidth_hz"
        ]

        fingerprint[
            "latest_clinician_level"
        ] = latest_decision[
            "final_relative_level"
        ]

    return fingerprint


# ============================================================
# STYLE
# ============================================================


def apply_dashboard_style():

    st.html(
        """
        <style>

        .audiologist-header {
            background:
                linear-gradient(
                    120deg,
                    #12384f,
                    #176b79,
                    #2499a3
                );

            border-radius: 20px;
            padding: 32px 36px;
            margin-bottom: 26px;

            box-shadow:
                0 10px 28px
                rgba(20,70,90,0.14);
        }

        .audiologist-title {
            color: white;
            font-size: 34px;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .audiologist-subtitle {
            color: #def5f5;
            font-size: 16px;
        }

        .clinical-warning {
            background: #f4f8fa;
            border-left: 5px solid #27889a;
            border-radius: 8px;
            padding: 14px 18px;
            color: #365965;
            margin-top: 10px;
            margin-bottom: 18px;
        }

        </style>
        """
    )


# ============================================================
# MAIN DASHBOARD
# ============================================================


def show_audiologist_dashboard():

    apply_dashboard_style()

    st.html(
        """
        <div class="audiologist-header">

            <div class="audiologist-title">
                🩺 Audiologist Dashboard
            </div>

            <div class="audiologist-subtitle">
                Professional review, AI supervision and
                longitudinal tinnitus monitoring.
            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="clinical-warning">
            SmartTinnitus is a research and engineering prototype.
            AI recommendations and response scores are not
            validated clinical treatment decisions.
        </div>
        """
    )

    conn = get_connection()

    # ========================================================
    # SELECT PATIENT
    # ========================================================

    patients_df = (
        get_patients(
            conn
        )
    )

    if patients_df.empty:

        st.warning(
            "No patients are registered."
        )

        conn.close()

        return

    patient_codes = (
        patients_df[
            "patient_code"
        ]
        .tolist()
    )

    selected_patient_code = (
        st.selectbox(
            "Select patient record",
            patient_codes,
            key="audiologist_patient_selector"
        )
    )

    patient = (
        patients_df[
            patients_df[
                "patient_code"
            ]
            ==
            selected_patient_code
        ]
        .iloc[
            0
        ]
        .to_dict()
    )

    patient_id = int(
        patient[
            "id"
        ]
    )

    profile = (
        get_latest_profile(
            conn,
            patient_id
        )
    )

    trials_df = (
        get_trials(
            conn,
            patient_id
        )
    )

    sessions_df = (
        get_sessions(
            conn,
            patient_id
        )
    )

    audiogram_df = (
        get_audiogram(
            conn,
            patient_id
        )
    )

    decisions_df = (
        get_audiologist_decisions(
            conn,
            patient_id
        )
    )

    # ========================================================
    # 1. PATIENT OVERVIEW
    # ========================================================

    st.markdown(
        "## 1. Patient Overview"
    )

    c1, c2, c3, c4 = (
        st.columns(
            4
        )
    )

    c1.metric(
        "Patient ID",
        patient[
            "patient_code"
        ]
    )

    c2.metric(
        "Age",
        patient[
            "age"
        ]
    )

    c3.metric(
        "Sex",
        patient[
            "sex"
        ]
    )

    c4.metric(
        "Sessions",
        len(
            sessions_df
        )
    )

    if profile is None:

        st.warning(
            "No tinnitus profile available."
        )

    else:

        st.markdown(
            "#### Tinnitus Characteristics"
        )

        c1, c2, c3, c4 = (
            st.columns(
                4
            )
        )

        c1.metric(
            "Laterality",
            safe_value(
                profile.get(
                    "laterality"
                )
            )
        )

        c2.metric(
            "Type",
            safe_value(
                profile.get(
                    "tinnitus_type"
                )
            )
        )

        c3.metric(
            "Pattern",
            safe_value(
                profile.get(
                    "temporal_pattern"
                )
            )
        )

        c4.metric(
            "Duration",
            (
                f"{safe_integer(profile.get('duration_months'))} months"
            )
        )

        st.markdown(
            "#### Patient-Reported Impact"
        )

        c1, c2, c3, c4 = (
            st.columns(
                4
            )
        )

        c1.metric(
            "Annoyance",
            (
                f"{safe_integer(profile.get('annoyance_score'))}/10"
            )
        )

        c2.metric(
            "Sleep Disturbance",
            (
                f"{safe_integer(profile.get('sleep_score'))}/10"
            )
        )

        c3.metric(
            "Concentration",
            (
                f"{safe_integer(profile.get('concentration_score'))}/10"
            )
        )

        c4.metric(
            "Stress",
            (
                f"{safe_integer(profile.get('stress_score'))}/10"
            )
        )

    st.markdown("---")

    # ========================================================
    # 2. AUDIOLOGICAL ASSESSMENT
    # ========================================================

    st.markdown(
        "## 2. Audiological Assessment"
    )

    if profile is None:

        st.info(
            "Complete the patient's tinnitus profile first."
        )

    else:

        pitch = (
            profile.get(
                "estimated_pitch_hz"
            )
        )

        loudness = (
            profile.get(
                "loudness_relative"
            )
        )

        c1, c2, c3 = (
            st.columns(
                3
            )
        )

        c1.metric(
            "Estimated Tinnitus Pitch",
            (
                f"{safe_integer(pitch)} Hz"
                if pitch is not None
                else "Not measured"
            )
        )

        c2.metric(
            "Relative Loudness",
            safe_float_display(
                loudness
            )
        )

        c3.metric(
            "Audiogram",
            (
                "Available"
                if not audiogram_df.empty
                else
                "Not recorded"
            )
        )

        st.caption(
            "Relative loudness is uncalibrated digital amplitude, "
            "not dB SPL or dB HL."
        )

        if not audiogram_df.empty:

            right, left = (
                prepare_audiogram_values(
                    audiogram_df
                )
            )

            complete = (
                right is not None
                and
                left is not None
                and
                all(
                    x is not None
                    for x
                    in right
                )
                and
                all(
                    x is not None
                    for x
                    in left
                )
            )

            if complete:

                st.pyplot(
                    plot_audiogram(
                        right,
                        left,
                        tinnitus_pitch=pitch
                    )
                )

            with st.expander(
                "View audiogram values"
            ):

                st.dataframe(
                    audiogram_df,
                    use_container_width=True,
                    hide_index=True
                )

    st.markdown("---")

    # ========================================================
    # 3. ACOUSTIC FINGERPRINT
    # ========================================================

    st.markdown(
        "## 3. Acoustic Fingerprint"
    )

    st.caption(
        "Compact evolving summary of tinnitus characteristics, "
        "recorded acoustic responses and the latest clinician decision."
    )

    acoustic_fingerprint = (
        build_acoustic_fingerprint(
            profile,
            trials_df,
            decisions_df
        )
    )

    # --------------------------------------------------------
    # Tinnitus signature
    # --------------------------------------------------------

    st.markdown(
        "#### Tinnitus Signature"
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    fingerprint_pitch = (
        acoustic_fingerprint[
            "tinnitus_pitch"
        ]
    )

    c1.metric(
        "Estimated Pitch",
        (
            f"{safe_integer(fingerprint_pitch)} Hz"
            if fingerprint_pitch is not None
            else
            "Not measured"
        )
    )

    fingerprint_loudness = (
        acoustic_fingerprint[
            "relative_loudness"
        ]
    )

    c2.metric(
        "Relative Loudness",
        (
            safe_float_display(
                fingerprint_loudness,
                decimals=2
            )
            if fingerprint_loudness is not None
            else
            "Not measured"
        )
    )

    c3.metric(
        "Tinnitus Type",
        safe_value(
            acoustic_fingerprint[
                "tinnitus_type"
            ]
        )
    )

    c4.metric(
        "Laterality",
        safe_value(
            acoustic_fingerprint[
                "laterality"
            ]
        )
    )

    # --------------------------------------------------------
    # Acoustic response signature
    # --------------------------------------------------------

    st.markdown(
        "#### Acoustic Response Signature"
    )

    if trials_df.empty:

        st.info(
            "The acoustic-response part of the fingerprint will "
            "appear after stimulus trials are recorded."
        )

    else:

        c1, c2, c3, c4 = st.columns(
            4
        )

        c1.metric(
            "Best Stimulus",
            safe_value(
                acoustic_fingerprint[
                    "best_stimulus"
                ]
            )
        )

        fingerprint_frequency = (
            acoustic_fingerprint[
                "best_frequency"
            ]
        )

        c2.metric(
            "Best Frequency",
            (
                f"{safe_integer(fingerprint_frequency)} Hz"
                if fingerprint_frequency is not None
                else
                "N/A"
            )
        )

        fingerprint_bandwidth = (
            acoustic_fingerprint[
                "best_bandwidth"
            ]
        )

        c3.metric(
            "Best Bandwidth",
            (
                f"{safe_integer(fingerprint_bandwidth)} Hz"
                if fingerprint_bandwidth is not None
                else
                "N/A"
            )
        )

        fingerprint_level = (
            acoustic_fingerprint[
                "best_level"
            ]
        )

        c4.metric(
            "Best Relative Level",
            (
                safe_float_display(
                    fingerprint_level,
                    decimals=2
                )
                if fingerprint_level is not None
                else
                "N/A"
            )
        )

        c1, c2, c3 = st.columns(
            3
        )

        c1.metric(
            "Average Comfort",
            (
                f"{acoustic_fingerprint['average_comfort']:.1f} / 10"
            )
        )

        c2.metric(
            "Average Noticeability",
            (
                f"{acoustic_fingerprint['average_noticeability']:.1f} / 10"
            )
        )

        c3.metric(
            "Best Response Score",
            (
                f"{acoustic_fingerprint['best_response_score']:.1f} / 100"
            )
        )

    # --------------------------------------------------------
    # Latest clinician-reviewed configuration
    # --------------------------------------------------------

    st.markdown(
        "#### Latest Clinician-Reviewed Configuration"
    )

    latest_decision = (
        acoustic_fingerprint[
            "latest_clinician_decision"
        ]
    )

    if latest_decision is None:

        st.info(
            "No audiologist decision has been recorded yet."
        )

    else:

        c1, c2, c3, c4 = st.columns(
            4
        )

        c1.metric(
            "Decision",
            latest_decision
        )

        latest_stimulus = (
            acoustic_fingerprint[
                "latest_clinician_stimulus"
            ]
        )

        c2.metric(
            "Final Stimulus",
            (
                safe_value(
                    latest_stimulus
                )
                if latest_stimulus is not None
                else
                "No final stimulus"
            )
        )

        latest_frequency = (
            acoustic_fingerprint[
                "latest_clinician_frequency"
            ]
        )

        if (
            latest_stimulus
            ==
            "White Noise"
        ):

            latest_frequency_display = (
                "N/A"
            )

        elif latest_frequency is None or pd.isna(
            latest_frequency
        ):

            latest_frequency_display = (
                "N/A"
            )

        else:

            latest_frequency_display = (
                f"{safe_integer(latest_frequency)} Hz"
            )

        c3.metric(
            "Final Frequency",
            latest_frequency_display
        )

        latest_level = (
            acoustic_fingerprint[
                "latest_clinician_level"
            ]
        )

        if (
            latest_level is None
            or
            pd.isna(
                latest_level
            )
        ):

            latest_level_display = (
                "N/A"
            )

        else:

            latest_level_display = (
                safe_float_display(
                    latest_level,
                    decimals=2
                )
            )

        c4.metric(
            "Final Relative Level",
            latest_level_display
        )

        latest_bandwidth = (
            acoustic_fingerprint[
                "latest_clinician_bandwidth"
            ]
        )

        if (
            latest_stimulus
            ==
            "Narrow-Band Noise"
            and
            latest_bandwidth is not None
            and
            not pd.isna(
                latest_bandwidth
            )
        ):

            st.write(
                "**Final bandwidth:** "
                f"{safe_integer(latest_bandwidth)} Hz"
            )

    st.caption(
        "The Acoustic Fingerprint is a descriptive SmartTinnitus "
        "prototype summary. It is not a diagnostic biomarker or "
        "validated clinical classification."
    )

    st.markdown("---")

    # ========================================================
    # 4. ACOUSTIC RESPONSE
    # ========================================================

    st.markdown(
        "## 4. Acoustic Response Analysis"
    )

    if trials_df.empty:

        st.info(
            "No stimulus trials recorded."
        )

    else:

        best_trial = (
            trials_df.loc[
                trials_df[
                    "response_score"
                ]
                .idxmax()
            ]
        )

        c1, c2, c3, c4 = (
            st.columns(
                4
            )
        )

        c1.metric(
            "Trials",
            len(
                trials_df
            )
        )

        c2.metric(
            "Best Stimulus",
            best_trial[
                "stimulus_type"
            ]
        )

        c3.metric(
            "Best Frequency",
            (
                f"{int(best_trial['center_frequency_hz'])} Hz"
            )
        )

        c4.metric(
            "Best Response",
            (
                f"{best_trial['response_score']:.1f} / 100"
            )
        )

        st.caption(
            "Response Score = 40% comfort + 50% low tinnitus "
            "noticeability + 10% willingness to listen again."
        )

        display_trials = (
            trials_df[
                [
                    "trial_id",
                    "session_date",
                    "stimulus_type",
                    "center_frequency_hz",
                    "bandwidth_hz",
                    "relative_level",
                    "comfort_score",
                    "tinnitus_noticeability",
                    "listen_again",
                    "response_score"
                ]
            ]
            .copy()
        )

        display_trials.columns = [
            "Trial",
            "Session Date",
            "Stimulus",
            "Frequency (Hz)",
            "Bandwidth (Hz)",
            "Relative Level",
            "Comfort /10",
            "Noticeability /10",
            "Listen Again",
            "Response Score /100"
        ]

        display_trials[
            "Response Score /100"
        ] = (
            display_trials[
                "Response Score /100"
            ]
            .round(
                1
            )
        )

        st.dataframe(
            display_trials,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # ========================================================
    # 5. AI + AUDIOLOGIST SUPERVISION
    # ========================================================

    st.markdown(
        "## 5. AI Personalization & Audiologist Decision"
    )

    ml_summary = (
        get_ml_dataset_summary()
    )

    saved_metrics = (
        get_saved_model_metrics()
    )

    c1, c2, c3 = (
        st.columns(
            3
        )
    )

    c1.metric(
        "AI Model",
        (
            "Available"
            if model_exists()
            else
            "Not trained"
        )
    )

    c2.metric(
        "Dataset Trials",
        ml_summary.get(
            "rows",
            0
        )
    )

    c3.metric(
        "Dataset Patients",
        ml_summary.get(
            "patients",
            0
        )
    )

    if (
        ml_summary.get(
            "patients",
            0
        ) < 5
        or
        ml_summary.get(
            "rows",
            0
        ) < 30
    ):

        st.warning(
            "The current dataset is very small. AI output should "
            "be interpreted as proof-of-concept only."
        )

    if (
        profile is None
        or
        profile.get(
            "estimated_pitch_hz"
        ) is None
    ):

        st.info(
            "A saved tinnitus pitch is required."
        )

    elif not model_exists():

        st.warning(
            "Train the AI model first."
        )

    else:

        try:

            patient_features = (
                create_patient_features(
                    patient,
                    profile
                )
            )

            (
                recommendation_df,
                adaptive_metadata
            ) = (
                get_ai_recommendations(
                    patient_features,
                    profile[
                        "estimated_pitch_hz"
                    ],
                    trials_df=trials_df
                )
            )

            adaptive_summary = (
                get_adaptive_personalization_summary(
                    tinnitus_pitch=profile[
                        "estimated_pitch_hz"
                    ],
                    trials_df=trials_df
                )
            )

            st.markdown(
                "### Adaptive Search Strategy"
            )

            c1, c2, c3, c4 = st.columns(
                4
            )

            c1.metric(
                "Mode",
                adaptive_summary[
                    "mode"
                ]
            )

            c2.metric(
                "Patient Trials",
                adaptive_summary[
                    "patient_trial_count"
                ]
            )

            c3.metric(
                "Search Center",
                (
                    f"{int(adaptive_summary['search_center_hz'])} Hz"
                )
            )

            c4.metric(
                "Candidates",
                adaptive_summary[
                    "candidate_count"
                ]
            )

            st.caption(
                adaptive_summary[
                    "reason"
                ]
            )

            st.write(
                "**Frequency region:** "
                f"{int(adaptive_summary['frequency_min_hz'])}–"
                f"{int(adaptive_summary['frequency_max_hz'])} Hz"
            )

            if (
                adaptive_summary[
                    "preferred_stimulus_type"
                ]
                is not None
            ):

                st.write(
                    "**Best historical stimulus type by mean response:** "
                    f"{adaptive_summary['preferred_stimulus_type']}"
                )

            best_ai = (
                recommendation_df.iloc[
                    0
                ]
            )

            st.markdown(
                "### AI Recommendation"
            )

            c1, c2, c3, c4 = (
                st.columns(
                    4
                )
            )

            c1.metric(
                "Stimulus",
                best_ai[
                    "Stimulus"
                ]
            )

            if (
                best_ai[
                    "Stimulus"
                ]
                ==
                "White Noise"
            ):

                ai_frequency_display = (
                    "Broadband"
                )

            else:

                ai_frequency_display = (
                    f"{int(best_ai['_center_frequency_hz'])} Hz"
                )

            c2.metric(
                "Frequency",
                ai_frequency_display
            )

            ai_bandwidth = (
                best_ai[
                    "_bandwidth_hz"
                ]
            )

            if pd.isna(
                ai_bandwidth
            ):

                bandwidth_display = (
                    "N/A"
                )

            else:

                bandwidth_display = (
                    f"{int(ai_bandwidth)} Hz"
                )

            c3.metric(
                "Bandwidth",
                bandwidth_display
            )

            c4.metric(
                "Predicted Response",
                (
                    f"{best_ai['Predicted Score']:.1f} / 100"
                )
            )

            st.markdown(
                "#### Top 5 AI Candidates"
            )

            top_five = (
                recommendation_df.head(
                    5
                )
                .copy()
            )

            top_five = top_five.drop(
                columns=[
                    "_center_frequency_hz",
                    "_bandwidth_hz"
                ],
                errors="ignore"
            )

            top_five[
                "Predicted Score"
            ] = (
                top_five[
                    "Predicted Score"
                ]
                .round(
                    1
                )
            )

            st.dataframe(
                top_five,
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # AUDIOLOGIST DECISION
            # =================================================

            st.markdown("---")

            st.markdown(
                "### 🩺 Audiologist Decision"
            )

            st.write(
                "Review the AI suggestion and select the "
                "professional decision."
            )

            decision = (
                st.radio(
                    "Decision",
                    [
                        "Approve",
                        "Modify",
                        "Reject"
                    ],
                    horizontal=True,
                    key="audiologist_ai_decision"
                )
            )

            # =================================================
            # SESSION LINK
            # =================================================

            selected_session_id = None

            if not sessions_df.empty:

                session_options = {
                    (
                        f"{row['session_date']} "
                        f"• Session {int(row['session_id'])}"
                    ):
                    int(
                        row[
                            "session_id"
                        ]
                    )

                    for _, row
                    in sessions_df.iterrows()
                }

                session_label = (
                    st.selectbox(
                        "Associated session",
                        list(
                            session_options.keys()
                        ),
                        key="audiologist_decision_session"
                    )
                )

                selected_session_id = (
                    session_options[
                        session_label
                    ]
                )

            else:

                st.warning(
                    "No follow-up session exists yet. "
                    "The decision can still be stored without a session."
                )

            # =================================================
            # APPROVE
            # =================================================

            if decision == "Approve":

                final_stimulus = (
                    best_ai[
                        "Stimulus"
                    ]
                )

                final_frequency = float(
                    best_ai[
                        "_center_frequency_hz"
                    ]
                )

                if pd.isna(
                    best_ai[
                        "Bandwidth (Hz)"
                    ]
                ):

                    final_bandwidth = (
                        None
                    )

                else:

                    final_bandwidth = float(
                        best_ai[
                            "_bandwidth_hz"
                        ]
                    )

                final_level = float(
                    best_ai[
                        "Relative Level"
                    ]
                )

                st.success(
                    "The final configuration will match "
                    "the AI recommendation."
                )

            # =================================================
            # MODIFY
            # =================================================

            elif decision == "Modify":

                st.info(
                    "Adjust the stimulus before approving it "
                    "for testing."
                )

                final_stimulus = (
                    st.selectbox(
                        "Final Stimulus Type",
                        [
                            "Pure Tone",
                            "White Noise",
                            "Narrow-Band Noise"
                        ],
                        index=(
                            [
                                "Pure Tone",
                                "White Noise",
                                "Narrow-Band Noise"
                            ]
                            .index(
                                best_ai[
                                    "Stimulus"
                                ]
                            )
                        ),
                        key="modified_stimulus"
                    )
                )

                final_frequency = (
                    st.number_input(
                        "Final Frequency (Hz)",
                        min_value=250.0,
                        max_value=10000.0,
                        value=float(
                            best_ai[
                                "_center_frequency_hz"
                            ]
                        ),
                        step=50.0,
                        key="modified_frequency"
                    )
                )

                if (
                    final_stimulus
                    ==
                    "Narrow-Band Noise"
                ):

                    default_bandwidth = (
                        500.0
                    )

                    if not pd.isna(
                        best_ai[
                            "Bandwidth (Hz)"
                        ]
                    ):

                        default_bandwidth = float(
                            best_ai[
                                "_bandwidth_hz"
                            ]
                        )

                    final_bandwidth = (
                        st.number_input(
                            "Final Bandwidth (Hz)",
                            min_value=100.0,
                            max_value=2000.0,
                            value=default_bandwidth,
                            step=50.0,
                            key="modified_bandwidth"
                        )
                    )

                else:

                    final_bandwidth = (
                        None
                    )

                final_level = (
                    st.number_input(
                        "Final Relative Level",
                        min_value=0.01,
                        max_value=0.10,
                        value=float(
                            best_ai[
                                "Relative Level"
                            ]
                        ),
                        step=0.01,
                        format="%.2f",
                        key="modified_level"
                    )
                )

            # =================================================
            # REJECT
            # =================================================

            else:

                final_stimulus = (
                    None
                )

                final_frequency = (
                    None
                )

                final_bandwidth = (
                    None
                )

                final_level = (
                    None
                )

                st.warning(
                    "The AI recommendation will be recorded as rejected."
                )

            # =================================================
            # FINAL-CONFIGURATION AI PREDICTION
            # =================================================

            if (
                decision
                in
                [
                    "Approve",
                    "Modify"
                ]
            ):

                final_predicted_score = (
                    predict_final_configuration(
                        patient_features=patient_features,
                        stimulus_type=final_stimulus,
                        frequency_hz=final_frequency,
                        bandwidth_hz=final_bandwidth,
                        relative_level=final_level
                    )
                )

                st.markdown(
                    "#### Prediction for Final Tested Configuration"
                )

                p1, p2 = st.columns(
                    2
                )

                p1.metric(
                    "Original AI Proposal",
                    (
                        f"{float(best_ai['Predicted Score']):.1f} / 100"
                    )
                )

                p2.metric(
                    "Final Configuration Prediction",
                    (
                        f"{float(final_predicted_score):.1f} / 100"
                        if final_predicted_score is not None
                        else "N/A"
                    )
                )

                if (
                    decision
                    ==
                    "Modify"
                ):

                    st.caption(
                        "The final prediction is recalculated for the exact "
                        "stimulus configuration selected by the audiologist. "
                        "This is the prediction that should later be compared "
                        "with the patient's observed response."
                    )

            else:

                final_predicted_score = (
                    None
                )

            # =================================================
            # CLINICIAN NOTES
            # =================================================

            clinician_notes = (
                st.text_area(
                    "Audiologist Notes",
                    placeholder=(
                        "Example: Reduced relative level because "
                        "the patient reported sound sensitivity."
                    ),
                    key="audiologist_notes"
                )
            )

            if st.button(
                "💾 Save Audiologist Decision",
                key="save_audiologist_decision"
            ):

                try:

                    save_audiologist_decision(
                        conn=conn,
                        patient_id=patient_id,
                        session_id=selected_session_id,
                        ai_recommendation=best_ai,
                        decision=decision,
                        final_stimulus_type=final_stimulus,
                        final_frequency_hz=final_frequency,
                        final_bandwidth_hz=final_bandwidth,
                        final_relative_level=final_level,
                        final_predicted_score=final_predicted_score,
                        clinician_notes=clinician_notes
                    )

                    st.success(
                        "✅ Audiologist decision saved."
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Could not save decision: {error}"
                    )

        except Exception as error:

            st.error(
                f"AI recommendation failed: {error}"
            )

    # ========================================================
    # AUDIOLOGIST DECISION HISTORY
    # ========================================================

    st.markdown(
        "### Decision History"
    )

    st.caption(
        "Traceability fields identify the tinnitus profile and AI model "
        "associated with each clinician-reviewed recommendation."
    )

    if decisions_df.empty:

        st.info(
            "No audiologist decisions have been recorded yet."
        )

    else:

        display_decisions = (
            decisions_df[
                [
                    "decision_id",
                    "session_date",
                    "profile_id",
                    "model_version",
                    "model_trained_at",
                    "recommendation_created_at",

                    "ai_stimulus_type",
                    "ai_frequency_hz",
                    "ai_bandwidth_hz",
                    "ai_relative_level",
                    "ai_predicted_score",

                    "decision",

                    "final_stimulus_type",
                    "final_frequency_hz",
                    "final_bandwidth_hz",
                    "final_relative_level",
                    "final_predicted_score",

                    "clinician_notes",
                    "created_at"
                ]
            ]
            .copy()
        )

        display_decisions.columns = [
            "ID",
            "Session Date",
            "Profile ID",
            "Model Version",
            "Model Trained At",
            "Recommendation Time",

            "AI Stimulus",
            "AI Frequency",
            "AI Bandwidth",
            "AI Level",
            "AI Predicted Score",

            "Decision",

            "Final Stimulus",
            "Final Frequency",
            "Final Bandwidth",
            "Final Level",
            "Final Predicted Score",

            "Audiologist Notes",
            "Decision Time"
        ]

        # Convert prediction columns safely because older
        # audiologist decisions may contain NULL/None values.
        # pandas Series.round() on an object column containing
        # Python None can raise:
        # TypeError: type NoneType doesn't define __round__ method

        display_decisions[
            "AI Predicted Score"
        ] = (
            pd.to_numeric(
                display_decisions[
                    "AI Predicted Score"
                ],
                errors="coerce"
            )
            .round(
                1
            )
        )

        display_decisions[
            "Final Predicted Score"
        ] = (
            pd.to_numeric(
                display_decisions[
                    "Final Predicted Score"
                ],
                errors="coerce"
            )
            .round(
                1
            )
        )

        st.dataframe(
            display_decisions,
            use_container_width=True,
            hide_index=True
        )

    if saved_metrics:

        with st.expander(
            "View AI validation information"
        ):

            c1, c2, c3 = (
                st.columns(
                    3
                )
            )

            mae = (
                saved_metrics.get(
                    "mae"
                )
            )

            rmse = (
                saved_metrics.get(
                    "rmse"
                )
            )

            r2 = (
                saved_metrics.get(
                    "r2"
                )
            )

            c1.metric(
                "MAE",
                (
                    f"{mae:.2f}"
                    if mae is not None
                    else
                    "N/A"
                )
            )

            c2.metric(
                "RMSE",
                (
                    f"{rmse:.2f}"
                    if rmse is not None
                    else
                    "N/A"
                )
            )

            c3.metric(
                "R²",
                (
                    f"{r2:.3f}"
                    if r2 is not None
                    else
                    "N/A"
                )
            )

            st.caption(
                "Current validation remains unstable because "
                "the dataset is very small."
            )

    st.markdown("---")

    # ========================================================
    # 6. FOLLOW-UP
    # ========================================================

    st.markdown(
        "## 6. Follow-up & Longitudinal Monitoring"
    )

    if sessions_df.empty:

        st.info(
            "No follow-up sessions recorded."
        )

    else:

        display_sessions = (
            sessions_df[
                [
                    "session_id",
                    "session_date",
                    "annoyance_before",
                    "annoyance_after",
                    "tinnitus_noticeability_before",
                    "tinnitus_noticeability_after",
                    "notes"
                ]
            ]
            .copy()
        )

        display_sessions.columns = [
            "Session",
            "Date",
            "Annoyance Before",
            "Annoyance After",
            "Noticeability Before",
            "Noticeability After",
            "Notes"
        ]

        st.dataframe(
            display_sessions,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # SESSION NUMBER FOR CHARTS
        # ====================================================
        #
        # We use sequential session number on the x-axis instead
        # of session_date. This prevents several sessions recorded
        # on the same date from being drawn at the same x-position.
        # ====================================================

        chart_sessions = (
            sessions_df
            .copy()
            .reset_index(
                drop=True
            )
        )

        chart_sessions[
            "Session"
        ] = range(
            1,
            len(chart_sessions) + 1
        )

        # ====================================================
        # ANNOYANCE EVOLUTION
        # ====================================================

        st.markdown(
            "#### Annoyance Evolution"
        )

        annoyance_chart = (
            chart_sessions[
                [
                    "Session",
                    "annoyance_before",
                    "annoyance_after"
                ]
            ]
            .copy()
        )

        annoyance_chart.columns = [
            "Session",
            "Before",
            "After"
        ]

        st.line_chart(
            annoyance_chart.set_index(
                "Session"
            )
        )

        # ====================================================
        # NOTICEABILITY EVOLUTION
        # ====================================================

        st.markdown(
            "#### Noticeability Evolution"
        )

        noticeability_chart = (
            chart_sessions[
                [
                    "Session",
                    "tinnitus_noticeability_before",
                    "tinnitus_noticeability_after"
                ]
            ]
            .copy()
        )

        noticeability_chart.columns = [
            "Session",
            "Before",
            "After"
        ]

        st.line_chart(
            noticeability_chart.set_index(
                "Session"
            )
        )

        st.caption(
            "These are descriptive patient-reported changes "
            "and do not establish treatment efficacy."
        )

    st.markdown("---")

    # ========================================================
    # 7. PROFESSIONAL REPORT
    # ========================================================

    st.markdown(
        "## 7. Professional Report"
    )

    patient_info_report = {
        "patient_code":
            patient.get(
                "patient_code"
            ),

        "age":
            patient.get(
                "age"
            ),

        "sex":
            patient.get(
                "sex"
            )
    }

    profile_report = None

    if profile is not None:

        profile_report = {
            "laterality":
                profile.get(
                    "laterality"
                ),

            "tinnitus_type":
                profile.get(
                    "tinnitus_type"
                ),

            "temporal_pattern":
                profile.get(
                    "temporal_pattern"
                ),

            "duration_months":
                profile.get(
                    "duration_months"
                ),

            "annoyance_score":
                profile.get(
                    "annoyance_score"
                ),

            "sleep_score":
                profile.get(
                    "sleep_score"
                ),

            "concentration_score":
                profile.get(
                    "concentration_score"
                ),

            "stress_score":
                profile.get(
                    "stress_score"
                ),

            "estimated_pitch_hz":
                profile.get(
                    "estimated_pitch_hz"
                ),

            "loudness_relative":
                profile.get(
                    "loudness_relative"
                )
        }

    trials_report = []

    if not trials_df.empty:

        for _, row in trials_df.iterrows():

            bandwidth_value = (
                row[
                    "bandwidth_hz"
                ]
            )

            if pd.isna(
                bandwidth_value
            ):

                bandwidth_value = (
                    None
                )

            session_id_value = (
                row[
                    "session_id"
                ]
            )

            if pd.isna(
                session_id_value
            ):

                session_id_value = (
                    None
                )

            else:

                session_id_value = int(
                    session_id_value
                )

            trials_report.append(
                {
                    "session_id":
                        session_id_value,

                    "stimulus_type":
                        row[
                            "stimulus_type"
                        ],

                    "frequency":
                        row[
                            "center_frequency_hz"
                        ],

                    "bandwidth":
                        bandwidth_value,

                    "level":
                        row[
                            "relative_level"
                        ],

                    "comfort":
                        row[
                            "comfort_score"
                        ],

                    "noticeability":
                        row[
                            "tinnitus_noticeability"
                        ],

                    "listen_again":
                        row[
                            "listen_again"
                        ],

                    "decision_id":
                        (
                            int(
                                row[
                                    "decision_id"
                                ]
                            )
                            if
                            "decision_id"
                            in row.index
                            and
                            pd.notna(
                                row[
                                    "decision_id"
                                ]
                            )
                            else None
                        ),

                    "profile_id":
                        (
                            int(
                                row[
                                    "profile_id"
                                ]
                            )
                            if
                            "profile_id"
                            in row.index
                            and
                            pd.notna(
                                row[
                                    "profile_id"
                                ]
                            )
                            else None
                        ),

                    "source_type":
                        (
                            row[
                                "source_type"
                            ]
                            if
                            "source_type"
                            in row.index
                            else None
                        ),

                    "model_version":
                        (
                            row[
                                "model_version"
                            ]
                            if
                            "model_version"
                            in row.index
                            else None
                        ),

                    "model_trained_at":
                        (
                            row[
                                "model_trained_at"
                            ]
                            if
                            "model_trained_at"
                            in row.index
                            else None
                        ),

                    "model_predicted_score":
                        (
                            row[
                                "model_predicted_score"
                            ]
                            if
                            "model_predicted_score"
                            in row.index
                            else None
                        ),

                    "recommendation_created_at":
                        (
                            row[
                                "recommendation_created_at"
                            ]
                            if
                            "recommendation_created_at"
                            in row.index
                            else None
                        ),

                    "tested_at":
                        (
                            row[
                                "tested_at"
                            ]
                            if
                            "tested_at"
                            in row.index
                            else None
                        ),

                    "created_at":
                        row[
                            "created_at"
                        ]
                }
            )

    sessions_report = []

    if not sessions_df.empty:

        for _, row in sessions_df.iterrows():

            sessions_report.append(
                {
                    "session_id":
                        int(
                            row[
                                "session_id"
                            ]
                        ),

                    "date":
                        row[
                            "session_date"
                        ],

                    "annoyance_before":
                        row[
                            "annoyance_before"
                        ],

                    "annoyance_after":
                        row[
                            "annoyance_after"
                        ],

                    "noticeability_before":
                        row[
                            "tinnitus_noticeability_before"
                        ],

                    "noticeability_after":
                        row[
                            "tinnitus_noticeability_after"
                        ],

                    "notes":
                        row[
                            "notes"
                        ]
                }
            )

    audiogram_report = (
        None
    )

    if not audiogram_df.empty:

        report_right, report_left = (
            prepare_audiogram_values(
                audiogram_df
            )
        )

        complete_report = (
            report_right is not None
            and
            report_left is not None
            and
            all(
                x is not None
                for x
                in report_right
            )
            and
            all(
                x is not None
                for x
                in report_left
            )
        )

        if complete_report:

            audiogram_report = {
                "frequencies":
                    AUDIOGRAM_FREQUENCIES,

                "right_ear":
                    report_right,

                "left_ear":
                    report_left,

                "tinnitus_pitch":
                    (
                        profile.get(
                            "estimated_pitch_hz"
                        )
                        if profile
                        else
                        None
                    )
            }

    report_decisions = []

    if not decisions_df.empty:

        for _, row in decisions_df.iterrows():

            report_decisions.append(
                {
                    key:
                        (
                            None
                            if pd.isna(
                                value
                            )
                            else value
                        )
                    for key, value
                    in row.to_dict().items()
                }
            )

    report_fingerprint = (
        build_acoustic_fingerprint(
            profile,
            trials_df,
            decisions_df
        )
    )

    report_filename = (
        f"SmartTinnitus_Audiologist_Report_"
        f"{selected_patient_code}.pdf"
    )

    if st.button(
        "📄 Generate Professional PDF",
        key="audiologist_generate_pdf"
    ):

        try:

            generate_patient_report(
                report_filename,
                patient_info_report,
                profile_report,
                trials_report,
                sessions_report,
                audiogram_report,
                acoustic_fingerprint=report_fingerprint,
                audiologist_decisions=report_decisions
            )

            st.session_state[
                "audiologist_report_file"
            ] = report_filename

            st.session_state[
                "audiologist_report_patient"
            ] = selected_patient_code

            st.success(
                "Professional report generated."
            )

        except Exception as error:

            st.error(
                f"Report generation failed: {error}"
            )

    if (
        st.session_state.get(
            "audiologist_report_patient"
        )
        ==
        selected_patient_code
        and
        st.session_state.get(
            "audiologist_report_file"
        )
    ):

        saved_report_file = (
            st.session_state[
                "audiologist_report_file"
            ]
        )

        if os.path.exists(
            saved_report_file
        ):

            with open(
                saved_report_file,
                "rb"
            ) as pdf_file:

                st.download_button(
                    "⬇ Download Professional Report",
                    data=pdf_file.read(),
                    file_name=os.path.basename(
                        saved_report_file
                    ),
                    mime="application/pdf",
                    key="audiologist_download_pdf"
                )

    conn.close()