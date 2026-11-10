import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from backend.db import get_connection
from ml_model import (
    MODEL_FILE,
    load_model,
    model_exists,
    predict_responses_batch
)
from optimization import calculate_response_score


# ============================================================
# SMARTTINNITUS — PATIENT ACOUSTIC DIGITAL TWIN
# ============================================================
#
# The "Digital Twin" in SmartTinnitus is an evolving,
# patient-specific computational representation of the
# RECORDED and MODEL-PREDICTED acoustic response profile.
#
# It is NOT:
# - a biological digital twin of the auditory system
# - a diagnostic model
# - a validated treatment-response model
#
# It combines:
# - tinnitus profile
# - recorded stimulus-response history
# - adaptive-session results
# - current prototype ML predictions
#
# to visualize a patient-specific acoustic response landscape.
# ============================================================


MIN_FREQUENCY = 250
MAX_FREQUENCY = 10000

MAP_LEVELS = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05
]


MODEL_NUMERIC_FEATURES = [
    "age",
    "duration_months",
    "annoyance_score",
    "sleep_score",
    "concentration_score",
    "stress_score",
    "estimated_pitch_hz",
    "loudness_relative",
    "center_frequency_hz",
    "bandwidth_hz",
    "relative_level"
]

MODEL_CATEGORICAL_FEATURES = [
    "sex",
    "laterality",
    "tinnitus_type",
    "temporal_pattern",
    "stimulus_type"
]

MODEL_FEATURES = (
    MODEL_NUMERIC_FEATURES
    +
    MODEL_CATEGORICAL_FEATURES
)


# ============================================================
# MODEL CACHE
# ============================================================


@st.cache_resource
def _get_twin_model(
    model_modification_time
):
    return load_model()


# ============================================================
# DATA HELPERS
# ============================================================


def _load_patients():
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            id,
            patient_code,
            age,
            sex
        FROM patients
        ORDER BY patient_code ASC, id ASC
        """,
        conn
    )

    conn.close()

    return df


def _load_patient(
    patient_id
):
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            id,
            patient_code,
            age,
            sex
        FROM patients
        WHERE id = ?
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


def _load_latest_profile(
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


def _load_trials(
    patient_id
):
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            id,
            session_id,
            decision_id,
            profile_id,
            adaptive_session_id,
            source_type,
            model_version,
            model_predicted_score,
            tested_at,
            stimulus_type,
            center_frequency_hz,
            bandwidth_hz,
            relative_level,
            comfort_score,
            tinnitus_noticeability,
            listen_again,
            created_at
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


def _load_adaptive_sessions(
    patient_id
):
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            id,
            status,
            best_response_score,
            promising_frequency_min_hz,
            promising_frequency_max_hz,
            started_at,
            completed_at,
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
# PATIENT FEATURES
# ============================================================


def _create_patient_features(
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
# UNCERTAINTY / CONFIDENCE HELPERS
# ============================================================


def _build_raw_model_row(
    patient_features,
    candidate
):
    row = {
        **patient_features,

        "center_frequency_hz":
            float(
                candidate[
                    "center_frequency_hz"
                ]
            ),

        "bandwidth_hz":
            float(
                candidate.get(
                    "bandwidth_hz",
                    0
                )
                or
                0
            ),

        "relative_level":
            float(
                candidate[
                    "relative_level"
                ]
            ),

        "stimulus_type":
            candidate[
                "stimulus_type"
            ]
    }

    return pd.DataFrame(
        [
            row
        ],
        columns=MODEL_FEATURES
    )


def _tree_prediction_dispersion(
    model,
    patient_features,
    candidate
):
    """
    Estimate model disagreement across Random Forest trees.

    This is NOT a calibrated confidence interval.
    It is a prototype uncertainty signal only.
    """

    try:

        raw_row = _build_raw_model_row(
            patient_features,
            candidate
        )

        # Expected current deployment structure:
        # sklearn Pipeline(preprocessor -> RandomForestRegressor)
        if hasattr(
            model,
            "steps"
        ) and len(
            model.steps
        ) >= 2:

            preprocessor = model[
                :-1
            ]

            estimator = model.steps[
                -1
            ][
                1
            ]

            transformed = preprocessor.transform(
                raw_row
            )

        else:

            estimator = model
            transformed = raw_row

        trees = getattr(
            estimator,
            "estimators_",
            None
        )

        if not trees:

            return None

        tree_predictions = np.array(
            [
                float(
                    tree.predict(
                        transformed
                    )[
                        0
                    ]
                )
                for tree
                in trees
            ]
        )

        return {
            "tree_std":
                float(
                    np.std(
                        tree_predictions
                    )
                ),

            "tree_min":
                float(
                    np.min(
                        tree_predictions
                    )
                ),

            "tree_max":
                float(
                    np.max(
                        tree_predictions
                    )
                ),

            "tree_count":
                int(
                    len(
                        tree_predictions
                    )
                )
        }

    except Exception:

        return None


def _local_evidence_support(
    trials_df,
    candidate
):
    """
    Quantify how much REAL patient evidence exists near a candidate.

    The score is deliberately simple and explainable:
    - same stimulus type matters most
    - nearby frequency supports frequency-specific candidates
    - nearby digital level increases support

    Returns a 0-1 support index.
    """

    if (
        trials_df is None
        or
        trials_df.empty
    ):

        return {
            "support_index":
                0.0,

            "matching_trials":
                0,

            "nearby_trials":
                0,

            "nearest_distance":
                None
        }

    same_type = trials_df[
        trials_df[
            "stimulus_type"
        ]
        ==
        candidate[
            "stimulus_type"
        ]
    ].copy()

    same_type = same_type[
        same_type[
            "response_score"
        ].notna()
    ]

    if same_type.empty:

        return {
            "support_index":
                0.0,

            "matching_trials":
                0,

            "nearby_trials":
                0,

            "nearest_distance":
                None
        }

    distances = []

    nearby = 0

    for _, row in same_type.iterrows():

        level = row[
            "relative_level"
        ]

        if pd.isna(
            level
        ):

            continue

        level_distance = abs(
            float(
                level
            )
            -
            float(
                candidate[
                    "relative_level"
                ]
            )
        ) / 0.04

        if (
            candidate[
                "stimulus_type"
            ]
            ==
            "White Noise"
        ):

            frequency_distance = 0.0

        else:

            frequency = row[
                "center_frequency_hz"
            ]

            if pd.isna(
                frequency
            ):

                continue

            frequency_distance = abs(
                float(
                    frequency
                )
                -
                float(
                    candidate[
                        "center_frequency_hz"
                    ]
                )
            ) / 1500.0

        bandwidth_distance = 0.0

        if (
            candidate[
                "stimulus_type"
            ]
            ==
            "Narrow-Band Noise"
        ):

            observed_bw = row[
                "bandwidth_hz"
            ]

            if pd.notna(
                observed_bw
            ):

                bandwidth_distance = abs(
                    float(
                        observed_bw
                    )
                    -
                    float(
                        candidate.get(
                            "bandwidth_hz",
                            0
                        )
                    )
                ) / 1000.0

        combined_distance = float(
            np.sqrt(
                frequency_distance ** 2
                +
                level_distance ** 2
                +
                bandwidth_distance ** 2
            )
        )

        distances.append(
            combined_distance
        )

        if combined_distance <= 0.45:

            nearby += 1

    if not distances:

        return {
            "support_index":
                0.0,

            "matching_trials":
                int(
                    len(
                        same_type
                    )
                ),

            "nearby_trials":
                0,

            "nearest_distance":
                None
        }

    nearest = float(
        min(
            distances
        )
    )

    # Evidence quantity saturates gradually.
    quantity_component = min(
        1.0,
        len(
            same_type
        ) / 6.0
    )

    nearby_component = min(
        1.0,
        nearby / 3.0
    )

    proximity_component = max(
        0.0,
        1.0
        -
        min(
            1.0,
            nearest
        )
    )

    support = (
        0.30
        *
        quantity_component
        +
        0.40
        *
        nearby_component
        +
        0.30
        *
        proximity_component
    )

    return {
        "support_index":
            float(
                max(
                    0.0,
                    min(
                        1.0,
                        support
                    )
                )
            ),

        "matching_trials":
            int(
                len(
                    same_type
                )
            ),

        "nearby_trials":
            int(
                nearby
            ),

        "nearest_distance":
            nearest
    }


def _confidence_assessment(
    model,
    patient_features,
    trials_df,
    candidate
):
    """
    Prototype confidence indicator.

    Combines:
    1. patient-specific local evidence support
    2. Random Forest tree agreement when available

    This must NOT be interpreted as a calibrated probability.
    """

    evidence = _local_evidence_support(
        trials_df,
        candidate
    )

    dispersion = _tree_prediction_dispersion(
        model,
        patient_features,
        candidate
    )

    evidence_score = float(
        evidence[
            "support_index"
        ]
    )

    if dispersion is None:

        agreement_score = 0.50

    else:

        tree_std = float(
            dispersion[
                "tree_std"
            ]
        )

        # Rough engineering normalization for a 0-100 target.
        # 0 std -> 1.0 agreement
        # >= 20 score points -> approaches 0.
        agreement_score = max(
            0.0,
            min(
                1.0,
                1.0
                -
                tree_std
                /
                20.0
            )
        )

    confidence_index = (
        0.60
        *
        evidence_score
        +
        0.40
        *
        agreement_score
    )

    confidence_index = float(
        max(
            0.0,
            min(
                1.0,
                confidence_index
            )
        )
    )

    if confidence_index >= 0.70:

        label = "High"

    elif confidence_index >= 0.45:

        label = "Moderate"

    else:

        label = "Low"

    reasons = []

    if evidence[
        "matching_trials"
    ] == 0:

        reasons.append(
            "No recorded patient trials use this stimulus type."
        )

    elif evidence[
        "nearby_trials"
    ] == 0:

        reasons.append(
            "Patient-specific evidence exists, but not close to this exact configuration."
        )

    else:

        reasons.append(
            f"{evidence['nearby_trials']} nearby patient trial(s) support this region."
        )

    if dispersion is not None:

        if dispersion[
            "tree_std"
        ] <= 5:

            reasons.append(
                "Random Forest trees show relatively strong agreement."
            )

        elif dispersion[
            "tree_std"
        ] <= 12:

            reasons.append(
                "Random Forest trees show moderate disagreement."
            )

        else:

            reasons.append(
                "Random Forest trees disagree substantially in this region."
            )

    return {
        "confidence_index":
            confidence_index,

        "confidence_label":
            label,

        "uncertainty_index":
            1.0
            -
            confidence_index,

        "evidence":
            evidence,

        "dispersion":
            dispersion,

        "reason":
            " ".join(
                reasons
            )
    }


# ============================================================
# ACTIVE LEARNING
# ============================================================


def _candidate_already_tested(
    trials_df,
    candidate
):
    if (
        trials_df is None
        or
        trials_df.empty
    ):

        return False

    same = trials_df[
        trials_df[
            "stimulus_type"
        ]
        ==
        candidate[
            "stimulus_type"
        ]
    ].copy()

    if same.empty:

        return False

    for _, row in same.iterrows():

        if pd.isna(
            row[
                "relative_level"
            ]
        ):

            continue

        level_match = abs(
            float(
                row[
                    "relative_level"
                ]
            )
            -
            float(
                candidate[
                    "relative_level"
                ]
            )
        ) < 0.005

        if (
            candidate[
                "stimulus_type"
            ]
            ==
            "White Noise"
        ):

            frequency_match = True
            bandwidth_match = True

        else:

            if pd.isna(
                row[
                    "center_frequency_hz"
                ]
            ):

                continue

            frequency_match = abs(
                float(
                    row[
                        "center_frequency_hz"
                    ]
                )
                -
                float(
                    candidate[
                        "center_frequency_hz"
                    ]
                )
            ) <= 25

            if (
                candidate[
                    "stimulus_type"
                ]
                ==
                "Narrow-Band Noise"
            ):

                bandwidth_match = (
                    pd.notna(
                        row[
                            "bandwidth_hz"
                        ]
                    )
                    and
                    abs(
                        float(
                            row[
                                "bandwidth_hz"
                            ]
                        )
                        -
                        float(
                            candidate[
                                "bandwidth_hz"
                            ]
                        )
                    )
                    <= 25
                )

            else:

                bandwidth_match = True

        if (
            level_match
            and
            frequency_match
            and
            bandwidth_match
        ):

            return True

    return False


def _generate_active_learning_pool(
    lower_frequency,
    upper_frequency
):
    frequencies = np.linspace(
        lower_frequency,
        upper_frequency,
        13
    )

    frequencies = np.unique(
        [
            round(
                float(
                    value
                )
                /
                100
            )
            *
            100
            for value
            in frequencies
        ]
    )

    pool = []

    for stimulus_type in [
        "Pure Tone",
        "Narrow-Band Noise",
        "White Noise"
    ]:

        if stimulus_type == "White Noise":

            stimulus_frequencies = [
                float(
                    (
                        lower_frequency
                        +
                        upper_frequency
                    )
                    /
                    2
                )
            ]

        else:

            stimulus_frequencies = (
                frequencies
            )

        bandwidths = (
            [
                250,
                500,
                750,
                1000
            ]
            if stimulus_type
            ==
            "Narrow-Band Noise"
            else
            [
                0
            ]
        )

        for frequency in stimulus_frequencies:

            for bandwidth in bandwidths:

                for level in MAP_LEVELS:

                    pool.append(
                        {
                            "stimulus_type":
                                stimulus_type,

                            "center_frequency_hz":
                                float(
                                    frequency
                                ),

                            "bandwidth_hz":
                                float(
                                    bandwidth
                                ),

                            "relative_level":
                                float(
                                    level
                                )
                        }
                    )

    return pool


def _rank_active_learning_candidates(
    patient_features,
    trials_df,
    lower_frequency,
    upper_frequency
):
    model = _get_model()

    if model is None:

        return pd.DataFrame()

    candidates = _generate_active_learning_pool(
        lower_frequency,
        upper_frequency
    )

    predictions = predict_responses_batch(
        patient_features,
        candidates,
        model=model
    )

    rows = []

    for candidate, prediction in zip(
        candidates,
        predictions
    ):

        confidence = _confidence_assessment(
            model=model,
            patient_features=patient_features,
            trials_df=trials_df,
            candidate=candidate
        )

        predicted_score = float(
            prediction
        )

        uncertainty = float(
            confidence[
                "uncertainty_index"
            ]
        )

        # Exploitation term: higher predicted response is attractive.
        exploitation = max(
            0.0,
            min(
                1.0,
                predicted_score
                /
                100.0
            )
        )

        # Active-learning priority balances:
        # 60% promising predicted response
        # 40% uncertainty / information value
        priority = (
            0.60
            *
            exploitation
            +
            0.40
            *
            uncertainty
        )

        already_tested = _candidate_already_tested(
            trials_df,
            candidate
        )

        if already_tested:

            priority *= 0.35

        rows.append(
            {
                "Stimulus":
                    candidate[
                        "stimulus_type"
                    ],

                "Frequency (Hz)":
                    (
                        "Broadband"
                        if candidate[
                            "stimulus_type"
                        ]
                        ==
                        "White Noise"
                        else int(
                            round(
                                candidate[
                                    "center_frequency_hz"
                                ]
                            )
                        )
                    ),

                "Bandwidth (Hz)":
                    (
                        int(
                            candidate[
                                "bandwidth_hz"
                            ]
                        )
                        if candidate[
                            "stimulus_type"
                        ]
                        ==
                        "Narrow-Band Noise"
                        else "N/A"
                    ),

                "Relative Level":
                    candidate[
                        "relative_level"
                    ],

                "Predicted Response":
                    predicted_score,

                "Confidence":
                    confidence[
                        "confidence_label"
                    ],

                "Confidence Index":
                    confidence[
                        "confidence_index"
                    ]
                    *
                    100.0,

                "Uncertainty Index":
                    uncertainty
                    *
                    100.0,

                "Nearby Evidence":
                    confidence[
                        "evidence"
                    ][
                        "nearby_trials"
                    ],

                "Already Tested":
                    (
                        "Yes"
                        if already_tested
                        else "No"
                    ),

                "Active Learning Priority":
                    priority
                    *
                    100.0,

                "_candidate":
                    candidate,

                "_reason":
                    confidence[
                        "reason"
                    ]
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return result

    return (
        result
        .sort_values(
            [
                "Active Learning Priority",
                "Predicted Response"
            ],
            ascending=[
                False,
                False
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# HISTORICAL SUMMARY
# ============================================================


def _best_historical_trial(
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


def _preferred_stimulus(
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

    summary = (
        valid
        .groupby(
            "stimulus_type"
        )
        .agg(
            trial_count=(
                "response_score",
                "count"
            ),
            mean_response=(
                "response_score",
                "mean"
            )
        )
        .reset_index()
        .sort_values(
            "mean_response",
            ascending=False
        )
    )

    if summary.empty:
        return None

    return summary.iloc[
        0
    ].to_dict()


def _frequency_search_range(
    profile,
    trials_df,
    adaptive_df
):
    pitch = profile.get(
        "estimated_pitch_hz"
    )

    try:
        pitch = float(
            pitch
        )

    except Exception:
        pitch = 3000.0

    center = pitch
    half_span = 2000.0

    # Completed adaptive sessions can tighten the visual search
    # around the most recent promising region.
    if (
        adaptive_df is not None
        and
        not adaptive_df.empty
    ):

        completed = adaptive_df[
            adaptive_df[
                "status"
            ]
            ==
            "completed"
        ]

        if not completed.empty:

            latest = completed.iloc[
                0
            ]

            low = latest[
                "promising_frequency_min_hz"
            ]

            high = latest[
                "promising_frequency_max_hz"
            ]

            if (
                pd.notna(
                    low
                )
                and
                pd.notna(
                    high
                )
            ):

                low = float(
                    low
                )

                high = float(
                    high
                )

                center = (
                    low
                    +
                    high
                ) / 2.0

                half_span = max(
                    1000.0,
                    (
                        high
                        -
                        low
                    )
                    /
                    2.0
                    +
                    750.0
                )

    # Historical high-performing frequency-specific trials can
    # also influence the map center.
    if not trials_df.empty:

        valid = trials_df[
            trials_df[
                "stimulus_type"
            ].isin(
                [
                    "Pure Tone",
                    "Narrow-Band Noise"
                ]
            )
            &
            trials_df[
                "center_frequency_hz"
            ].notna()
            &
            trials_df[
                "response_score"
            ].notna()
        ].copy()

        if len(
            valid
        ) >= 3:

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

            historical_center = float(
                top[
                    "center_frequency_hz"
                ].median()
            )

            center = (
                center
                +
                historical_center
            ) / 2.0

    lower = max(
        MIN_FREQUENCY,
        center
        -
        half_span
    )

    upper = min(
        MAX_FREQUENCY,
        center
        +
        half_span
    )

    if (
        upper
        -
        lower
        <
        1000
    ):
        lower = max(
            MIN_FREQUENCY,
            center
            -
            500
        )

        upper = min(
            MAX_FREQUENCY,
            center
            +
            500
        )

    return (
        float(
            lower
        ),
        float(
            upper
        )
    )


# ============================================================
# DIGITAL TWIN PREDICTIONS
# ============================================================


def _get_model():
    if not model_exists():
        return None

    modification_time = (
        os.path.getmtime(
            MODEL_FILE
        )
    )

    return _get_twin_model(
        modification_time
    )


def _predict_candidates(
    patient_features,
    candidates
):
    model = _get_model()

    if model is None:
        return None

    predictions = predict_responses_batch(
        patient_features,
        candidates,
        model=model
    )

    return [
        float(
            value
        )
        for value
        in predictions
    ]


def _build_response_landscape(
    patient_features,
    lower_frequency,
    upper_frequency,
    stimulus_type,
    bandwidth_hz
):
    frequencies = np.linspace(
        lower_frequency,
        upper_frequency,
        25
    )

    frequencies = np.array(
        [
            round(
                float(
                    value
                )
                /
                50
            )
            *
            50
            for value
            in frequencies
        ]
    )

    frequencies = np.unique(
        frequencies
    )

    candidates = []

    for level in MAP_LEVELS:

        for frequency in frequencies:

            candidates.append(
                {
                    "stimulus_type":
                        stimulus_type,

                    "center_frequency_hz":
                        float(
                            frequency
                        ),

                    "bandwidth_hz":
                        (
                            float(
                                bandwidth_hz
                            )
                            if stimulus_type
                            ==
                            "Narrow-Band Noise"
                            else 0.0
                        ),

                    "relative_level":
                        float(
                            level
                        )
                }
            )

    predictions = _predict_candidates(
        patient_features,
        candidates
    )

    if predictions is None:
        return None

    matrix = np.array(
        predictions
    ).reshape(
        len(
            MAP_LEVELS
        ),
        len(
            frequencies
        )
    )

    best_index = np.unravel_index(
        np.argmax(
            matrix
        ),
        matrix.shape
    )

    best_level = MAP_LEVELS[
        best_index[
            0
        ]
    ]

    best_frequency = float(
        frequencies[
            best_index[
                1
            ]
        ]
    )

    best_prediction = float(
        matrix[
            best_index
        ]
    )

    return {
        "frequencies":
            frequencies,

        "levels":
            MAP_LEVELS,

        "matrix":
            matrix,

        "best_frequency":
            best_frequency,

        "best_level":
            float(
                best_level
            ),

        "best_prediction":
            best_prediction
    }


def _predict_single_configuration(
    patient_features,
    stimulus_type,
    frequency_hz,
    bandwidth_hz,
    relative_level
):
    candidate = {
        "stimulus_type":
            stimulus_type,

        "center_frequency_hz":
            float(
                frequency_hz
            ),

        "bandwidth_hz":
            (
                float(
                    bandwidth_hz
                )
                if stimulus_type
                ==
                "Narrow-Band Noise"
                else 0.0
            ),

        "relative_level":
            float(
                relative_level
            )
    }

    predictions = _predict_candidates(
        patient_features,
        [
            candidate
        ]
    )

    if not predictions:
        return None

    return float(
        predictions[
            0
        ]
    )


# ============================================================
# MAP
# ============================================================


def _plot_response_map(
    landscape,
    tinnitus_pitch,
    trials_df,
    stimulus_type,
    bandwidth_hz
):
    frequencies = landscape[
        "frequencies"
    ]

    levels = landscape[
        "levels"
    ]

    matrix = landscape[
        "matrix"
    ]

    fig, ax = plt.subplots(
        figsize=(
            10,
            5.2
        )
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=[
            float(
                frequencies[
                    0
                ]
            ),
            float(
                frequencies[
                    -1
                ]
            ),
            float(
                levels[
                    0
                ]
            ),
            float(
                levels[
                    -1
                ]
            )
        ]
    )

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "Predicted prototype response score /100"
    )

    try:
        tinnitus_pitch = float(
            tinnitus_pitch
        )

        ax.axvline(
            tinnitus_pitch,
            linestyle="--",
            linewidth=1.5,
            label="Estimated tinnitus pitch"
        )

    except Exception:
        pass

    ax.scatter(
        [
            landscape[
                "best_frequency"
            ]
        ],
        [
            landscape[
                "best_level"
            ]
        ],
        marker="*",
        s=180,
        label="Highest predicted point"
    )

    # Overlay observed trials of the same stimulus type.
    if not trials_df.empty:

        observed = trials_df[
            (
                trials_df[
                    "stimulus_type"
                ]
                ==
                stimulus_type
            )
            &
            trials_df[
                "center_frequency_hz"
            ].notna()
            &
            trials_df[
                "relative_level"
            ].notna()
            &
            trials_df[
                "response_score"
            ].notna()
        ].copy()

        if (
            stimulus_type
            ==
            "Narrow-Band Noise"
            and
            not observed.empty
        ):
            observed = observed[
                observed[
                    "bandwidth_hz"
                ].fillna(
                    0
                )
                .astype(float)
                ==
                float(
                    bandwidth_hz
                )
            ]

        if not observed.empty:

            ax.scatter(
                observed[
                    "center_frequency_hz"
                ].astype(float),
                observed[
                    "relative_level"
                ].astype(float),
                marker="o",
                facecolors="none",
                s=80,
                linewidths=1.5,
                label="Observed patient trials"
            )

    ax.set_xlabel(
        "Frequency (Hz)"
    )

    ax.set_ylabel(
        "Relative digital level"
    )

    ax.set_title(
        f"Predicted Acoustic Response Landscape — {stimulus_type}"
    )

    ax.legend(
        loc="best",
        fontsize=8
    )

    fig.tight_layout()

    return fig


# ============================================================
# UI
# ============================================================


def show_digital_twin_page(
    current_role,
    auth_patient_id=None
):
    st.title(
        "Patient Acoustic Digital Twin"
    )

    st.caption(
        "Explore the patient's evolving acoustic-response profile, "
        "recorded evidence, model predictions and uncertainty."
    )

    st.warning(
        "This Digital Twin represents a prototype acoustic-response model. "
        "It is not a biological model of the auditory system, does not diagnose "
        "tinnitus, and does not predict validated treatment effectiveness."
    )

    # --------------------------------------------------------
    # Resolve patient
    # --------------------------------------------------------

    if (
        current_role
        ==
        "patient"
    ):

        if auth_patient_id is None:

            st.error(
                "This patient login is not linked to a patient record."
            )

            return

        patient_id = int(
            auth_patient_id
        )

        patient = _load_patient(
            patient_id
        )

        if patient is None:

            st.error(
                "The linked patient record could not be found."
            )

            return

        st.caption(
            f"Logged in as Patient {patient['patient_code']}"
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
            for _, row
            in patients_df.iterrows()
        }

        selected = st.selectbox(
            "Patient",
            list(
                options.keys()
            ),
            key="digital_twin_patient"
        )

        patient_id = options[
            selected
        ]

        patient = _load_patient(
            patient_id
        )

    profile = _load_latest_profile(
        patient_id
    )

    if profile is None:

        st.info(
            "Complete the tinnitus assessment before generating "
            "the Acoustic Digital Twin."
        )

        return

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

        st.info(
            "An estimated tinnitus pitch is required before generating "
            "the response landscape."
        )

        return

    trials_df = _load_trials(
        patient_id
    )

    adaptive_df = _load_adaptive_sessions(
        patient_id
    )

    best_trial = _best_historical_trial(
        trials_df
    )

    preferred = _preferred_stimulus(
        trials_df
    )

    # --------------------------------------------------------
    # Twin status
    # --------------------------------------------------------

    st.markdown(
        "## Digital Twin Snapshot"
    )

    st.caption(
        "A concise view of the patient's current acoustic-response evidence."
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Patient",
        patient[
            "patient_code"
        ]
    )

    c2.metric(
        "Tinnitus Pitch",
        f"{float(tinnitus_pitch):.0f} Hz"
    )

    c3.metric(
        "Recorded Trials",
        len(
            trials_df
        )
    )

    completed_adaptive = (
        0
        if adaptive_df.empty
        else int(
            (
                adaptive_df[
                    "status"
                ]
                ==
                "completed"
            ).sum()
        )
    )

    c4.metric(
        "Completed Adaptive Sessions",
        completed_adaptive
    )

    c1, c2, c3 = st.columns(
        3
    )

    c1.metric(
        "Best Observed Response",
        (
            f"{float(best_trial['response_score']):.1f} / 100"
            if best_trial is not None
            else "No observations yet"
        )
    )

    c2.metric(
        "Best Historical Stimulus",
        (
            str(
                best_trial[
                    "stimulus_type"
                ]
            )
            if best_trial is not None
            else "Not available"
        )
    )

    c3.metric(
        "Preferred Stimulus by Mean Response",
        (
            str(
                preferred[
                    "stimulus_type"
                ]
            )
            if preferred is not None
            else "Not available"
        )
    )

    if not model_exists():

        st.error(
            "The current AI model is not available. The observed patient history "
            "can still be reviewed, but the predicted response landscape cannot "
            "be generated until the audiologist trains the model."
        )

        if not trials_df.empty:

            st.dataframe(
                trials_df,
                use_container_width=True,
                hide_index=True
            )

        return

    patient_features = _create_patient_features(
        patient,
        profile
    )

    lower_frequency, upper_frequency = (
        _frequency_search_range(
            profile,
            trials_df,
            adaptive_df
        )
    )

    st.markdown("---")

    view = st.radio(
        "Digital Twin View",
        [
            "Acoustic Response Map",
            "Configuration Simulator",
            "Confidence & Active Learning",
            "Observed Evidence",
            "Twin Evolution"
        ],
        horizontal=True,
        key="digital_twin_view"
    )

    # ========================================================
    # RESPONSE MAP
    # ========================================================

    if (
        view
        ==
        "Acoustic Response Map"
    ):

        st.markdown(
            "## Interactive Acoustic Response Map"
        )

        st.caption(
            "Use this map to inspect where the current prototype model predicts "
            "higher or lower response scores across acoustic settings."
        )

        st.write(
            "The map simulates the current model's predicted prototype "
            "response score across frequency and relative digital level."
        )

        c1, c2 = st.columns(
            2
        )

        stimulus_type = c1.selectbox(
            "Stimulus type",
            [
                "Pure Tone",
                "Narrow-Band Noise"
            ],
            key="twin_map_stimulus"
        )

        if (
            stimulus_type
            ==
            "Narrow-Band Noise"
        ):

            bandwidth_hz = c2.selectbox(
                "Bandwidth",
                [
                    250,
                    500,
                    750,
                    1000
                ],
                index=1,
                key="twin_map_bandwidth"
            )

        else:

            bandwidth_hz = 0

            c2.metric(
                "Bandwidth",
                "N/A"
            )

        landscape = _build_response_landscape(
            patient_features=patient_features,
            lower_frequency=lower_frequency,
            upper_frequency=upper_frequency,
            stimulus_type=stimulus_type,
            bandwidth_hz=bandwidth_hz
        )

        if landscape is None:

            st.error(
                "The response map could not be generated."
            )

            return

        best_map_candidate = {
            "stimulus_type":
                stimulus_type,

            "center_frequency_hz":
                landscape[
                    "best_frequency"
                ],

            "bandwidth_hz":
                (
                    float(
                        bandwidth_hz
                    )
                    if stimulus_type
                    ==
                    "Narrow-Band Noise"
                    else 0.0
                ),

            "relative_level":
                landscape[
                    "best_level"
                ]
        }

        map_confidence = _confidence_assessment(
            model=_get_model(),
            patient_features=patient_features,
            trials_df=trials_df,
            candidate=best_map_candidate
        )

        m1, m2, m3, m4 = st.columns(
            4
        )

        m1.metric(
            "Highest Predicted Point",
            f"{landscape['best_prediction']:.1f} / 100"
        )

        m2.metric(
            "Predicted Frequency",
            f"{landscape['best_frequency']:.0f} Hz"
        )

        m3.metric(
            "Predicted Relative Level",
            f"{landscape['best_level']:.2f}"
        )

        m4.metric(
            "Prototype Confidence",
            (
                f"{map_confidence['confidence_label']} "
                f"({map_confidence['confidence_index'] * 100:.0f}%)"
            )
        )

        st.caption(
            map_confidence[
                "reason"
            ]
        )

        figure = _plot_response_map(
            landscape=landscape,
            tinnitus_pitch=tinnitus_pitch,
            trials_df=trials_df,
            stimulus_type=stimulus_type,
            bandwidth_hz=bandwidth_hz
        )

        st.pyplot(
            figure,
            clear_figure=True,
            use_container_width=True
        )

        st.caption(
            "Star = highest point predicted by the current prototype model. "
            "Dashed line = estimated tinnitus pitch. Hollow circles = recorded "
            "patient trials at matching stimulus settings."
        )

        st.info(
            "A high predicted score identifies a configuration that the current "
            "prototype model ranks as promising to test. It does not establish "
            "clinical benefit."
        )

    # ========================================================
    # SIMULATOR
    # ========================================================

    elif (
        view
        ==
        "Configuration Simulator"
    ):

        st.markdown(
            "## What-If Configuration Simulator"
        )

        st.caption(
            "Simulate one exact acoustic configuration without recording it as an observed trial."
        )

        st.write(
            "Change the acoustic parameters to see how the current patient-specific "
            "model scores that exact candidate configuration."
        )

        stimulus_type = st.selectbox(
            "Stimulus",
            [
                "Pure Tone",
                "Narrow-Band Noise",
                "White Noise"
            ],
            key="twin_sim_stimulus"
        )

        c1, c2, c3 = st.columns(
            3
        )

        if (
            stimulus_type
            ==
            "White Noise"
        ):

            frequency_hz = float(
                tinnitus_pitch
            )

            c1.metric(
                "Frequency",
                "Broadband"
            )

        else:

            frequency_hz = c1.number_input(
                "Frequency (Hz)",
                min_value=250.0,
                max_value=10000.0,
                value=float(
                    round(
                        float(
                            tinnitus_pitch
                        )
                    )
                ),
                step=50.0,
                key="twin_sim_frequency"
            )

        if (
            stimulus_type
            ==
            "Narrow-Band Noise"
        ):

            bandwidth_hz = c2.selectbox(
                "Bandwidth (Hz)",
                [
                    250,
                    500,
                    750,
                    1000
                ],
                index=1,
                key="twin_sim_bandwidth"
            )

        else:

            bandwidth_hz = 0

            c2.metric(
                "Bandwidth",
                "N/A"
            )

        relative_level = c3.number_input(
            "Relative Level",
            min_value=0.01,
            max_value=0.05,
            value=0.03,
            step=0.01,
            format="%.2f",
            key="twin_sim_level"
        )

        prediction = _predict_single_configuration(
            patient_features=patient_features,
            stimulus_type=stimulus_type,
            frequency_hz=frequency_hz,
            bandwidth_hz=bandwidth_hz,
            relative_level=relative_level
        )

        simulator_candidate = {
            "stimulus_type":
                stimulus_type,

            "center_frequency_hz":
                float(
                    frequency_hz
                ),

            "bandwidth_hz":
                (
                    float(
                        bandwidth_hz
                    )
                    if stimulus_type
                    ==
                    "Narrow-Band Noise"
                    else 0.0
                ),

            "relative_level":
                float(
                    relative_level
                )
        }

        simulator_confidence = _confidence_assessment(
            model=_get_model(),
            patient_features=patient_features,
            trials_df=trials_df,
            candidate=simulator_candidate
        )

        st.markdown(
            "### Digital Twin Prediction"
        )

        if prediction is None:

            st.error(
                "Prediction could not be generated."
            )

        else:

            c1, c2, c3 = st.columns(
                3
            )

            c1.metric(
                "Predicted Prototype Response",
                f"{prediction:.1f} / 100"
            )

            c2.metric(
                "Prototype Confidence",
                (
                    f"{simulator_confidence['confidence_label']} "
                    f"({simulator_confidence['confidence_index'] * 100:.0f}%)"
                )
            )

            if best_trial is not None:

                difference = (
                    float(
                        prediction
                    )
                    -
                    float(
                        best_trial[
                            "response_score"
                        ]
                    )
                )

                c3.metric(
                    "vs. Best Observed Response",
                    f"{difference:+.1f}"
                )

            else:

                c3.metric(
                    "Observed Comparison",
                    "No prior trials"
                )

            st.caption(
                simulator_confidence[
                    "reason"
                ]
            )

            if simulator_confidence[
                "dispersion"
            ] is not None:

                st.write(
                    "**Random Forest disagreement:** "
                    f"tree prediction SD = "
                    f"{simulator_confidence['dispersion']['tree_std']:.1f} "
                    "score points."
                )

            st.caption(
                "Confidence is a prototype engineering indicator based on local "
                "patient evidence and model agreement. It is not a calibrated "
                "probability or clinical confidence interval."
            )

    # ========================================================
    # CONFIDENCE + ACTIVE LEARNING
    # ========================================================

    elif (
        view
        ==
        "Confidence & Active Learning"
    ):

        st.markdown(
            "## Confidence-Aware Active Learning"
        )

        st.caption(
            "Identify promising but insufficiently explored configurations that could "
            "add useful information to the patient-specific model."
        )

        st.write(
            "SmartTinnitus identifies configurations that are both promising "
            "and informative to test. This balances exploitation of high "
            "predicted response regions with exploration of uncertain regions."
        )

        st.info(
            "Active-learning priority is NOT a treatment recommendation. "
            "It is a prototype research score used to decide which acoustic "
            "configuration could be most informative to test next."
        )

        ranked = _rank_active_learning_candidates(
            patient_features=patient_features,
            trials_df=trials_df,
            lower_frequency=lower_frequency,
            upper_frequency=upper_frequency
        )

        if ranked.empty:

            st.error(
                "Active-learning candidates could not be generated."
            )

        else:

            best_active = ranked.iloc[
                0
            ]

            candidate = best_active[
                "_candidate"
            ]

            c1, c2, c3, c4 = st.columns(
                4
            )

            c1.metric(
                "Suggested Informative Test",
                best_active[
                    "Stimulus"
                ]
            )

            c2.metric(
                "Predicted Response",
                f"{float(best_active['Predicted Response']):.1f} / 100"
            )

            c3.metric(
                "Prototype Confidence",
                (
                    f"{best_active['Confidence']} "
                    f"({float(best_active['Confidence Index']):.0f}%)"
                )
            )

            c4.metric(
                "Learning Priority",
                f"{float(best_active['Active Learning Priority']):.0f} / 100"
            )

            detail1, detail2, detail3 = st.columns(
                3
            )

            detail1.metric(
                "Frequency",
                (
                    "Broadband"
                    if best_active[
                        "Stimulus"
                    ]
                    ==
                    "White Noise"
                    else f"{best_active['Frequency (Hz)']} Hz"
                )
            )

            detail2.metric(
                "Bandwidth",
                (
                    f"{best_active['Bandwidth (Hz)']} Hz"
                    if best_active[
                        "Stimulus"
                    ]
                    ==
                    "Narrow-Band Noise"
                    else "N/A"
                )
            )

            detail3.metric(
                "Relative Level",
                f"{float(best_active['Relative Level']):.2f}"
            )

            st.caption(
                best_active[
                    "_reason"
                ]
            )

            st.markdown(
                "### Why this candidate?"
            )

            st.write(
                "The priority combines **60% predicted response potential** "
                "and **40% uncertainty / information value**. Previously tested "
                "exact configurations are down-weighted so the system favors "
                "useful new information."
            )

            display_columns = [
                "Stimulus",
                "Frequency (Hz)",
                "Bandwidth (Hz)",
                "Relative Level",
                "Predicted Response",
                "Confidence",
                "Confidence Index",
                "Uncertainty Index",
                "Nearby Evidence",
                "Already Tested",
                "Active Learning Priority"
            ]

            top_candidates = ranked[
                display_columns
            ].head(
                10
            ).copy()

            for column in [
                "Predicted Response",
                "Confidence Index",
                "Uncertainty Index",
                "Active Learning Priority"
            ]:

                top_candidates[
                    column
                ] = pd.to_numeric(
                    top_candidates[
                        column
                    ],
                    errors="coerce"
                ).round(
                    1
                )

            st.markdown(
                "### Top Informative Candidates"
            )

            st.dataframe(
                top_candidates,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # Uncertainty map for the chosen stimulus family
            # ------------------------------------------------

            st.markdown(
                "### Uncertainty Landscape"
            )

            selected_type = st.selectbox(
                "Stimulus family for uncertainty map",
                [
                    "Pure Tone",
                    "Narrow-Band Noise"
                ],
                key="twin_uncertainty_type"
            )

            if (
                selected_type
                ==
                "Narrow-Band Noise"
            ):

                selected_bw = st.selectbox(
                    "Bandwidth for uncertainty map",
                    [
                        250,
                        500,
                        750,
                        1000
                    ],
                    index=1,
                    key="twin_uncertainty_bw"
                )

            else:

                selected_bw = 0

            frequencies = np.linspace(
                lower_frequency,
                upper_frequency,
                20
            )

            frequencies = np.unique(
                [
                    round(
                        float(
                            value
                        )
                        /
                        50
                    )
                    *
                    50
                    for value
                    in frequencies
                ]
            )

            uncertainty_matrix = np.zeros(
                (
                    len(
                        MAP_LEVELS
                    ),
                    len(
                        frequencies
                    )
                )
            )

            model = _get_model()

            for level_index, level in enumerate(
                MAP_LEVELS
            ):

                for frequency_index, frequency in enumerate(
                    frequencies
                ):

                    map_candidate = {
                        "stimulus_type":
                            selected_type,

                        "center_frequency_hz":
                            float(
                                frequency
                            ),

                        "bandwidth_hz":
                            (
                                float(
                                    selected_bw
                                )
                                if selected_type
                                ==
                                "Narrow-Band Noise"
                                else 0.0
                            ),

                        "relative_level":
                            float(
                                level
                            )
                    }

                    confidence = _confidence_assessment(
                        model=model,
                        patient_features=patient_features,
                        trials_df=trials_df,
                        candidate=map_candidate
                    )

                    uncertainty_matrix[
                        level_index,
                        frequency_index
                    ] = (
                        confidence[
                            "uncertainty_index"
                        ]
                        *
                        100.0
                    )

            fig, ax = plt.subplots(
                figsize=(
                    10,
                    5
                )
            )

            image = ax.imshow(
                uncertainty_matrix,
                aspect="auto",
                origin="lower",
                extent=[
                    float(
                        frequencies[
                            0
                        ]
                    ),
                    float(
                        frequencies[
                            -1
                        ]
                    ),
                    float(
                        MAP_LEVELS[
                            0
                        ]
                    ),
                    float(
                        MAP_LEVELS[
                            -1
                        ]
                    )
                ]
            )

            colorbar = fig.colorbar(
                image,
                ax=ax
            )

            colorbar.set_label(
                "Prototype uncertainty index /100"
            )

            ax.axvline(
                float(
                    tinnitus_pitch
                ),
                linestyle="--",
                linewidth=1.5,
                label="Estimated tinnitus pitch"
            )

            ax.set_xlabel(
                "Frequency (Hz)"
            )

            ax.set_ylabel(
                "Relative digital level"
            )

            ax.set_title(
                f"Uncertainty Landscape — {selected_type}"
            )

            ax.legend(
                fontsize=8
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True,
                use_container_width=True
            )

            st.caption(
                "Higher uncertainty means the current prototype has less "
                "support or greater model disagreement in that region. "
                "It is not a clinical risk map."
            )


    # ========================================================
    # OBSERVED EVIDENCE
    # ========================================================

    elif (
        view
        ==
        "Observed Evidence"
    ):

        st.markdown(
            "## Recorded Patient Evidence"
        )

        if trials_df.empty:

            st.info(
                "No stimulus-response trials have been recorded yet."
            )

        else:

            display = trials_df[
                [
                    "id",
                    "source_type",
                    "stimulus_type",
                    "center_frequency_hz",
                    "bandwidth_hz",
                    "relative_level",
                    "comfort_score",
                    "tinnitus_noticeability",
                    "response_score",
                    "tested_at"
                ]
            ].copy()

            display.columns = [
                "Trial ID",
                "Source",
                "Stimulus",
                "Frequency (Hz)",
                "Bandwidth (Hz)",
                "Level",
                "Comfort",
                "Noticeability",
                "Observed Response /100",
                "Tested"
            ]

            display[
                "Observed Response /100"
            ] = pd.to_numeric(
                display[
                    "Observed Response /100"
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

            summary = (
                trials_df[
                    trials_df[
                        "response_score"
                    ].notna()
                ]
                .groupby(
                    "stimulus_type"
                )
                .agg(
                    Trials=(
                        "response_score",
                        "count"
                    ),
                    Mean_Response=(
                        "response_score",
                        "mean"
                    ),
                    Best_Response=(
                        "response_score",
                        "max"
                    )
                )
                .reset_index()
            )

            if not summary.empty:

                summary[
                    "Mean_Response"
                ] = summary[
                    "Mean_Response"
                ].round(
                    1
                )

                summary[
                    "Best_Response"
                ] = summary[
                    "Best_Response"
                ].round(
                    1
                )

                summary.columns = [
                    "Stimulus",
                    "Trials",
                    "Mean Response /100",
                    "Best Response /100"
                ]

                st.markdown(
                    "### Evidence Summary"
                )

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True
                )

    # ========================================================
    # EVOLUTION
    # ========================================================

    else:

        st.markdown(
            "## Digital Twin Evolution"
        )

        st.write(
            "Adaptive sessions progressively add patient-specific observations "
            "that refine the response history used by SmartTinnitus."
        )

        if adaptive_df.empty:

            st.info(
                "No adaptive sessions have been recorded yet."
            )

        else:

            display = adaptive_df.copy()

            display.columns = [
                "Adaptive Session",
                "Status",
                "Best Response /100",
                "Promising Min Hz",
                "Promising Max Hz",
                "Started",
                "Completed",
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

            completed = adaptive_df[
                adaptive_df[
                    "status"
                ]
                ==
                "completed"
            ].copy()

            if not completed.empty:

                chronological = completed.sort_values(
                    "id"
                )

                fig, ax = plt.subplots(
                    figsize=(
                        9,
                        4
                    )
                )

                ax.plot(
                    chronological[
                        "id"
                    ],
                    chronological[
                        "best_response_score"
                    ],
                    marker="o"
                )

                ax.set_xlabel(
                    "Adaptive Session ID"
                )

                ax.set_ylabel(
                    "Best observed response /100"
                )

                ax.set_title(
                    "Digital Twin Evidence Growth Across Adaptive Sessions"
                )

                ax.grid(
                    True,
                    alpha=0.25
                )

                fig.tight_layout()

                st.pyplot(
                    fig,
                    clear_figure=True,
                    use_container_width=True
                )

        st.caption(
            "The Digital Twin becomes more informative as additional real patient "
            "responses are recorded. Historical evidence and model predictions "
            "must remain clearly distinguished."
        )
