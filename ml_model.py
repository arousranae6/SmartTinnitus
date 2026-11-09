import os
import json
from datetime import datetime

import joblib
import pandas as pd

from database import get_connection

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =====================================================
# FILES
# =====================================================

MODEL_FILE = (
    "smarttinnitus_response_model.joblib"
)

METRICS_FILE = (
    "smarttinnitus_model_metrics.json"
)

DATASET_FILE = (
    "smarttinnitus_ml_dataset.csv"
)


# =====================================================
# BUILD DATASET
# =====================================================

def build_ml_dataset():

    conn = get_connection()

    query = """
    SELECT
        patients.id AS patient_id,
        patients.age,
        patients.sex,

        tinnitus_profiles.laterality,
        tinnitus_profiles.tinnitus_type,
        tinnitus_profiles.temporal_pattern,
        tinnitus_profiles.duration_months,
        tinnitus_profiles.annoyance_score,
        tinnitus_profiles.sleep_score,
        tinnitus_profiles.concentration_score,
        tinnitus_profiles.stress_score,
        tinnitus_profiles.estimated_pitch_hz,
        tinnitus_profiles.loudness_relative,

        sessions.id AS session_id,
        sessions.session_date,
        sessions.annoyance_before,
        sessions.annoyance_after,
        sessions.tinnitus_noticeability_before,
        sessions.tinnitus_noticeability_after,

        stimulus_trials.stimulus_type,
        stimulus_trials.center_frequency_hz,
        stimulus_trials.bandwidth_hz,
        stimulus_trials.relative_level,
        stimulus_trials.comfort_score,
        stimulus_trials.tinnitus_noticeability,
        stimulus_trials.listen_again

    FROM stimulus_trials

    JOIN patients
        ON stimulus_trials.patient_id =
           patients.id

    LEFT JOIN sessions
        ON stimulus_trials.session_id =
           sessions.id

    LEFT JOIN tinnitus_profiles
        ON tinnitus_profiles.id = (
            SELECT tp2.id
            FROM tinnitus_profiles tp2
            WHERE tp2.patient_id =
                  patients.id
            ORDER BY tp2.id DESC
            LIMIT 1
        )

    ORDER BY
        patients.id,
        sessions.session_date,
        stimulus_trials.id
    """

    df = pd.read_sql_query(
        query,
        conn
    )

    conn.close()

    if df.empty:

        return df

    # =================================================
    # LISTEN AGAIN
    # =================================================

    df[
        "listen_again_numeric"
    ] = (
        df[
            "listen_again"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "yes": 1,
                "no": 0
            }
        )
        .fillna(0)
        .astype(int)
    )

    # =================================================
    # NEW RESPONSE SCORE: 0-100
    # =================================================
    #
    # 40% Comfort
    # 50% Low tinnitus noticeability
    # 10% Listen again
    #
    # Comfort:
    # comfort / 10 * 40
    #
    # Noticeability:
    # (10 - noticeability) / 10 * 50
    #
    # Listen again:
    # Yes = 10
    # No  = 0
    #
    # This is NOT a validated clinical score.
    # =================================================

    comfort_component = (
        (
            df[
                "comfort_score"
            ]
            /
            10
        )
        *
        40
    )

    noticeability_component = (
        (
            (
                10
                -
                df[
                    "tinnitus_noticeability"
                ]
            )
            /
            10
        )
        *
        50
    )

    listen_again_component = (
        df[
            "listen_again_numeric"
        ]
        *
        10
    )

    df[
        "response_score"
    ] = (
        comfort_component
        +
        noticeability_component
        +
        listen_again_component
    )

    # =================================================
    # SESSION CHANGES
    # =================================================

    df[
        "annoyance_change"
    ] = (
        df[
            "annoyance_before"
        ]
        -
        df[
            "annoyance_after"
        ]
    )

    df[
        "noticeability_change"
    ] = (
        df[
            "tinnitus_noticeability_before"
        ]
        -
        df[
            "tinnitus_noticeability_after"
        ]
    )

    # =================================================
    # SESSION STATUS
    # =================================================

    df[
        "has_session"
    ] = (
        df[
            "session_id"
        ]
        .notna()
        .astype(int)
    )

    # =================================================
    # BANDWIDTH
    # =================================================

    df[
        "bandwidth_hz"
    ] = (
        df[
            "bandwidth_hz"
        ]
        .fillna(0)
    )

    return df


# =====================================================
# SAVE DATASET
# =====================================================

def save_ml_dataset(
    filename=DATASET_FILE
):

    df = (
        build_ml_dataset()
    )

    df.to_csv(
        filename,
        index=False
    )

    return filename


# =====================================================
# DATASET SUMMARY
# =====================================================

def get_ml_dataset_summary():

    df = (
        build_ml_dataset()
    )

    if df.empty:

        return {
            "rows": 0,
            "patients": 0,
            "sessions": 0,
            "session_linked_trials": 0,
            "legacy_trials": 0
        }

    total_rows = (
        len(df)
    )

    total_patients = (
        df[
            "patient_id"
        ]
        .nunique()
    )

    total_sessions = (
        df[
            "session_id"
        ]
        .dropna()
        .nunique()
    )

    linked = int(
        df[
            "has_session"
        ]
        .sum()
    )

    legacy = (
        total_rows
        -
        linked
    )

    return {
        "rows":
            int(
                total_rows
            ),

        "patients":
            int(
                total_patients
            ),

        "sessions":
            int(
                total_sessions
            ),

        "session_linked_trials":
            int(
                linked
            ),

        "legacy_trials":
            int(
                legacy
            )
    }


# =====================================================
# MODEL FEATURES
# =====================================================

def get_model_features():

    numeric_features = [
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

    categorical_features = [
        "sex",
        "laterality",
        "tinnitus_type",
        "temporal_pattern",
        "stimulus_type"
    ]

    return (
        numeric_features,
        categorical_features
    )


# =====================================================
# MODEL PIPELINE
# =====================================================

def create_model_pipeline():

    (
        numeric_features,
        categorical_features
    ) = get_model_features()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=1
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    return pipeline


# =====================================================
# PREPARE TRAINING DATA
# =====================================================

def prepare_training_data():

    df = (
        build_ml_dataset()
    )

    if df.empty:

        raise ValueError(
            "The ML dataset is empty."
        )

    (
        numeric_features,
        categorical_features
    ) = get_model_features()

    feature_columns = (
        numeric_features
        +
        categorical_features
    )

    X = (
        df[
            feature_columns
        ]
        .copy()
    )

    y = (
        df[
            "response_score"
        ]
        .copy()
    )

    groups = (
        df[
            "patient_id"
        ]
        .copy()
    )

    return (
        X,
        y,
        groups,
        df
    )


# =====================================================
# SAVE METRICS
# =====================================================

def save_model_metrics(
    results
):

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )


# =====================================================
# LOAD METRICS
# =====================================================

def get_saved_model_metrics():

    if not os.path.exists(
        METRICS_FILE
    ):

        return None

    try:

        with open(
            METRICS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

    except Exception:

        return None


# =====================================================
# MODEL EXISTS
# =====================================================

def model_exists():

    return os.path.exists(
        MODEL_FILE
    )


# =====================================================
# PATIENT-WISE VALIDATION
# =====================================================

def evaluate_model_patient_wise(
    X,
    y,
    groups
):

    if (
        groups.nunique()
        <
        3
    ):

        raise ValueError(
            "At least 3 different patients are required "
            "for patient-wise validation."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.33,
        random_state=42
    )

    (
        train_index,
        test_index
    ) = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X.iloc[
        train_index
    ]

    X_test = X.iloc[
        test_index
    ]

    y_train = y.iloc[
        train_index
    ]

    y_test = y.iloc[
        test_index
    ]

    train_groups = groups.iloc[
        train_index
    ]

    test_groups = groups.iloc[
        test_index
    ]

    validation_model = (
        create_model_pipeline()
    )

    validation_model.fit(
        X_train,
        y_train
    )

    predictions = (
        validation_model.predict(
            X_test
        )
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = (
        mse
        **
        0.5
    )

    if len(
        y_test
    ) >= 2:

        try:

            r2 = r2_score(
                y_test,
                predictions
            )

        except Exception:

            r2 = None

    else:

        r2 = None

    return {
        "validation_method":
            "Patient-wise GroupShuffleSplit",

        "training_samples":
            int(
                len(
                    X_train
                )
            ),

        "testing_samples":
            int(
                len(
                    X_test
                )
            ),

        "training_patients":
            int(
                train_groups
                .nunique()
            ),

        "testing_patients":
            int(
                test_groups
                .nunique()
            ),

        "training_patient_ids":
            sorted(
                train_groups
                .unique()
                .tolist()
            ),

        "testing_patient_ids":
            sorted(
                test_groups
                .unique()
                .tolist()
            ),

        "mae":
            float(
                mae
            ),

        "rmse":
            float(
                rmse
            ),

        "r2":
            (
                float(
                    r2
                )
                if r2 is not None
                else None
            )
    }


# =====================================================
# DEPLOYMENT MODEL
# =====================================================

def train_deployment_model(
    X,
    y
):

    deployment_model = (
        create_model_pipeline()
    )

    deployment_model.fit(
        X,
        y
    )

    return deployment_model


# =====================================================
# TRAIN MODEL
# =====================================================

def train_model(
    save_model=True
):

    (
        X,
        y,
        groups,
        df
    ) = prepare_training_data()

    if len(
        df
    ) < 10:

        raise ValueError(
            "At least 10 stimulus trials are required "
            "to train the prototype model."
        )

    # =================================================
    # STEP 1:
    # PATIENT-WISE VALIDATION
    # =================================================

    validation_results = (
        evaluate_model_patient_wise(
            X,
            y,
            groups
        )
    )

    # =================================================
    # STEP 2:
    # FINAL DEPLOYMENT MODEL USING ALL DATA
    # =================================================

    deployment_model = (
        train_deployment_model(
            X,
            y
        )
    )

    summary = (
        get_ml_dataset_summary()
    )

    results = {
        "trained_at":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "model_type":
            "RandomForestRegressor",

        "response_score_scale":
            "0-100",

        "response_score_definition":
            (
                "40% comfort + "
                "50% low tinnitus noticeability + "
                "10% willingness to listen again"
            ),

        "deployment_training_method":
            "All available trials",

        "deployment_samples":
            int(
                len(
                    df
                )
            ),

        "deployment_patients":
            int(
                groups.nunique()
            ),

        "total_samples":
            int(
                len(
                    df
                )
            ),

        "patients":
            int(
                summary[
                    "patients"
                ]
            ),

        "sessions":
            int(
                summary[
                    "sessions"
                ]
            ),

        "session_linked_trials":
            int(
                summary[
                    "session_linked_trials"
                ]
            ),

        "legacy_trials":
            int(
                summary[
                    "legacy_trials"
                ]
            ),

        "validation_method":
            validation_results[
                "validation_method"
            ],

        "training_samples":
            validation_results[
                "training_samples"
            ],

        "testing_samples":
            validation_results[
                "testing_samples"
            ],

        "training_patients":
            validation_results[
                "training_patients"
            ],

        "testing_patients":
            validation_results[
                "testing_patients"
            ],

        "training_patient_ids":
            validation_results[
                "training_patient_ids"
            ],

        "testing_patient_ids":
            validation_results[
                "testing_patient_ids"
            ],

        "mae":
            validation_results[
                "mae"
            ],

        "rmse":
            validation_results[
                "rmse"
            ],

        "r2":
            validation_results[
                "r2"
            ]
    }

    if save_model:

        joblib.dump(
            deployment_model,
            MODEL_FILE
        )

        save_model_metrics(
            results
        )

        save_ml_dataset(
            DATASET_FILE
        )

    return (
        deployment_model,
        results
    )


# =====================================================
# LOAD MODEL
# =====================================================

def load_model():

    if not model_exists():

        raise FileNotFoundError(
            "No trained SmartTinnitus model was found."
        )

    return joblib.load(
        MODEL_FILE
    )


# =====================================================
# CREATE PREDICTION ROW
# =====================================================

def create_prediction_row(
    patient_data,
    stimulus_data
):

    return {
        "age":
            patient_data.get(
                "age"
            ),

        "duration_months":
            patient_data.get(
                "duration_months"
            ),

        "annoyance_score":
            patient_data.get(
                "annoyance_score"
            ),

        "sleep_score":
            patient_data.get(
                "sleep_score"
            ),

        "concentration_score":
            patient_data.get(
                "concentration_score"
            ),

        "stress_score":
            patient_data.get(
                "stress_score"
            ),

        "estimated_pitch_hz":
            patient_data.get(
                "estimated_pitch_hz"
            ),

        "loudness_relative":
            patient_data.get(
                "loudness_relative"
            ),

        "center_frequency_hz":
            stimulus_data.get(
                "center_frequency_hz"
            ),

        "bandwidth_hz":
            stimulus_data.get(
                "bandwidth_hz",
                0
            ),

        "relative_level":
            stimulus_data.get(
                "relative_level"
            ),

        "sex":
            patient_data.get(
                "sex"
            ),

        "laterality":
            patient_data.get(
                "laterality"
            ),

        "tinnitus_type":
            patient_data.get(
                "tinnitus_type"
            ),

        "temporal_pattern":
            patient_data.get(
                "temporal_pattern"
            ),

        "stimulus_type":
            stimulus_data.get(
                "stimulus_type"
            )
    }


# =====================================================
# SINGLE PREDICTION
# =====================================================

def predict_response(
    patient_data,
    stimulus_data,
    model=None
):

    if model is None:

        model = (
            load_model()
        )

    input_data = pd.DataFrame(
        [
            create_prediction_row(
                patient_data,
                stimulus_data
            )
        ]
    )

    prediction = (
        model.predict(
            input_data
        )[0]
    )

    return float(
        prediction
    )


# =====================================================
# BATCH PREDICTION
# =====================================================

def predict_responses_batch(
    patient_data,
    stimulus_candidates,
    model=None
):

    if model is None:

        model = (
            load_model()
        )

    if not stimulus_candidates:

        return []

    rows = []

    for stimulus_data in stimulus_candidates:

        rows.append(
            create_prediction_row(
                patient_data,
                stimulus_data
            )
        )

    input_data = (
        pd.DataFrame(
            rows
        )
    )

    predictions = (
        model.predict(
            input_data
        )
    )

    return [
        float(
            value
        )
        for value in predictions
    ]


# =====================================================
# RUN DIRECTLY
# =====================================================

if __name__ == "__main__":

    print()
    print(
        "========================================"
    )

    print(
        "SmartTinnitus ML Training"
    )

    print(
        "========================================"
    )

    print()

    print(
        "Response score:"
    )

    print(
        "40% Comfort"
    )

    print(
        "50% Low tinnitus noticeability"
    )

    print(
        "10% Willingness to listen again"
    )

    print(
        "Scale: 0 - 100"
    )

    print()

    summary = (
        get_ml_dataset_summary()
    )

    print(
        "Trials:",
        summary[
            "rows"
        ]
    )

    print(
        "Patients:",
        summary[
            "patients"
        ]
    )

    print(
        "Sessions:",
        summary[
            "sessions"
        ]
    )

    print(
        "Session-linked trials:",
        summary[
            "session_linked_trials"
        ]
    )

    print(
        "Legacy trials:",
        summary[
            "legacy_trials"
        ]
    )

    print()

    try:

        model, results = (
            train_model(
                save_model=True
            )
        )

        print(
            "STEP 1: Patient-wise validation completed."
        )

        print()

        print(
            "Training patients:",
            results[
                "training_patient_ids"
            ]
        )

        print(
            "Testing patients:",
            results[
                "testing_patient_ids"
            ]
        )

        print(
            "Training trials:",
            results[
                "training_samples"
            ]
        )

        print(
            "Testing trials:",
            results[
                "testing_samples"
            ]
        )

        print()

        print(
            "MAE:",
            round(
                results[
                    "mae"
                ],
                3
            )
        )

        print(
            "RMSE:",
            round(
                results[
                    "rmse"
                ],
                3
            )
        )

        if (
            results[
                "r2"
            ]
            is not None
        ):

            print(
                "R²:",
                round(
                    results[
                        "r2"
                    ],
                    3
                )
            )

        else:

            print(
                "R²: unavailable"
            )

        print()

        print(
            "STEP 2: Deployment model trained."
        )

        print(
            "Deployment samples:",
            results[
                "deployment_samples"
            ]
        )

        print(
            "Deployment patients:",
            results[
                "deployment_patients"
            ]
        )

        print()

        print(
            "Saved deployment model:",
            MODEL_FILE
        )

        print(
            "Saved metrics:",
            METRICS_FILE
        )

        print(
            "Saved dataset:",
            DATASET_FILE
        )

        print()

        print(
            "IMPORTANT:"
        )

        print(
            "The 0-100 response score is a prototype "
            "engineering score, not a validated clinical scale."
        )

    except Exception as error:

        print(
            "Training error:"
        )

        print(
            error
        )