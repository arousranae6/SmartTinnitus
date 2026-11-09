import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from database import get_connection
from optimization import calculate_response_score
from ml_model import (
    get_ml_dataset_summary,
    get_saved_model_metrics,
    model_exists
)


# ============================================================
# SMARTTINNITUS — ML EVALUATION & VALIDATION
# ============================================================
#
# This page documents and visualizes the evaluation state of
# the current prototype model.
#
# Important methodological distinction:
# 1. Patient-wise validation model:
#    evaluates generalization to held-out patients.
# 2. Deployment model:
#    retrained on all available trials after validation.
#
# These must not be confused.
# ============================================================


MODEL_FILE = "smarttinnitus_response_model.joblib"
METRICS_FILE = "smarttinnitus_model_metrics.json"


# ============================================================
# DATA
# ============================================================


def _load_trial_evaluation_data():
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            st.id,
            st.patient_id,
            st.session_id,
            st.stimulus_type,
            st.center_frequency_hz,
            st.bandwidth_hz,
            st.relative_level,
            st.comfort_score,
            st.tinnitus_noticeability,
            st.listen_again,
            st.source_type,
            st.model_predicted_score,
            st.tested_at,
            p.patient_code
        FROM stimulus_trials st
        LEFT JOIN patients p
            ON st.patient_id = p.id
        ORDER BY st.id ASC
        """,
        conn
    )

    conn.close()

    if df.empty:
        return df

    scores = []

    for _, row in df.iterrows():

        try:
            score = calculate_response_score(
                row["comfort_score"],
                row["tinnitus_noticeability"],
                row["listen_again"]
            )

        except Exception:
            score = np.nan

        scores.append(score)

    df["response_score"] = pd.to_numeric(
        scores,
        errors="coerce"
    )

    df["model_predicted_score"] = pd.to_numeric(
        df["model_predicted_score"],
        errors="coerce"
    )

    return df


def _safe_metrics():
    try:
        metrics = get_saved_model_metrics()

        if isinstance(metrics, dict):
            return metrics

    except Exception:
        pass

    if os.path.exists(METRICS_FILE):

        try:
            with open(
                METRICS_FILE,
                "r",
                encoding="utf-8"
            ) as file:
                metrics = json.load(file)

            if isinstance(metrics, dict):
                return metrics

        except Exception:
            pass

    return {}


# ============================================================
# MODEL / DATASET STATUS
# ============================================================


def _dataset_stage(
    rows,
    patients
):
    if (
        patients < 5
        or
        rows < 30
    ):
        return (
            "Very small prototype dataset",
            "Reliable generalization cannot yet be established."
        )

    if (
        patients < 10
        or
        rows < 100
    ):
        return (
            "Early experimental dataset",
            "Validation is possible, but performance estimates remain highly unstable."
        )

    return (
        "Preliminary evaluation dataset",
        "The dataset is larger, but external validation would still be required before clinical claims."
    )


def _metric_interpretation(
    metrics,
    rows,
    patients
):
    if not metrics:
        return (
            "No saved validation result",
            "Train the model from the Audiologist AI Personalization page before interpreting performance."
        )

    if (
        patients < 5
        or
        rows < 30
    ):
        return (
            "Validation available, but unstable",
            "The held-out patient sample is too small for a reliable performance estimate."
        )

    return (
        "Preliminary validation available",
        "Metrics should still be treated as experimental because they come from a limited research dataset."
    )


# ============================================================
# DESCRIPTIVE ERROR ANALYSIS
# ============================================================


def _linked_prediction_errors(
    df
):
    if df.empty:
        return pd.DataFrame()

    linked = df[
        df["model_predicted_score"].notna()
        &
        df["response_score"].notna()
    ].copy()

    if linked.empty:
        return linked

    linked["prediction_error"] = (
        linked["response_score"]
        -
        linked["model_predicted_score"]
    )

    linked["absolute_error"] = (
        linked["prediction_error"].abs()
    )

    return linked


# ============================================================
# CHARTS
# ============================================================


def _plot_trials_per_patient(
    df
):
    counts = (
        df
        .groupby("patient_code")
        .size()
        .sort_values(
            ascending=False
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    ax.bar(
        counts.index.astype(str),
        counts.values
    )

    ax.set_xlabel(
        "Patient"
    )

    ax.set_ylabel(
        "Recorded trials"
    )

    ax.set_title(
        "Trial Distribution Across Patients"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    fig.tight_layout()

    return fig


def _plot_stimulus_distribution(
    df
):
    counts = (
        df["stimulus_type"]
        .fillna("Unknown")
        .value_counts()
    )

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.bar(
        counts.index.astype(str),
        counts.values
    )

    ax.set_xlabel(
        "Stimulus type"
    )

    ax.set_ylabel(
        "Recorded trials"
    )

    ax.set_title(
        "Stimulus-Type Distribution"
    )

    ax.tick_params(
        axis="x",
        rotation=20
    )

    fig.tight_layout()

    return fig


def _plot_observed_vs_predicted(
    linked
):
    fig, ax = plt.subplots(
        figsize=(6.5, 6)
    )

    ax.scatter(
        linked["model_predicted_score"],
        linked["response_score"]
    )

    minimum = min(
        float(
            linked["model_predicted_score"].min()
        ),
        float(
            linked["response_score"].min()
        )
    )

    maximum = max(
        float(
            linked["model_predicted_score"].max()
        ),
        float(
            linked["response_score"].max()
        )
    )

    ax.plot(
        [
            minimum,
            maximum
        ],
        [
            minimum,
            maximum
        ],
        linestyle="--",
        linewidth=1.5
    )

    ax.set_xlabel(
        "Stored model prediction /100"
    )

    ax.set_ylabel(
        "Observed response /100"
    )

    ax.set_title(
        "Stored Predictions vs Observed Responses"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    fig.tight_layout()

    return fig


def _plot_error_distribution(
    linked
):
    fig, ax = plt.subplots(
        figsize=(8.5, 4.5)
    )

    ax.hist(
        linked["prediction_error"],
        bins=min(
            10,
            max(
                3,
                len(linked)
            )
        )
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.5
    )

    ax.set_xlabel(
        "Observed − predicted score"
    )

    ax.set_ylabel(
        "Trials"
    )

    ax.set_title(
        "Prediction Error Distribution"
    )

    fig.tight_layout()

    return fig


# ============================================================
# UI
# ============================================================


def show_ml_evaluation_page(
    current_role
):
    st.title(
        "ML Evaluation & Validation"
    )

    st.caption(
        "Transparent evaluation of dataset composition, patient-wise validation, "
        "prediction error and current model readiness."
    )

    if current_role != "audiologist":

        st.error(
            "ML evaluation is available only to the audiologist role."
        )

        return

    st.warning(
        "The current machine-learning component is a research prototype. "
        "Validation metrics estimate prediction error on held-out patient data; "
        "they do not establish clinical effectiveness or diagnostic accuracy."
    )

    dataset_summary = get_ml_dataset_summary()
    metrics = _safe_metrics()
    trial_df = _load_trial_evaluation_data()

    rows = int(
        dataset_summary.get(
            "rows",
            len(
                trial_df
            )
        )
    )

    patients = int(
        dataset_summary.get(
            "patients",
            trial_df["patient_id"].nunique()
            if not trial_df.empty
            else 0
        )
    )

    sessions = int(
        dataset_summary.get(
            "sessions",
            trial_df["session_id"].nunique()
            if (
                not trial_df.empty
                and
                "session_id"
                in trial_df.columns
            )
            else 0
        )
    )

    # --------------------------------------------------------
    # Evaluation state
    # --------------------------------------------------------

    st.markdown(
        "## Evaluation Status"
    )

    st.caption(
        "Current database size, model availability and evaluation maturity."
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Trials",
        rows
    )

    c2.metric(
        "Patients",
        patients
    )

    c3.metric(
        "Sessions",
        sessions
    )

    c4.metric(
        "Model Artifact",
        (
            "Available"
            if model_exists()
            else "Not available"
        )
    )

    stage, stage_explanation = _dataset_stage(
        rows,
        patients
    )

    evaluation_status, evaluation_explanation = (
        _metric_interpretation(
            metrics,
            rows,
            patients
        )
    )

    c1, c2 = st.columns(
        2
    )

    c1.info(
        f"**Dataset stage:** {stage}\n\n{stage_explanation}"
    )

    c2.info(
        f"**Evaluation status:** {evaluation_status}\n\n{evaluation_explanation}"
    )

    # --------------------------------------------------------
    # Validation design
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "## Validation Design"
    )

    st.markdown(
        """
**Target:** SmartTinnitus prototype response score (0–100).

**Validation split:** patient-wise. Trials from a patient assigned to the test set are not also used to train the validation model.

**Why this matters:** a random trial-level split could place trials from the same patient in both training and testing, producing an overly optimistic estimate of generalization.

**Deployment:** after validation, a separate deployment model is trained using all currently available trials. The deployment model is therefore not the model used to calculate the held-out validation metrics.
"""
    )

    st.info(
        "Patient-wise separation is the correct direction for this project because the intended question is whether the model can generalize to patient data that were not used during validation training."
    )

    # --------------------------------------------------------
    # Saved metrics
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "## Saved Patient-Wise Validation"
    )

    st.caption(
        "Held-out patient performance saved from the latest validation run."
    )

    if not metrics:

        st.warning(
            "No saved model-validation metrics were found."
        )

    else:

        c1, c2, c3 = st.columns(
            3
        )

        c1.metric(
            "Training Patients",
            metrics.get(
                "training_patients",
                "N/A"
            )
        )

        c2.metric(
            "Testing Patients",
            metrics.get(
                "testing_patients",
                "N/A"
            )
        )

        c3.metric(
            "Validation Method",
            "Patient-wise"
        )

        c1, c2 = st.columns(
            2
        )

        c1.metric(
            "Training Trials",
            metrics.get(
                "training_samples",
                "N/A"
            )
        )

        c2.metric(
            "Testing Trials",
            metrics.get(
                "testing_samples",
                "N/A"
            )
        )

        c1, c2, c3 = st.columns(
            3
        )

        mae = metrics.get(
            "mae"
        )

        rmse = metrics.get(
            "rmse"
        )

        r2 = metrics.get(
            "r2"
        )

        c1.metric(
            "MAE ↓",
            (
                f"{float(mae):.2f}"
                if mae is not None
                else "N/A"
            )
        )

        c2.metric(
            "RMSE ↓",
            (
                f"{float(rmse):.2f}"
                if rmse is not None
                else "N/A"
            )
        )

        c3.metric(
            "R² ↑",
            (
                f"{float(r2):.3f}"
                if r2 is not None
                else "N/A"
            )
        )

        st.markdown(
            """
**How to read these metrics**

- **MAE:** average absolute difference between predicted and observed prototype response scores.
- **RMSE:** similar to MAE but penalizes larger errors more strongly.
- **R²:** describes how much variation in held-out scores is explained relative to a mean-prediction baseline. With very small test sets, R² can be highly unstable and can be negative.
"""
        )

        st.warning(
            "Do not select or report only the most favorable split. With the current small patient count, a single GroupShuffleSplit can change substantially depending on which patients are held out."
        )

    # --------------------------------------------------------
    # Dataset composition
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "## Dataset Composition"
    )

    st.caption(
        "Inspect patient imbalance and stimulus representation before interpreting metrics."
    )

    if trial_df.empty:

        st.info(
            "No stimulus trials are currently available."
        )

    else:

        c1, c2 = st.columns(
            2
        )

        with c1:

            st.pyplot(
                _plot_trials_per_patient(
                    trial_df
                ),
                clear_figure=True,
                use_container_width=True
            )

        with c2:

            st.pyplot(
                _plot_stimulus_distribution(
                    trial_df
                ),
                clear_figure=True,
                use_container_width=True
            )

        patient_counts = (
            trial_df
            .groupby(
                [
                    "patient_id",
                    "patient_code"
                ],
                dropna=False
            )
            .size()
            .reset_index(
                name="Trials"
            )
            .sort_values(
                "Trials",
                ascending=False
            )
        )

        patient_counts.columns = [
            "Patient ID",
            "Patient",
            "Trials"
        ]

        st.dataframe(
            patient_counts,
            use_container_width=True,
            hide_index=True
        )

        if (
            patient_counts["Trials"].max()
            >
            max(
                2,
                2
                *
                patient_counts["Trials"].median()
            )
        ):

            st.warning(
                "The number of trials is unbalanced across patients. This can cause patients with many trials to have greater influence on model fitting."
            )

    # --------------------------------------------------------
    # Stored prediction audit
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "## Stored Prediction Audit"
    )

    st.caption(
        "Retrospective comparison of stored model predictions with later observed responses."
    )

    linked = _linked_prediction_errors(
        trial_df
    )

    if linked.empty:

        st.info(
            "No trial rows currently contain both a stored model prediction and an observed response score. "
            "This section will populate as model-linked trials are completed."
        )

    else:

        descriptive_mae = float(
            linked["absolute_error"].mean()
        )

        descriptive_rmse = float(
            np.sqrt(
                np.mean(
                    linked["prediction_error"]
                    **
                    2
                )
            )
        )

        bias = float(
            linked["prediction_error"].mean()
        )

        c1, c2, c3, c4 = st.columns(
            4
        )

        c1.metric(
            "Model-Linked Trials",
            len(
                linked
            )
        )

        c2.metric(
            "Descriptive MAE",
            f"{descriptive_mae:.2f}"
        )

        c3.metric(
            "Descriptive RMSE",
            f"{descriptive_rmse:.2f}"
        )

        c4.metric(
            "Mean Error",
            f"{bias:+.2f}"
        )

        st.caption(
            "These are retrospective descriptive errors from stored predictions attached to completed trials. "
            "They are NOT a replacement for held-out patient-wise validation because some predictions may originate from deployment models trained on overlapping historical data."
        )

        c1, c2 = st.columns(
            2
        )

        with c1:

            st.pyplot(
                _plot_observed_vs_predicted(
                    linked
                ),
                clear_figure=True,
                use_container_width=True
            )

        with c2:

            st.pyplot(
                _plot_error_distribution(
                    linked
                ),
                clear_figure=True,
                use_container_width=True
            )

        display = linked[
            [
                "id",
                "patient_code",
                "source_type",
                "stimulus_type",
                "model_predicted_score",
                "response_score",
                "prediction_error",
                "absolute_error"
            ]
        ].copy()

        display.columns = [
            "Trial ID",
            "Patient",
            "Source",
            "Stimulus",
            "Predicted /100",
            "Observed /100",
            "Observed − Predicted",
            "Absolute Error"
        ]

        for column in [
            "Predicted /100",
            "Observed /100",
            "Observed − Predicted",
            "Absolute Error"
        ]:

            display[column] = pd.to_numeric(
                display[column],
                errors="coerce"
            ).round(
                1
            )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # Limitations
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "## Validation Limitations"
    )

    st.markdown(
        """
- **Small number of independent patients:** multiple trials from one patient do not replace additional independent participants.
- **Single patient-wise split:** performance can vary strongly depending on which patients are held out.
- **No external validation cohort:** the model has not been tested on an independent external dataset.
- **Prototype target:** the 0–100 response score is an engineering preference/tolerability score, not a validated clinical outcome.
- **Data imbalance:** stimulus types, frequencies, and patient trial counts may not be evenly represented.
- **No causal inference:** prediction accuracy cannot establish that an acoustic stimulus causes clinical improvement.
"""
    )

    # --------------------------------------------------------
    # Readiness
    # --------------------------------------------------------

    st.markdown(
        "## Current Model Readiness"
    )

    if (
        patients < 5
        or
        rows < 30
    ):

        st.error(
            "Current status: proof-of-concept ML only. The dataset is too small for reliable performance claims."
        )

    elif (
        patients < 10
        or
        rows < 100
    ):

        st.warning(
            "Current status: early experimental model. Suitable for demonstrating the SmartTinnitus pipeline, but not for clinical-performance claims."
        )

    else:

        st.info(
            "Current status: preliminary research evaluation. Larger prospective and external validation would still be required."
        )

    st.success(
        "For the PFE, the defensible claim is that SmartTinnitus implements a leakage-aware patient-wise validation workflow and transparently reports its current limitations."
    )
