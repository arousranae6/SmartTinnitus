import streamlit as st
from datetime import datetime
import sqlite3
import pandas as pd
from pathlib import Path
from login import (
    initialize_auth_state,
    show_login_page,
    show_logged_in_user_card
)
from patient_access import (
    create_patient_login
)
from provenance import (
    get_latest_profile_id,
    get_model_provenance,
    now_timestamp
)

from adaptive_personalization import (
    generate_adaptive_candidate_stimuli,
    get_adaptive_personalization_summary
)

from adaptive_session import (
    show_adaptive_session_page
)

from digital_twin import (
    show_digital_twin_page as _show_digital_twin_page
)

from streamlit.delta_generator import DeltaGenerator

from smarttinnitus_chatbot import (
    show_smarttinnitus_chatbot
)
import smarttinnitus_chatbot as smarttinnitus_chatbot_module

from ml_evaluation import (
    show_ml_evaluation_page
)

from database import (
    get_connection,
    initialize_database
)

from audio_utils import (
    generate_pure_tone,
    generate_stimulus
)

from optimization import (
    calculate_response_score,
    find_best_trial,
    recommend_next_trial
)

from pitch_matching import (
    initialize_pitch_state,
    get_pitch_options,
    update_pitch_search,
    reset_pitch_search,
    get_final_pitch
)

from audiogram import (
    AUDIOGRAM_FREQUENCIES,
    plot_audiogram
)

from report import (
    generate_patient_report
)

from ml_model import (
    load_model,
    predict_responses_batch,
    train_model,
    get_ml_dataset_summary,
    get_saved_model_metrics,
    model_exists
)

from care_coordination import (
    ensure_care_coordination_table,
    get_patient_care_summary,
    show_patient_care_page,
    show_audiologist_care_coordination,
)

# Backend/service layer. New UI code should consume services instead of adding
# direct SQL to app.py.
from backend.patient_service import (
    list_patients as backend_list_patients,
    list_patient_summaries as backend_list_patient_summaries,
    get_patient_id_by_code as backend_get_patient_id_by_code,
)
from backend.patient_context import (
    get_active_patient_id,
    get_active_patient_code,
    set_active_patient,
)
from backend.acoustic_assignment_service import (
    create_assignment as backend_create_acoustic_assignment,
    get_pending_assignment as backend_get_pending_acoustic_assignment,
    submit_assignment_feedback as backend_submit_acoustic_feedback,
    list_assignment_feedback as backend_list_acoustic_feedback,
)


# =====================================================
# ML MODEL ARTIFACT
# =====================================================

MODEL_FILE = "smarttinnitus_response_model.joblib"




# =====================================================
# PATIENT-FRIENDLY DIGITAL TWIN WRAPPER
# =====================================================

PATIENT_DIGITAL_TWIN_EXPLANATIONS = {
    "Acoustic Response Map": (
        "This map shows where the current model predicts higher or lower "
        "response scores across different sound settings. "
        "**How it helps you:** it lets you understand which frequency and "
        "relative-level areas may be worth exploring in future sound tests. "
        "These are model predictions, not proof that a sound will improve tinnitus."
    ),
    "Configuration Simulator": (
        "This view shows what the current model would predict for a hypothetical sound configuration. "
        "**How it helps you:** it explains the model's behavior, but it is informational only; "
        "actual sound selection and testing are performed by the audiologist."
    ),
    "Confidence & Active Learning": (
        "This view shows how strongly a model prediction is supported by your "
        "recorded trials and where the model still has uncertainty. "
        "**How it helps you:** it tells you when a prediction is supported by more "
        "personal evidence and where another carefully selected test could add useful information. "
        "Confidence here is a prototype model indicator, not a probability of treatment success."
    ),
    "Observed Evidence": (
        "This view contains the sounds you actually tested and the feedback you really gave. "
        "**How it helps you:** it separates your real listening history from AI/model predictions, "
        "so you can see which tested sounds were more comfortable or received better response scores."
    ),
    "Twin Evolution": (
        "This view follows how your personal acoustic-response information builds up over time "
        "from audiologist-supervised manual and AI-assisted acoustic results. "
        "**How it helps you:** it lets you see whether your preferred sounds and recorded responses "
        "stay similar or change as more results are collected. "
        "It describes your SmartTinnitus history; it is not a measure of medical disease progression."
    ),
}


AUDIOLOGIST_DIGITAL_TWIN_EXPLANATIONS = {
    "Acoustic Response Map": (
        "**Predicted view.** This map visualizes the model-estimated response landscape "
        "across acoustic parameter combinations. It helps identify regions that may be "
        "worth exploring, but it does not represent observed patient benefit."
    ),
    "Configuration Simulator": (
        "**Predicted view.** Use this as a what-if tool: choose an acoustic configuration "
        "and inspect the current model estimate before that configuration is necessarily tested. "
        "Simulation output is not an observed patient response."
    ),
    "Confidence & Active Learning": (
        "**Model-support view.** This indicates where the current patient-specific model has "
        "more or less supporting evidence and where another supervised test could be informative. "
        "Confidence is not a probability of clinical success."
    ),
    "Observed Evidence": (
        "**Observed view.** This contains configurations the patient actually tested together "
        "with recorded feedback. Treat this as patient-specific evidence and keep it separate "
        "from model predictions."
    ),
    "Twin Evolution": (
        "**Longitudinal model view.** This shows how recorded acoustic evidence and Standard/AI "
        "personalization results accumulate over time. It describes evolution of the SmartTinnitus "
        "response model, not progression of tinnitus as a disease."
    ),
}


def _resolve_digital_twin_patient_id(
    selected_value
):
    """Best-effort conversion of a Digital Twin patient selector value to patient_id."""

    if selected_value is None:
        return None

    if isinstance(
        selected_value,
        int
    ):
        return selected_value

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT id
            FROM patients
            WHERE patient_code = ?
            LIMIT 1
            """,
            (
                str(
                    selected_value
                ),
            )
        ).fetchone()

        return (
            row[0]
            if row is not None
            else None
        )

    finally:
        connection.close()


def _get_audiologist_twin_evidence_summary(
    patient_id
):
    """Summarize current acoustic evidence using the new SmartTinnitus architecture."""

    if patient_id is None:
        return {
            "observed_trials": 0,
            "standard_results": 0,
            "ai_results": 0,
            "audiogram_available": False,
        }

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                SUM(
                    CASE
                        WHEN comfort_score IS NOT NULL
                         AND tinnitus_noticeability IS NOT NULL
                         AND source_type NOT IN ('patient_standard_result')
                        THEN 1
                        ELSE 0
                    END
                ) AS observed_trials,
                SUM(
                    CASE
                        WHEN source_type IN ('audiologist_manual_result', 'patient_standard_result')
                        THEN 1
                        ELSE 0
                    END
                ) AS standard_results,
                SUM(
                    CASE
                        WHEN source_type IN ('audiologist_ai_result', 'patient_ai_result')
                        THEN 1
                        ELSE 0
                    END
                ) AS ai_results
            FROM stimulus_trials
            WHERE patient_id = ?
            """,
            (
                patient_id,
            )
        ).fetchone()

        audiogram_row = connection.execute(
            """
            SELECT COUNT(*)
            FROM audiograms
            WHERE patient_id = ?
            """,
            (
                patient_id,
            )
        ).fetchone()

        return {
            "observed_trials": int(
                row[0]
                or
                0
            ),
            "standard_results": int(
                row[1]
                or
                0
            ),
            "ai_results": int(
                row[2]
                or
                0
            ),
            "audiogram_available": bool(
                audiogram_row is not None
                and
                int(
                    audiogram_row[0]
                    or
                    0
                ) > 0
            ),
        }

    finally:
        connection.close()


def _render_audiologist_twin_evolution(
    patient_id
):
    """Render Twin Evolution without the obsolete separate Adaptive Session concept."""

    st.markdown(
        "## Twin Evolution"
    )

    if patient_id is None:

        st.info(
            "Select a patient to review their Digital Twin evolution."
        )

        return

    connection = get_connection()

    try:
        evolution_df = pd.read_sql_query(
            """
            SELECT
                id,
                source_type,
                stimulus_type,
                center_frequency_hz,
                bandwidth_hz,
                relative_level,
                model_predicted_score,
                comfort_score,
                tinnitus_noticeability,
                listen_again,
                COALESCE(tested_at, created_at) AS evidence_date
            FROM stimulus_trials
            WHERE patient_id = ?
              AND (
                    comfort_score IS NOT NULL
                    OR source_type IN (
                        'audiologist_manual_result',
                        'audiologist_ai_result',
                        'patient_standard_result',
                        'patient_ai_result'
                    )
              )
            ORDER BY id ASC
            """,
            connection,
            params=(
                patient_id,
            )
        )

        if evolution_df.empty:

            st.info(
                "No recorded acoustic evidence is available for this patient yet."
            )

            return

        source_labels = {
            "patient_personalization_initial":
                "Standard · Initial Test",
            "patient_personalization_refined":
                "Standard · Refined Test",
            "audiologist_manual_result":
                "Audiologist Manual · Supervised Result",
            "audiologist_ai_result":
                "AI-Assisted · Supervised Result",
            "patient_standard_result":
                "Legacy Standard · Final Result",
            "patient_ai_result":
                "Legacy AI · Patient-Tested Result",
            "audiologist_supervised_manual":
                "Audiologist · Supervised Test",
            "manual":
                "Manual Test",
        }

        observed_scores = []

        for _, row in evolution_df.iterrows():

            if (
                pd.notna(
                    row[
                        "comfort_score"
                    ]
                )
                and
                pd.notna(
                    row[
                        "tinnitus_noticeability"
                    ]
                )
            ):

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

            else:
                score = None

            observed_scores.append(
                score
            )

        evolution_df[
            "Observed Response"
        ] = observed_scores

        evolution_df[
            "Evidence Source"
        ] = evolution_df[
            "source_type"
        ].map(
            source_labels
        ).fillna(
            evolution_df[
                "source_type"
            ]
        )

        display_df = evolution_df[
            [
                "evidence_date",
                "Evidence Source",
                "stimulus_type",
                "center_frequency_hz",
                "relative_level",
                "model_predicted_score",
                "Observed Response"
            ]
        ].copy()

        display_df.columns = [
            "Date",
            "Evidence Source",
            "Stimulus",
            "Frequency (Hz)",
            "Relative Level",
            "Predicted Response /100",
            "Observed Response /100"
        ]

        for column in [
            "Predicted Response /100",
            "Observed Response /100"
        ]:
            display_df[
                column
            ] = pd.to_numeric(
                display_df[
                    column
                ],
                errors="coerce"
            ).round(
                1
            )

        summary = _get_audiologist_twin_evidence_summary(
            patient_id
        )

        e1, e2, e3, e4 = st.columns(
            4
        )

        e1.metric(
            "Observed Acoustic Trials",
            summary[
                "observed_trials"
            ]
        )

        e2.metric(
            "Manual Results",
            summary[
                "standard_results"
            ]
        )

        e3.metric(
            "AI-Tested Results",
            summary[
                "ai_results"
            ]
        )

        e4.metric(
            "Audiogram",
            (
                "Available"
                if summary[
                    "audiogram_available"
                ]
                else "Not available"
            )
        )

        st.caption(
            "Observed response = actual recorded patient feedback. "
            "Predicted response = model estimate. These values are intentionally kept separate."
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    finally:
        connection.close()


def _get_patient_personalization_result_counts(
    patient_id
):
    """Count completed Standard and AI patient personalization results."""

    if patient_id is None:
        return 0, 0

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                SUM(
                    CASE
                        WHEN source_type IN ('audiologist_manual_result', 'patient_standard_result')
                        THEN 1
                        ELSE 0
                    END
                ),
                SUM(
                    CASE
                        WHEN source_type IN ('audiologist_ai_result', 'patient_ai_result')
                        THEN 1
                        ELSE 0
                    END
                )
            FROM stimulus_trials
            WHERE patient_id = ?
            """,
            (
                patient_id,
            )
        ).fetchone()

        standard_count = int(
            row[0]
            or
            0
        )

        ai_count = int(
            row[1]
            or
            0
        )

        return (
            standard_count,
            ai_count
        )

    finally:
        connection.close()


def _render_patient_twin_evolution(
    patient_id
):
    """
    Patient-side Twin Evolution using the current Standard/AI architecture.

    The old separate Adaptive Session workflow is intentionally not shown here.
    """

    st.markdown(
        "## Twin Evolution"
    )

    st.info(
        "💡 **What this means for you:** "
        + PATIENT_DIGITAL_TWIN_EXPLANATIONS[
            "Twin Evolution"
        ]
    )

    connection = get_connection()

    try:
        evolution_df = pd.read_sql_query(
            """
            SELECT
                id,
                source_type,
                stimulus_type,
                center_frequency_hz,
                bandwidth_hz,
                relative_level,
                model_predicted_score,
                comfort_score,
                tinnitus_noticeability,
                listen_again,
                COALESCE(tested_at, created_at) AS result_date
            FROM stimulus_trials
            WHERE patient_id = ?
              AND source_type IN (
                    'audiologist_manual_result',
                    'audiologist_ai_result',
                    'patient_standard_result',
                    'patient_ai_result'
              )
            ORDER BY id ASC
            """,
            connection,
            params=(
                patient_id,
            )
        )

        if evolution_df.empty:

            st.info(
                "No completed supervised manual or AI-assisted acoustic results are available yet. "
                "Your audiologist must first complete a supervised acoustic test."
            )

            return

        response_scores = []

        for _, row in evolution_df.iterrows():

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

            response_scores.append(
                score
            )

        evolution_df[
            "Response Score"
        ] = response_scores

        evolution_df[
            "Pathway"
        ] = evolution_df[
            "source_type"
        ].map(
            {
                "audiologist_manual_result":
                    "Audiologist Manual",
                "audiologist_ai_result":
                    "AI-Assisted",
                "patient_standard_result":
                    "Legacy Standard",
                "patient_ai_result":
                    "Legacy AI-Assisted"
            }
        )

        display_df = evolution_df[
            [
                "result_date",
                "Pathway",
                "stimulus_type",
                "center_frequency_hz",
                "relative_level",
                "comfort_score",
                "tinnitus_noticeability",
                "Response Score"
            ]
        ].copy()

        display_df.columns = [
            "Date",
            "Pathway",
            "Stimulus",
            "Frequency (Hz)",
            "Relative Level",
            "Comfort /10",
            "Tinnitus Noticeability /10",
            "Observed Response /100"
        ]

        display_df[
            "Observed Response /100"
        ] = pd.to_numeric(
            display_df[
                "Observed Response /100"
            ],
            errors="coerce"
        ).round(
            1
        )

        standard_count = int(
            (
                display_df[
                    "Pathway"
                ]
                .isin(["Audiologist Manual", "Legacy Standard"])
            ).sum()
        )

        ai_count = int(
            (
                display_df[
                    "Pathway"
                ]
                .isin(["AI-Assisted", "Legacy AI-Assisted"])
            ).sum()
        )

        c1, c2, c3 = st.columns(
            3
        )

        c1.metric(
            "Manual Results",
            standard_count
        )

        c2.metric(
            "AI-Assisted Results",
            ai_count
        )

        c3.metric(
            "Total Supervised Results",
            len(
                display_df
            )
        )

        st.markdown(
            "### Your supervised acoustic history"
        )

        st.caption(
            "Observed response scores come from your recorded feedback. "
            "AI predictions, when available, are not treated as observed results."
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    finally:
        connection.close()


def show_digital_twin_page(
    current_role=None,
    auth_patient_id=None
):
    """
    Role-aware wrapper around the existing Digital Twin.

    Patient accounts receive plain-language explanations.
    Audiologist accounts receive professional interpretation guidance and
    current SmartTinnitus terminology without the obsolete separate
    Adaptive Session framing.
    """

    # =================================================
    # AUDIOLOGIST DIGITAL TWIN
    # =================================================

    if current_role != "patient":

        # Final clinical Digital Twin keeps only the views that add value to
        # the clinician-controlled workflow. Sound simulation belongs in
        # Acoustic Personalization; active-learning diagnostics are research-only.
        if st.session_state.get("digital_twin_view") in {
            "Configuration Simulator",
            "Confidence & Active Learning",
        }:
            st.session_state["digital_twin_view"] = "Acoustic Response Map"

        original_radio = st.radio
        original_selectbox = st.selectbox
        original_title = st.title
        original_metric = DeltaGenerator.metric
        original_caption = st.caption
        original_info = st.info
        original_warning = st.warning
        original_write = st.write
        original_markdown = st.markdown

        selected_view = {
            "value":
                st.session_state.get(
                    "digital_twin_view",
                    "Acoustic Response Map"
                )
        }

        selected_patient = {
            "id": auth_patient_id
        }

        def _replace_old_twin_terms(
            value
        ):

            if not isinstance(
                value,
                str
            ):
                return value

            replacements = [
                (
                    "Completed Adaptive Sessions",
                    "Recorded Acoustic Evidence"
                ),
                (
                    "Adaptive Sessions",
                    "Adaptive Testing"
                ),
                (
                    "Adaptive Session",
                    "Adaptive Testing"
                ),
                (
                    "adaptive sessions",
                    "adaptive testing"
                ),
                (
                    "adaptive session",
                    "adaptive testing"
                ),
            ]

            for old, new in replacements:
                value = value.replace(
                    old,
                    new
                )

            return value

        def professional_radio(
            label,
            *args,
            **kwargs
        ):

            if label == "Digital Twin View":
                allowed_views = [
                    "Acoustic Response Map",
                    "Observed Evidence",
                    "Twin Evolution",
                ]
                args_list = list(args)
                if args_list:
                    args_list[0] = [
                        option for option in list(args_list[0])
                        if option in allowed_views
                    ]
                elif "options" in kwargs:
                    kwargs["options"] = [
                        option for option in list(kwargs["options"])
                        if option in allowed_views
                    ]
                args = tuple(args_list)

            value = original_radio(
                label,
                *args,
                **kwargs
            )

            if label == "Digital Twin View":

                selected_view[
                    "value"
                ] = value

                if value == "Twin Evolution":

                    return (
                        "__AUDIOLOGIST_EVIDENCE_EVOLUTION__"
                    )

            return value

        def professional_selectbox(
            label,
            *args,
            **kwargs
        ):

            # The audiologist already opened a patient from Patients. Hide the
            # Digital Twin's legacy patient selector and return the matching
            # option automatically.
            if (
                isinstance(label, str)
                and "patient" in label.lower()
                and auth_patient_id is not None
            ):
                options = None
                if args:
                    options = args[0]
                elif "options" in kwargs:
                    options = kwargs["options"]

                if options is not None:
                    for option in list(options):
                        option_text = str(option)
                        resolved_id = _resolve_digital_twin_patient_id(option)
                        try:
                            is_match = int(resolved_id) == int(auth_patient_id)
                        except Exception:
                            is_match = resolved_id == auth_patient_id

                        # The underlying Digital Twin commonly formats options as
                        # "001 (ID 1)". Match that form directly so the legacy
                        # selector can be hidden instead of rendered.
                        try:
                            id_marker_match = f"(ID {int(auth_patient_id)})" in option_text
                        except Exception:
                            id_marker_match = False

                        if is_match or id_marker_match:
                            selected_patient["id"] = auth_patient_id
                            return option

            value = original_selectbox(
                label,
                *args,
                **kwargs
            )

            if (
                isinstance(label, str)
                and "patient" in label.lower()
            ):
                resolved_id = _resolve_digital_twin_patient_id(value)
                if resolved_id is not None:
                    selected_patient["id"] = resolved_id

            return value

        def professional_metric(
            self,
            *args,
            **kwargs
        ):

            args_list = list(
                args
            )

            label = (
                kwargs.get(
                    "label"
                )
                if
                "label" in kwargs
                else
                (
                    args_list[0]
                    if args_list
                    else None
                )
            )

            if label == "Completed Adaptive Sessions":

                summary = (
                    _get_audiologist_twin_evidence_summary(
                        selected_patient[
                            "id"
                        ]
                    )
                )

                new_label = (
                    "Recorded Acoustic Evidence"
                )

                new_value = (
                    summary[
                        "observed_trials"
                    ]
                )

                if args_list:

                    args_list[
                        0
                    ] = new_label

                else:

                    kwargs[
                        "label"
                    ] = new_label

                if len(
                    args_list
                ) >= 2:

                    args_list[
                        1
                    ] = new_value

                else:

                    kwargs[
                        "value"
                    ] = new_value

            elif isinstance(
                label,
                str
            ):

                new_label = _replace_old_twin_terms(
                    label
                )

                if args_list:

                    args_list[
                        0
                    ] = new_label

                else:

                    kwargs[
                        "label"
                    ] = new_label

            return original_metric(
                self,
                *args_list,
                **kwargs
            )

        def professional_caption(
            body,
            *args,
            **kwargs
        ):
            cleaned = _replace_old_twin_terms(body)

            if isinstance(cleaned, str):
                hidden_phrases = [
                    "Use this map to inspect where the current prototype model predicts",
                    "Star = highest point predicted by the current prototype model",
                    "Explore the patient's evolving acoustic-response profile",
                ]

                if any(phrase in cleaned for phrase in hidden_phrases):
                    return None

            return original_caption(cleaned, *args, **kwargs)

        def professional_info(
            body,
            *args,
            **kwargs
        ):
            cleaned = _replace_old_twin_terms(body)

            if isinstance(cleaned, str):
                hidden_phrases = [
                    "A high predicted score identifies a configuration",
                    "What this view means:",
                    "Professional interpretation:",
                    "Digital Twin = patient-specific acoustic-response model",
                ]

                if any(phrase in cleaned for phrase in hidden_phrases):
                    return None

            return original_info(cleaned, *args, **kwargs)

        def professional_write(
            *args,
            **kwargs
        ):
            updated_args = [
                _replace_old_twin_terms(value)
                for value in args
            ]

            for value in updated_args:
                if isinstance(value, str) and (
                    "The map simulates the current model's predicted prototype response score" in value
                ):
                    return None

            return original_write(*updated_args, **kwargs)

        def professional_markdown(
            body,
            *args,
            **kwargs
        ):

            return original_markdown(
                _replace_old_twin_terms(
                    body
                ),
                *args,
                **kwargs
            )

        def professional_title(body, *args, **kwargs):
            if body == "Patient Acoustic Digital Twin":
                return original_title(body, *args, **kwargs)
            if isinstance(body, str) and "Digital Twin" in body:
                body = "Patient Acoustic Digital Twin"
            return original_title(body, *args, **kwargs)

        st.radio = professional_radio
        st.selectbox = professional_selectbox
        st.title = professional_title
        DeltaGenerator.metric = (
            professional_metric
        )
        st.caption = professional_caption
        st.info = professional_info
        st.write = professional_write
        st.markdown = professional_markdown

        def clean_digital_twin_caption(*args, **kwargs):
            text = args[0] if args else kwargs.get("body", "")
            if isinstance(text, str) and text.startswith(
                "Explore the patient's evolving acoustic-response profile"
            ):
                return None
            return original_caption(*args, **kwargs)

        def clean_digital_twin_warning(*args, **kwargs):
            text = args[0] if args else kwargs.get("body", "")
            if isinstance(text, str) and text.startswith(
                "This Digital Twin represents a prototype acoustic-response model"
            ):
                return None
            return original_warning(*args, **kwargs)

        st.caption = clean_digital_twin_caption
        st.warning = clean_digital_twin_warning

        try:

            _show_digital_twin_page(
                current_role=current_role,
                auth_patient_id=auth_patient_id
            )

        finally:

            st.radio = original_radio
            st.selectbox = original_selectbox
            st.title = original_title
            DeltaGenerator.metric = (
                original_metric
            )
            st.caption = original_caption
            st.info = original_info
            st.warning = original_warning
            st.write = original_write
            st.markdown = original_markdown

        if (
            selected_view[
                "value"
            ]
            ==
            "Twin Evolution"
        ):

            _render_audiologist_twin_evolution(
                selected_patient[
                    "id"
                ]
            )

        return

    # =================================================
    # PATIENT DIGITAL TWIN
    # =================================================

    st.info(
        "The patient Digital Twin is informational. It summarizes recorded and model-predicted "
        "acoustic-response evidence, but it does not let the patient independently generate or test treatment sounds."
    )

    standard_count, ai_count = (
        _get_patient_personalization_result_counts(
            auth_patient_id
        )
    )

    # Patient Digital Twin is informational only. Remove the configuration simulator
    # so the patient does not independently explore/select acoustic prescriptions.
    if st.session_state.get("digital_twin_view") == "Configuration Simulator":
        st.session_state["digital_twin_view"] = "Acoustic Response Map"

    original_radio = st.radio
    original_metric = DeltaGenerator.metric

    selected_view = {
        "value":
            st.session_state.get(
                "digital_twin_view",
                "Acoustic Response Map"
            )
    }

    def patient_friendly_radio(
        label,
        *args,
        **kwargs
    ):

        if label == "Digital Twin View":
            args_list = list(args)
            if args_list:
                args_list[0] = [
                    option for option in args_list[0]
                    if option not in {"Configuration Simulator", "Confidence & Active Learning"}
                ]
            elif "options" in kwargs:
                kwargs["options"] = [
                    option for option in kwargs["options"]
                    if option not in {"Configuration Simulator", "Confidence & Active Learning"}
                ]
            args = tuple(args_list)

        value = original_radio(
            label,
            *args,
            **kwargs
        )

        if label == "Digital Twin View":

            selected_view[
                "value"
            ] = value

            if value != "Twin Evolution":

                explanation = (
                    PATIENT_DIGITAL_TWIN_EXPLANATIONS.get(
                        value
                    )
                )

                if explanation:

                    st.info(
                        "💡 **What this means for you:** "
                        +
                        explanation
                    )

            # Do not render the obsolete Adaptive Session based
            # Twin Evolution branch from the old module.
            if value == "Twin Evolution":

                return (
                    "__PATIENT_PERSONALIZATION_EVOLUTION__"
                )

        return value

    def patient_friendly_metric(
        self,
        *args,
        **kwargs
    ):

        args_list = list(
            args
        )

        label = (
            kwargs.get(
                "label"
            )
            if
            "label" in kwargs
            else
            (
                args_list[0]
                if args_list
                else None
            )
        )

        if label == "Completed Adaptive Sessions":

            new_label = (
                "Completed Supervised Acoustic Results"
            )

            new_value = (
                f"Manual {standard_count} • AI {ai_count}"
            )

            if args_list:

                args_list[
                    0
                ] = new_label

            else:

                kwargs[
                    "label"
                ] = new_label

            if len(
                args_list
            ) >= 2:

                args_list[
                    1
                ] = new_value

            else:

                kwargs[
                    "value"
                ] = new_value

        return original_metric(
            self,
            *args_list,
            **kwargs
        )

    st.radio = patient_friendly_radio
    DeltaGenerator.metric = (
        patient_friendly_metric
    )

    try:

        _show_digital_twin_page(
            current_role=current_role,
            auth_patient_id=auth_patient_id
        )

    finally:

        st.radio = original_radio
        DeltaGenerator.metric = (
            original_metric
        )

    if (
        selected_view[
            "value"
        ]
        ==
        "Twin Evolution"
    ):

        _render_patient_twin_evolution(
            auth_patient_id
        )



# =====================================================
# PATIENT FOLLOW-UP & PROGRESS
# =====================================================

def ensure_patient_followup_progress_columns(connection):
    """
    Extend the existing sessions table with patient-friendly longitudinal fields.

    Existing audiologist/session data is preserved. Columns are only added when
    missing so older SmartTinnitus databases remain compatible.
    """

    existing_columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(sessions)"
        ).fetchall()
    }

    additions = {
        "sleep_impact": "INTEGER",
        "concentration_impact": "INTEGER",
        "stress_score": "INTEGER",
        "change_since_last": "TEXT",
        "used_personalized_sound": "INTEGER",
        "sound_comfort": "INTEGER",
        "sound_listen_again": "INTEGER",
        "patient_note": "TEXT",
        "followup_source": "TEXT",
    }

    for column_name, column_type in additions.items():

        if column_name not in existing_columns:

            connection.execute(
                f"ALTER TABLE sessions "
                f"ADD COLUMN {column_name} {column_type}"
            )

    connection.commit()


def _patient_trend_label(values):
    """Describe the direction of the two most recent patient-reported scores."""

    clean_values = [
        float(value)
        for value in values
        if value is not None
        and
        not pd.isna(value)
    ]

    if len(clean_values) < 2:
        return "— Not enough data"

    previous_value = clean_values[-2]
    current_value = clean_values[-1]

    if current_value < previous_value:
        return "↓ Decreasing"

    if current_value > previous_value:
        return "↑ Increasing"

    return "→ Stable"


def _load_patient_followup_history(connection, patient_id):

    ensure_patient_followup_progress_columns(
        connection
    )

    return pd.read_sql_query(
        """
        SELECT
            id,
            session_date,
            annoyance_after AS annoyance,
            sleep_impact,
            concentration_impact,
            stress_score,
            tinnitus_noticeability_after AS noticeability,
            change_since_last,
            used_personalized_sound,
            sound_comfort,
            sound_listen_again,
            COALESCE(patient_note, notes) AS patient_note
        FROM sessions
        WHERE patient_id = ?
        ORDER BY session_date ASC, id ASC
        """,
        connection,
        params=(
            patient_id,
        )
    )


def _get_patient_acoustic_progress(connection, patient_id):
    """
    Return observed acoustic-response summary plus the latest Standard and AI
    personalization results. Observed and AI-predicted scores remain separate.
    """

    trials_df = pd.read_sql_query(
        """
        SELECT
            id,
            source_type,
            stimulus_type,
            center_frequency_hz,
            bandwidth_hz,
            relative_level,
            model_predicted_score,
            comfort_score,
            tinnitus_noticeability,
            listen_again,
            COALESCE(tested_at, created_at) AS result_date
        FROM stimulus_trials
        WHERE patient_id = ?
          AND comfort_score IS NOT NULL
          AND tinnitus_noticeability IS NOT NULL
        ORDER BY id ASC
        """,
        connection,
        params=(
            patient_id,
        )
    )

    preferred_stimulus = "Not enough data"
    best_observed_score = None

    if not trials_df.empty:

        observed_scores = []

        for _, row in trials_df.iterrows():

            observed_scores.append(
                calculate_response_score(
                    row["comfort_score"],
                    row["tinnitus_noticeability"],
                    row["listen_again"]
                )
            )

        trials_df[
            "observed_response_score"
        ] = observed_scores

        best_observed_score = float(
            trials_df[
                "observed_response_score"
            ].max()
        )

        mean_by_stimulus = (
            trials_df
            .groupby(
                "stimulus_type",
                dropna=True
            )[
                "observed_response_score"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        if not mean_by_stimulus.empty:

            preferred_stimulus = str(
                mean_by_stimulus.index[0]
            )

    def latest_result(
        source_types
    ):

        if isinstance(source_types, str):
            source_types = (source_types,)

        placeholders = ",".join(["?"] * len(source_types))

        result = connection.execute(
            f"""
            SELECT
                id,
                stimulus_type,
                center_frequency_hz,
                bandwidth_hz,
                relative_level,
                model_predicted_score,
                comfort_score,
                tinnitus_noticeability,
                listen_again,
                COALESCE(tested_at, created_at)
            FROM stimulus_trials
            WHERE patient_id = ?
              AND source_type IN ({placeholders})
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                patient_id,
                *source_types
            )
        ).fetchone()

        if result is None:
            return None

        (
            trial_id,
            stimulus_type,
            frequency_hz,
            bandwidth_hz,
            relative_level,
            predicted_score,
            comfort_score,
            noticeability,
            listen_again,
            result_date
        ) = result

        observed_score = None

        if (
            comfort_score is not None
            and
            noticeability is not None
        ):

            observed_score = calculate_response_score(
                comfort_score,
                noticeability,
                listen_again
            )

        return {
            "trial_id": trial_id,
            "stimulus_type": stimulus_type,
            "frequency_hz": frequency_hz,
            "bandwidth_hz": bandwidth_hz,
            "relative_level": relative_level,
            "predicted_score": predicted_score,
            "observed_score": observed_score,
            "result_date": result_date,
        }

    return {
        "preferred_stimulus": preferred_stimulus,
        "best_observed_score": best_observed_score,
        "standard": latest_result(
            ("audiologist_manual_result", "patient_standard_result")
        ),
        "ai": latest_result(
            ("audiologist_ai_result", "patient_ai_result")
        ),
    }


def show_patient_followup_progress_page(
    patient_id
):
    """
    Combined patient page:
    - record a new follow-up,
    - review previous follow-ups,
    - visualize patient-reported trends,
    - review Standard/AI acoustic-response progress,
    - see independent professional statuses.
    """

    st.title(
        "Follow-up & Progress"
    )

    st.caption(
        "Record how you are doing now and review how your patient-reported "
        "tinnitus impact and acoustic responses have changed over time."
    )

    st.info(
        "Scores and trends on this page are patient-reported observations. "
        "They are not proof of treatment effectiveness or disease progression."
    )

    if patient_id is None:

        st.error(
            "No patient record is linked to this account."
        )

        return

    connection = get_connection()

    try:

        ensure_patient_followup_progress_columns(
            connection
        )

        history_df = _load_patient_followup_history(
            connection,
            patient_id
        )

        acoustic_progress = _get_patient_acoustic_progress(
            connection,
            patient_id
        )

        # -------------------------------------------------
        # PROGRESS SUMMARY
        # -------------------------------------------------

        st.markdown(
            "## Progress Summary"
        )

        total_followups = int(
            len(
                history_df
            )
        )

        last_followup = (
            str(
                history_df.iloc[
                    -1
                ][
                    "session_date"
                ]
            )
            if not history_df.empty
            else "None yet"
        )

        annoyance_trend = _patient_trend_label(
            history_df[
                "annoyance"
            ].tolist()
            if not history_df.empty
            else []
        )

        noticeability_trend = _patient_trend_label(
            history_df[
                "noticeability"
            ].tolist()
            if not history_df.empty
            else []
        )

        summary_1, summary_2, summary_3, summary_4 = st.columns(
            4
        )

        summary_1.metric(
            "Total Follow-ups",
            total_followups
        )

        summary_2.metric(
            "Last Follow-up",
            last_followup
        )

        summary_3.metric(
            "Annoyance Trend",
            annoyance_trend
        )

        summary_4.metric(
            "Noticeability Trend",
            noticeability_trend
        )

        preferred_stimulus = acoustic_progress[
            "preferred_stimulus"
        ]

        best_observed_score = acoustic_progress[
            "best_observed_score"
        ]

        acoustic_1, acoustic_2 = st.columns(
            2
        )

        acoustic_1.metric(
            "Most Consistently Preferred Stimulus",
            preferred_stimulus
        )

        acoustic_2.metric(
            "Best Observed Acoustic Response",
            (
                f"{best_observed_score:.1f} / 100"
                if best_observed_score is not None
                else "Not enough data"
            )
        )

        # Success survives the rerun after saving.
        if st.session_state.pop(
            "patient_followup_saved",
            False
        ):

            st.success(
                "✅ Follow-up saved."
            )

        st.markdown("---")

        # -------------------------------------------------
        # NEW FOLLOW-UP
        # -------------------------------------------------

        st.markdown(
            "## Start a New Follow-up"
        )

        st.write(
            "Use the same 0–10 scales each time so your progress can be "
            "compared consistently."
        )

        used_sound_choice = st.radio(
            "Did you use one of your personalized sounds since your last follow-up?",
            [
                "No",
                "Yes"
            ],
            horizontal=True,
            key="patient_followup_used_sound"
        )

        with st.form(
            "patient_followup_form"
        ):

            session_date = st.date_input(
                "Follow-up Date",
                key="patient_followup_date"
            )

            f1, f2 = st.columns(
                2
            )

            with f1:

                annoyance = st.slider(
                    "Current annoyance",
                    0,
                    10,
                    5,
                    help="0 = no annoyance; 10 = extremely annoying."
                )

                sleep_impact = st.slider(
                    "Current sleep impact",
                    0,
                    10,
                    5,
                    help="0 = no impact; 10 = very strong impact."
                )

                concentration_impact = st.slider(
                    "Current concentration impact",
                    0,
                    10,
                    5,
                    help="0 = no impact; 10 = very strong impact."
                )

            with f2:

                stress_score = st.slider(
                    "Current stress",
                    0,
                    10,
                    5,
                    help="0 = no tinnitus-related stress; 10 = very strong stress."
                )

                noticeability = st.slider(
                    "Current tinnitus noticeability",
                    0,
                    10,
                    5,
                    help="0 = not noticeable; 10 = extremely noticeable."
                )

                change_since_last = st.selectbox(
                    "Compared with your last follow-up, how does your tinnitus feel?",
                    [
                        "Better",
                        "About the same",
                        "Worse",
                        "Not sure"
                    ]
                )

            sound_comfort = None
            sound_listen_again = None

            if used_sound_choice == "Yes":

                st.markdown(
                    "### Personalized sound feedback"
                )

                s1, s2 = st.columns(
                    2
                )

                with s1:

                    sound_comfort = st.slider(
                        "How comfortable was the personalized sound?",
                        0,
                        10,
                        5
                    )

                with s2:

                    sound_listen_again_label = st.radio(
                        "Would you use that sound again?",
                        [
                            "Yes",
                            "No"
                        ],
                        horizontal=True
                    )

                    sound_listen_again = (
                        1
                        if sound_listen_again_label == "Yes"
                        else 0
                    )

            patient_note = st.text_area(
                "Optional note",
                placeholder=(
                    "Anything important you noticed since your last follow-up..."
                )
            )

            save_followup = st.form_submit_button(
                "Save Follow-up",
                use_container_width=True
            )

        if save_followup:

            previous_annoyance = (
                int(
                    history_df.iloc[
                        -1
                    ][
                        "annoyance"
                    ]
                )
                if (
                    not history_df.empty
                    and
                    not pd.isna(
                        history_df.iloc[
                            -1
                        ][
                            "annoyance"
                        ]
                    )
                )
                else annoyance
            )

            previous_noticeability = (
                int(
                    history_df.iloc[
                        -1
                    ][
                        "noticeability"
                    ]
                )
                if (
                    not history_df.empty
                    and
                    not pd.isna(
                        history_df.iloc[
                            -1
                        ][
                            "noticeability"
                        ]
                    )
                )
                else noticeability
            )

            connection.execute(
                """
                INSERT INTO sessions
                (
                    patient_id,
                    session_date,
                    annoyance_before,
                    annoyance_after,
                    tinnitus_noticeability_before,
                    tinnitus_noticeability_after,
                    notes,
                    created_at,
                    sleep_impact,
                    concentration_impact,
                    stress_score,
                    change_since_last,
                    used_personalized_sound,
                    sound_comfort,
                    sound_listen_again,
                    patient_note,
                    followup_source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_id,
                    session_date.strftime(
                        "%Y-%m-%d"
                    ),
                    previous_annoyance,
                    annoyance,
                    previous_noticeability,
                    noticeability,
                    patient_note,
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    sleep_impact,
                    concentration_impact,
                    stress_score,
                    change_since_last,
                    (
                        1
                        if used_sound_choice == "Yes"
                        else 0
                    ),
                    sound_comfort,
                    sound_listen_again,
                    patient_note,
                    "patient"
                )
            )

            connection.commit()

            st.session_state[
                "patient_followup_saved"
            ] = True

            st.rerun()

        st.markdown("---")

        # -------------------------------------------------
        # FOLLOW-UP HISTORY
        # -------------------------------------------------

        st.markdown(
            "## Previous Follow-ups"
        )

        if history_df.empty:

            st.info(
                "No follow-up has been recorded yet."
            )

        else:

            display_history = history_df.copy()

            display_history[
                "Used Personalized Sound"
            ] = display_history[
                "used_personalized_sound"
            ].map(
                {
                    1: "Yes",
                    0: "No"
                }
            ).fillna(
                "Not recorded"
            )

            display_history[
                "Would Use Sound Again"
            ] = display_history[
                "sound_listen_again"
            ].map(
                {
                    1: "Yes",
                    0: "No"
                }
            ).fillna(
                "Not recorded"
            )

            display_history = display_history[
                [
                    "session_date",
                    "annoyance",
                    "sleep_impact",
                    "concentration_impact",
                    "stress_score",
                    "noticeability",
                    "change_since_last",
                    "Used Personalized Sound",
                    "sound_comfort",
                    "Would Use Sound Again",
                    "patient_note"
                ]
            ].rename(
                columns={
                    "session_date": "Date",
                    "annoyance": "Annoyance",
                    "sleep_impact": "Sleep Impact",
                    "concentration_impact": "Concentration Impact",
                    "stress_score": "Stress",
                    "noticeability": "Noticeability",
                    "change_since_last": "Compared with Last Follow-up",
                    "sound_comfort": "Sound Comfort",
                    "patient_note": "Note",
                }
            )

            st.dataframe(
                display_history.sort_values(
                    "Date",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("---")

        # -------------------------------------------------
        # MY PROGRESS OVER TIME
        # -------------------------------------------------

        st.markdown(
            "## My Progress Over Time"
        )

        st.caption(
            "These charts show your own repeated 0–10 reports. "
            "Lower or higher values are descriptive observations, not a clinical outcome by themselves."
        )

        if len(
            history_df
        ) < 2:

            st.info(
                "At least two follow-ups are needed to show a meaningful trend."
            )

        else:

            trend_df = history_df[
                [
                    "session_date",
                    "annoyance",
                    "sleep_impact",
                    "concentration_impact",
                    "stress_score",
                    "noticeability"
                ]
            ].copy()

            trend_df = trend_df.rename(
                columns={
                    "session_date": "Date",
                    "annoyance": "Annoyance",
                    "sleep_impact": "Sleep",
                    "concentration_impact": "Concentration",
                    "stress_score": "Stress",
                    "noticeability": "Noticeability",
                }
            )

            trend_df = trend_df.set_index(
                "Date"
            )

            st.line_chart(
                trend_df,
                use_container_width=True
            )

            st.write(
                f"**Annoyance:** {annoyance_trend}  |  "
                f"**Tinnitus noticeability:** {noticeability_trend}"
            )

        st.markdown("---")

        # -------------------------------------------------
        # ACOUSTIC PERSONALIZATION PROGRESS
        # -------------------------------------------------

        st.markdown(
            "## Professional Acoustic Testing Progress"
        )

        st.write(
            "Audiologist-manual and AI-assisted supervised testing stay separate. "
            "Each result keeps its own observed/predicted information and professional status."
        )

        standard_result = acoustic_progress[
            "standard"
        ]

        ai_result = acoustic_progress[
            "ai"
        ]

        p1, p2 = st.columns(
            2
        )

        with p1:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### 🎧 Audiologist Manual"
                )

                if standard_result is None:

                    st.info(
                        "No completed supervised manual result yet."
                    )

                else:

                    st.metric(
                        "Latest Stimulus",
                        standard_result[
                            "stimulus_type"
                        ]
                        or
                        "Not recorded"
                    )

                    st.metric(
                        "Frequency",
                        (
                            f"{int(round(standard_result['frequency_hz']))} Hz"
                            if standard_result[
                                "frequency_hz"
                            ] is not None
                            else "N/A"
                        )
                    )

                    st.metric(
                        "Observed Response",
                        (
                            f"{float(standard_result['observed_score']):.1f} / 100"
                            if standard_result[
                                "observed_score"
                            ] is not None
                            else "N/A"
                        )
                    )

                    standard_review = get_personalization_review(
                        connection,
                        "standard",
                        standard_result[
                            "trial_id"
                        ]
                    )

                    standard_status = (
                        standard_review[0]
                        if standard_review is not None
                        else "Not submitted"
                    )

                    st.write(
                        f"**Professional Status:** {standard_status}"
                    )

        with p2:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### 🤖 AI-Assisted Personalization"
                )

                if ai_result is None:

                    st.info(
                        "No completed AI-assisted result yet."
                    )

                else:

                    st.metric(
                        "Latest Stimulus",
                        ai_result[
                            "stimulus_type"
                        ]
                        or
                        "Not recorded"
                    )

                    st.metric(
                        "Frequency",
                        (
                            f"{int(round(ai_result['frequency_hz']))} Hz"
                            if ai_result[
                                "frequency_hz"
                            ] is not None
                            else "N/A"
                        )
                    )

                    st.metric(
                        "AI Predicted Response",
                        (
                            f"{float(ai_result['predicted_score']):.1f} / 100"
                            if ai_result[
                                "predicted_score"
                            ] is not None
                            else "N/A"
                        )
                    )

                    st.metric(
                        "Observed Response",
                        (
                            f"{float(ai_result['observed_score']):.1f} / 100"
                            if ai_result[
                                "observed_score"
                            ] is not None
                            else "N/A"
                        )
                    )

                    ai_review = get_personalization_review(
                        connection,
                        "ai",
                        ai_result[
                            "trial_id"
                        ]
                    )

                    ai_status = (
                        ai_review[0]
                        if ai_review is not None
                        else "Not submitted"
                    )

                    st.write(
                        f"**Professional Status:** {ai_status}"
                    )

        with st.expander(
            "Professional decision history"
        ):

            ensure_personalization_review_table(
                connection
            )

            review_df = pd.read_sql_query(
                """
                SELECT
                    pathway,
                    status,
                    decision,
                    final_stimulus_type,
                    final_frequency_hz,
                    clinician_note,
                    COALESCE(reviewed_at, created_at) AS review_date
                FROM personalization_reviews
                WHERE patient_id = ?
                ORDER BY id DESC
                """,
                connection,
                params=(
                    patient_id,
                )
            )

            if review_df.empty:

                st.info(
                    "No professional personalization decision has been recorded yet."
                )

            else:

                review_df[
                    "pathway"
                ] = review_df[
                    "pathway"
                ].map(
                    {
                        "standard": "Audiologist Manual",
                        "ai": "AI-Assisted"
                    }
                ).fillna(
                    review_df[
                        "pathway"
                    ]
                )

                review_df = review_df.rename(
                    columns={
                        "pathway": "Pathway",
                        "status": "Status",
                        "decision": "Decision",
                        "final_stimulus_type": "Final Stimulus",
                        "final_frequency_hz": "Final Frequency (Hz)",
                        "clinician_note": "Clinician Note",
                        "review_date": "Date",
                    }
                )

                st.dataframe(
                    review_df,
                    use_container_width=True,
                    hide_index=True
                )

    finally:

        connection.close()



# =====================================================
# AUDIOLOGIST FOLLOW-UP & PROGRESS
# =====================================================

def show_audiologist_followup_progress_page():
    """
    Professional longitudinal workspace.

    Patients remains the quick latest-state overview.
    This page is reserved for recording supervised follow-ups,
    reviewing the complete history, and interpreting descriptive trends.
    """

    st.title(
        "Follow-up & Progress"
    )

    st.caption(
        "Record supervised follow-ups and review the patient's complete longitudinal "
        "history. Patient-reported and audiologist-supervised records are kept distinguishable."
    )

    st.info(
        "Patients shows the latest follow-up snapshot. This page is for the full history "
        "and longitudinal work, so the latest-summary panel is intentionally not repeated here."
    )

    connection = get_connection()

    try:

        ensure_patient_followup_progress_columns(
            connection
        )

        patient_rows = connection.execute(
            """
            SELECT id, patient_code
            FROM patients
            ORDER BY patient_code
            """
        ).fetchall()

        if not patient_rows:

            st.warning(
                "Create a patient first."
            )

            return

        patient_options = {
            patient_code: patient_id
            for patient_id, patient_code
            in patient_rows
        }

        patient_codes = list(
            patient_options.keys()
        )

        active_patient_code = st.session_state.get(
            "audiologist_active_patient_code"
        )

        default_index = (
            patient_codes.index(
                active_patient_code
            )
            if active_patient_code in patient_codes
            else 0
        )

        selected_code = st.selectbox(
            "Patient",
            patient_codes,
            index=default_index,
            key="followup_patient"
        )

        patient_id = patient_options[
            selected_code
        ]

        st.session_state[
            "audiologist_active_patient_id"
        ] = patient_id

        st.session_state[
            "audiologist_active_patient_code"
        ] = selected_code

        record_tab, history_tab, trends_tab = st.tabs(
            [
                "🩺 Record Supervised Follow-up",
                "📋 Complete History",
                "📈 Trends & Longitudinal Analysis"
            ]
        )

        # -------------------------------------------------
        # RECORD SUPERVISED FOLLOW-UP
        # -------------------------------------------------

        with record_tab:

            st.markdown(
                "## Record Supervised Follow-up"
            )

            st.write(
                "Use this when the audiologist is with the patient and records "
                "short-term patient-reported measures before and after the supervised session."
            )

            st.warning(
                "Before/after differences are descriptive patient-reported observations. "
                "They are not evidence of therapeutic efficacy."
            )

            with st.form(
                "audiologist_supervised_followup_form"
            ):

                session_date = st.date_input(
                    "Session Date"
                )

                before_col, after_col = st.columns(
                    2
                )

                with before_col:

                    st.markdown(
                        "### Before"
                    )

                    annoyance_before = st.slider(
                        "Annoyance Before",
                        0,
                        10,
                        5,
                        key="audio_followup_annoyance_before"
                    )

                    noticeability_before = st.slider(
                        "Noticeability Before",
                        0,
                        10,
                        5,
                        key="audio_followup_noticeability_before"
                    )

                with after_col:

                    st.markdown(
                        "### After"
                    )

                    annoyance_after = st.slider(
                        "Annoyance After",
                        0,
                        10,
                        5,
                        key="audio_followup_annoyance_after"
                    )

                    noticeability_after = st.slider(
                        "Noticeability After",
                        0,
                        10,
                        5,
                        key="audio_followup_noticeability_after"
                    )

                notes = st.text_area(
                    "Professional Session Note",
                    placeholder=(
                        "Optional contextual note about the supervised session..."
                    )
                )

                save_session = st.form_submit_button(
                    "Save Supervised Follow-up",
                    use_container_width=True
                )

            if save_session:

                connection.execute(
                    """
                    INSERT INTO sessions
                    (
                        patient_id,
                        session_date,
                        annoyance_before,
                        annoyance_after,
                        tinnitus_noticeability_before,
                        tinnitus_noticeability_after,
                        notes,
                        created_at,
                        followup_source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patient_id,
                        session_date.strftime(
                            "%Y-%m-%d"
                        ),
                        int(
                            annoyance_before
                        ),
                        int(
                            annoyance_after
                        ),
                        int(
                            noticeability_before
                        ),
                        int(
                            noticeability_after
                        ),
                        notes,
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "audiologist_supervised"
                    )
                )

                connection.commit()

                st.success(
                    "Supervised follow-up saved."
                )

        # -------------------------------------------------
        # LOAD COMPLETE HISTORY ONCE FOR REVIEW + TRENDS
        # -------------------------------------------------

        history_df = pd.read_sql_query(
            """
            SELECT
                id,
                session_date,
                annoyance_before,
                annoyance_after,
                tinnitus_noticeability_before,
                tinnitus_noticeability_after,
                sleep_impact,
                concentration_impact,
                stress_score,
                change_since_last,
                used_personalized_sound,
                sound_comfort,
                sound_listen_again,
                COALESCE(patient_note, notes) AS note,
                followup_source,
                created_at
            FROM sessions
            WHERE patient_id = ?
            ORDER BY session_date ASC, id ASC
            """,
            connection,
            params=(
                patient_id,
            )
        )

        if not history_df.empty:

            def infer_source(row):

                explicit_source = row.get(
                    "followup_source"
                )

                if explicit_source == "patient":

                    return "👤 Patient-reported"

                if explicit_source == "audiologist_supervised":

                    return "👨‍⚕️ Audiologist-supervised"

                patient_specific_fields = [
                    "sleep_impact",
                    "concentration_impact",
                    "stress_score",
                    "change_since_last",
                    "used_personalized_sound",
                    "sound_comfort",
                    "sound_listen_again",
                ]

                if any(
                    pd.notna(
                        row.get(
                            field
                        )
                    )
                    for field in patient_specific_fields
                ):

                    return "👤 Patient-reported"

                return "👨‍⚕️ Audiologist-supervised / legacy"

            history_df[
                "Source"
            ] = history_df.apply(
                infer_source,
                axis=1
            )

        # -------------------------------------------------
        # COMPLETE HISTORY
        # -------------------------------------------------

        with history_tab:

            st.markdown(
                "## Complete Follow-up History"
            )

            if history_df.empty:

                st.info(
                    "No follow-up has been recorded for this patient yet."
                )

            else:

                display_df = history_df.copy()

                def format_before_after(
                    before,
                    after
                ):

                    before_ok = pd.notna(
                        before
                    )

                    after_ok = pd.notna(
                        after
                    )

                    if (
                        before_ok
                        and
                        after_ok
                        and
                        float(
                            before
                        )
                        !=
                        float(
                            after
                        )
                    ):

                        return (
                            f"{int(before)} → {int(after)}"
                        )

                    if after_ok:

                        return str(
                            int(
                                after
                            )
                        )

                    if before_ok:

                        return str(
                            int(
                                before
                            )
                        )

                    return "—"

                display_df[
                    "Annoyance"
                ] = display_df.apply(
                    lambda row:
                        format_before_after(
                            row[
                                "annoyance_before"
                            ],
                            row[
                                "annoyance_after"
                            ]
                        ),
                    axis=1
                )

                display_df[
                    "Noticeability"
                ] = display_df.apply(
                    lambda row:
                        format_before_after(
                            row[
                                "tinnitus_noticeability_before"
                            ],
                            row[
                                "tinnitus_noticeability_after"
                            ]
                        ),
                    axis=1
                )

                display_df[
                    "Used Personalized Sound"
                ] = display_df[
                    "used_personalized_sound"
                ].map(
                    {
                        1: "Yes",
                        0: "No"
                    }
                ).fillna(
                    "—"
                )

                display_df[
                    "Would Use Again"
                ] = display_df[
                    "sound_listen_again"
                ].map(
                    {
                        1: "Yes",
                        0: "No"
                    }
                ).fillna(
                    "—"
                )

                display_df = display_df[
                    [
                        "session_date",
                        "Source",
                        "Annoyance",
                        "Noticeability",
                        "sleep_impact",
                        "concentration_impact",
                        "stress_score",
                        "change_since_last",
                        "Used Personalized Sound",
                        "sound_comfort",
                        "Would Use Again",
                        "note"
                    ]
                ].rename(
                    columns={
                        "session_date": "Date",
                        "sleep_impact": "Sleep Impact",
                        "concentration_impact": "Concentration Impact",
                        "stress_score": "Stress",
                        "change_since_last": "Compared with Previous",
                        "sound_comfort": "Sound Comfort",
                        "note": "Note"
                    }
                )

                st.dataframe(
                    display_df.sort_values(
                        "Date",
                        ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Patient-reported entries come from the patient's Follow-up & Progress page. "
                    "Audiologist-supervised entries are recorded in this professional workspace."
                )

        # -------------------------------------------------
        # TRENDS & LONGITUDINAL ANALYSIS
        # -------------------------------------------------

        with trends_tab:

            st.markdown(
                "## Trends & Longitudinal Analysis"
            )

            if history_df.empty:

                st.info(
                    "At least one follow-up is needed before longitudinal information can be shown."
                )

            else:

                analysis_df = history_df.copy()

                analysis_df[
                    "Annoyance"
                ] = pd.to_numeric(
                    analysis_df[
                        "annoyance_after"
                    ],
                    errors="coerce"
                )

                analysis_df[
                    "Noticeability"
                ] = pd.to_numeric(
                    analysis_df[
                        "tinnitus_noticeability_after"
                    ],
                    errors="coerce"
                )

                patient_count = int(
                    analysis_df[
                        "Source"
                    ].str.startswith(
                        "👤"
                    ).sum()
                )

                supervised_count = int(
                    analysis_df[
                        "Source"
                    ].str.startswith(
                        "👨‍⚕️"
                    ).sum()
                )

                t1, t2, t3 = st.columns(
                    3
                )

                t1.metric(
                    "Recorded Follow-ups",
                    len(
                        analysis_df
                    )
                )

                t2.metric(
                    "Patient-reported",
                    patient_count
                )

                t3.metric(
                    "Audiologist-supervised",
                    supervised_count
                )

                trend_col1, trend_col2 = st.columns(
                    2
                )

                with trend_col1:

                    st.metric(
                        "Annoyance Direction",
                        _patient_trend_label(
                            analysis_df[
                                "Annoyance"
                            ].tolist()
                        )
                    )

                with trend_col2:

                    st.metric(
                        "Noticeability Direction",
                        _patient_trend_label(
                            analysis_df[
                                "Noticeability"
                            ].tolist()
                        )
                    )

                chart_df = analysis_df[
                    [
                        "session_date",
                        "Annoyance",
                        "Noticeability"
                    ]
                ].copy()

                chart_df = chart_df.dropna(
                    subset=[
                        "session_date"
                    ]
                )

                if not chart_df.empty:

                    st.markdown(
                        "### Patient-reported scores over time"
                    )

                    st.line_chart(
                        chart_df.set_index(
                            "session_date"
                        )
                    )

                supervised_df = analysis_df[
                    analysis_df[
                        "Source"
                    ].str.startswith(
                        "👨‍⚕️"
                    )
                ].copy()

                if not supervised_df.empty:

                    st.markdown(
                        "### Supervised before/after observations"
                    )

                    supervised_display = supervised_df[
                        [
                            "session_date",
                            "annoyance_before",
                            "annoyance_after",
                            "tinnitus_noticeability_before",
                            "tinnitus_noticeability_after"
                        ]
                    ].copy()

                    supervised_display[
                        "Annoyance Change"
                    ] = (
                        pd.to_numeric(
                            supervised_display[
                                "annoyance_after"
                            ],
                            errors="coerce"
                        )
                        -
                        pd.to_numeric(
                            supervised_display[
                                "annoyance_before"
                            ],
                            errors="coerce"
                        )
                    )

                    supervised_display[
                        "Noticeability Change"
                    ] = (
                        pd.to_numeric(
                            supervised_display[
                                "tinnitus_noticeability_after"
                            ],
                            errors="coerce"
                        )
                        -
                        pd.to_numeric(
                            supervised_display[
                                "tinnitus_noticeability_before"
                            ],
                            errors="coerce"
                        )
                    )

                    supervised_display = supervised_display.rename(
                        columns={
                            "session_date": "Date",
                            "annoyance_before": "Annoyance Before",
                            "annoyance_after": "Annoyance After",
                            "tinnitus_noticeability_before": "Noticeability Before",
                            "tinnitus_noticeability_after": "Noticeability After"
                        }
                    )

                    st.dataframe(
                        supervised_display.sort_values(
                            "Date",
                            ascending=False
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                st.warning(
                    "Longitudinal trends and before/after differences are descriptive. "
                    "They should not be interpreted as proof that a sound or intervention caused improvement."
                )

    finally:

        connection.close()


# =====================================================
# GET PATIENT TRIAL HISTORY FOR ADAPTIVE PERSONALIZATION
# =====================================================

def get_patient_trial_history_for_adaptation(
    connection,
    patient_id
):

    query = """
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
    """

    history_df = pd.read_sql_query(
        query,
        connection,
        params=(
            patient_id,
        )
    )

    if history_df.empty:

        history_df[
            "response_score"
        ] = []

        return history_df

    response_scores = []

    for _, row in history_df.iterrows():

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

        response_scores.append(
            score
        )

    history_df[
        "response_score"
    ] = response_scores

    return history_df



# =====================================================
# ROLE-BASED PATIENT ACCESS
# =====================================================
#
# Audiologist:
#   can choose among all available patients.
#
# Patient:
#   is automatically locked to the patient_id linked
#   to the login account. No patient-selection dropdown
#   is shown.
# =====================================================


def _extract_patient_id_from_option_value(
    value
):

    if isinstance(
        value,
        dict
    ):

        return value.get(
            "id"
        )

    if isinstance(
        value,
        (
            tuple,
            list
        )
    ):

        if len(
            value
        ) > 0:

            return value[
                0
            ]

        return None

    return value


def select_patient_for_current_user(
    options,
    label="Patient",
    key=None
):

    if not options:

        return None

    current_role = (
        st.session_state.get(
            "auth_role"
        )
    )

    # -------------------------------------------------
    # PATIENT ACCOUNT
    # -------------------------------------------------

    if (
        current_role
        ==
        "patient"
    ):

        linked_patient_id = (
            st.session_state.get(
                "auth_patient_id"
            )
        )

        if linked_patient_id is None:

            st.error(
                "This patient login is not linked to a patient record. "
                "Please contact the audiologist."
            )

            st.stop()

        for option_label, option_value in options.items():

            option_patient_id = (
                _extract_patient_id_from_option_value(
                    option_value
                )
            )

            try:

                same_patient = (
                    int(
                        option_patient_id
                    )
                    ==
                    int(
                        linked_patient_id
                    )
                )

            except Exception:

                same_patient = (
                    option_patient_id
                    ==
                    linked_patient_id
                )

            if same_patient:

                st.caption(
                    f"Logged in as Patient {option_label}"
                )

                return option_label

        st.error(
            "The patient record linked to this login account "
            "could not be found."
        )

        st.stop()

    # -------------------------------------------------
    # AUDIOLOGIST ACCOUNT
    # -------------------------------------------------

    return st.selectbox(
        label,
        list(
            options.keys()
        ),
        key=key
    )



def navigate_to_page(
    target_page
):

    # When a patient opens Acoustic Personalization,
    # always begin on the two-pathway choice screen.
    if (
        target_page == "Acoustic Personalization"
        and
        st.session_state.get("auth_role") == "patient"
    ):
        st.session_state[
            "patient_personalization_path"
        ] = None

    st.session_state[
        "main_navigation"
    ] = target_page


def start_new_patient_assessment():
    """
    Reset the guided patient assessment and navigate safely.

    This is used as a Streamlit button callback so main_navigation
    is changed before the sidebar radio widget is instantiated.
    """

    st.session_state[
        "patient_assessment_step"
    ] = 1

    st.session_state[
        "patient_assessment_profile_id"
    ] = None

    st.session_state[
        "guided_manual_pitch_hz"
    ] = 4000

    st.session_state[
        "guided_manual_loudness"
    ] = 0.03

    st.session_state[
        "guided_safety_level"
    ] = 0.03

    st.session_state[
        "guided_safety_confirmed"
    ] = False

    reset_pitch_search(
        st.session_state
    )

    st.session_state[
        "main_navigation"
    ] = "Tinnitus Assessment"


def reset_patient_personalization_state():
    """Reset the patient's standard (non-AI) personalization workflow."""

    st.session_state[
        "patient_personalization_step"
    ] = 1

    st.session_state[
        "patient_personalization_initial_index"
    ] = 0

    st.session_state[
        "patient_personalization_refine_index"
    ] = 0

    st.session_state[
        "patient_personalization_initial_results"
    ] = []

    st.session_state[
        "patient_personalization_refined_results"
    ] = []

    st.session_state[
        "patient_personalization_refine_candidates"
    ] = []

    st.session_state[
        "patient_personalization_standard_result_trial_id"
    ] = None


def reset_patient_ai_personalization_state():
    """Reset the patient's optional AI-personalization workflow."""

    st.session_state[
        "patient_ai_recommendation"
    ] = None

    st.session_state[
        "patient_ai_top_candidates"
    ] = []

    st.session_state[
        "patient_ai_recommendation_created_at"
    ] = None

    st.session_state[
        "patient_ai_result_trial_id"
    ] = None

    st.session_state[
        "patient_ai_observed_score"
    ] = None


def set_patient_personalization_path(pathway):
    """Switch the patient personalization sub-view safely via callback."""

    st.session_state[
        "patient_personalization_path"
    ] = pathway


def ensure_personalization_review_table(connection):
    """
    Create professional-review records for personalization results.

    Standard and AI pathways are reviewed independently. Each completed
    pathway has its own row, its own status, its own professional decision,
    and its own final clinician-reviewed configuration.
    """

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS personalization_reviews
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            profile_id INTEGER,
            pathway TEXT NOT NULL,
            source_trial_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending Review',
            decision TEXT,
            final_stimulus_type TEXT,
            final_frequency_hz REAL,
            final_bandwidth_hz REAL,
            final_relative_level REAL,
            clinician_note TEXT,
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            UNIQUE(pathway, source_trial_id)
        )
        """
    )

    connection.commit()


def create_personalization_review_request(
    connection,
    patient_id,
    profile_id,
    pathway,
    source_trial_id
):
    """Create one pending professional-review request for a pathway result."""

    ensure_personalization_review_table(
        connection
    )

    connection.execute(
        """
        INSERT OR IGNORE INTO personalization_reviews
        (
            patient_id,
            profile_id,
            pathway,
            source_trial_id,
            status,
            created_at
        )
        VALUES (?, ?, ?, ?, 'Pending Review', ?)
        """,
        (
            patient_id,
            profile_id,
            pathway,
            source_trial_id,
            now_timestamp()
        )
    )

    connection.commit()


def get_personalization_review(connection, pathway, source_trial_id):
    """Return the current professional status for one saved pathway result."""

    ensure_personalization_review_table(
        connection
    )

    row = connection.execute(
        """
        SELECT
            status,
            decision,
            final_stimulus_type,
            final_frequency_hz,
            final_bandwidth_hz,
            final_relative_level,
            clinician_note,
            reviewed_at
        FROM personalization_reviews
        WHERE pathway = ?
          AND source_trial_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (
            pathway,
            source_trial_id
        )
    ).fetchone()

    return row



def get_latest_patient_pathway_review(connection, patient_id, pathway):
    """
    Return the latest professional review for one patient and one pathway.

    Standard and AI are queried independently so their statuses can differ.
    """

    ensure_personalization_review_table(
        connection
    )

    return connection.execute(
        """
        SELECT
            status,
            decision,
            final_stimulus_type,
            final_frequency_hz,
            final_bandwidth_hz,
            final_relative_level,
            clinician_note,
            reviewed_at,
            source_trial_id
        FROM personalization_reviews
        WHERE patient_id = ?
          AND pathway = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (
            patient_id,
            pathway
        )
    ).fetchone()




# =====================================================
# AUDIOLOGIST PATIENT HUB + INDEPENDENT PATHWAY REVIEW
# =====================================================

def get_latest_personalization_trial(
    connection,
    patient_id,
    pathway
):
    """Return the latest completed Standard or AI patient result."""

    source_types = (
        ("audiologist_manual_result", "patient_standard_result")
        if pathway == "standard"
        else ("audiologist_ai_result", "patient_ai_result")
    )

    row = connection.execute(
        """
        SELECT
            id,
            stimulus_type,
            center_frequency_hz,
            bandwidth_hz,
            relative_level,
            model_predicted_score,
            comfort_score,
            tinnitus_noticeability,
            listen_again,
            COALESCE(tested_at, created_at)
        FROM stimulus_trials
        WHERE patient_id = ?
          AND source_type IN (?, ?)
        ORDER BY id DESC
        LIMIT 1
        """,
        (
            patient_id,
            source_types[0],
            source_types[1]
        )
    ).fetchone()

    if row is None:
        return None

    (
        trial_id,
        stimulus_type,
        frequency_hz,
        bandwidth_hz,
        relative_level,
        predicted_score,
        comfort_score,
        noticeability,
        listen_again,
        result_date
    ) = row

    observed_score = None

    if (
        comfort_score is not None
        and
        noticeability is not None
    ):

        observed_score = calculate_response_score(
            comfort_score,
            noticeability,
            listen_again
        )

    return {
        "trial_id": trial_id,
        "stimulus_type": stimulus_type,
        "frequency_hz": frequency_hz,
        "bandwidth_hz": bandwidth_hz,
        "relative_level": relative_level,
        "predicted_score": predicted_score,
        "comfort_score": comfort_score,
        "noticeability": noticeability,
        "listen_again": listen_again,
        "observed_score": observed_score,
        "result_date": result_date,
    }


def save_independent_personalization_review(
    connection,
    patient_id,
    profile_id,
    pathway,
    trial,
    status,
    decision,
    clinician_note="",
    final_stimulus_type=None,
    final_frequency_hz=None,
    final_bandwidth_hz=None,
    final_relative_level=None
):
    """
    Save one pathway-specific professional decision.

    Standard and AI rows are updated independently. Patient-facing pages read
    the same personalization_reviews record, so the new status is visible
    there immediately after the next Streamlit rerun/page load.
    """

    create_personalization_review_request(
        connection,
        patient_id,
        profile_id,
        pathway,
        trial["trial_id"]
    )

    if status == "Approved":

        final_stimulus_type = trial[
            "stimulus_type"
        ]

        final_frequency_hz = trial[
            "frequency_hz"
        ]

        final_bandwidth_hz = trial[
            "bandwidth_hz"
        ]

        final_relative_level = trial[
            "relative_level"
        ]

    elif status == "Rejected":

        final_stimulus_type = None
        final_frequency_hz = None
        final_bandwidth_hz = None
        final_relative_level = None

    connection.execute(
        """
        UPDATE personalization_reviews
        SET
            status = ?,
            decision = ?,
            final_stimulus_type = ?,
            final_frequency_hz = ?,
            final_bandwidth_hz = ?,
            final_relative_level = ?,
            clinician_note = ?,
            reviewed_at = ?
        WHERE pathway = ?
          AND source_trial_id = ?
        """,
        (
            status,
            decision,
            final_stimulus_type,
            final_frequency_hz,
            final_bandwidth_hz,
            final_relative_level,
            clinician_note.strip()
            if clinician_note
            else None,
            now_timestamp(),
            pathway,
            trial["trial_id"]
        )
    )

    connection.commit()


def _safe_sound_frequency(
    stimulus_type,
    frequency_hz
):

    if (
        stimulus_type == "White Noise"
        or
        frequency_hz is None
    ):
        return 1000.0

    return float(
        frequency_hz
    )


def _safe_sound_bandwidth(
    bandwidth_hz
):

    return float(
        bandwidth_hz
        if bandwidth_hz is not None
        else 500.0
    )


def _safe_sound_level(
    relative_level
):

    level = float(
        relative_level
        if relative_level is not None
        else 0.03
    )

    return max(
        0.01,
        min(
            0.20,
            level
        )
    )


def render_audiologist_pathway_review(
    connection,
    patient_id,
    profile_id,
    pathway,
    trial
):
    """Render one independent Standard/AI professional-review card."""

    pathway_title = (
        "Audiologist Manual Recommendation"
        if pathway == "standard"
        else "AI-Assisted Recommendation"
    )

    icon = (
        "🎧"
        if pathway == "standard"
        else "🤖"
    )

    with st.container(
        border=True
    ):

        st.markdown(
            f"### {icon} {pathway_title}"
        )

        if trial is None:

            st.info(
                f"No completed {pathway_title} result is available yet."
            )

            return

        review = get_personalization_review(
            connection,
            pathway,
            trial[
                "trial_id"
            ]
        )

        if review is None:

            create_personalization_review_request(
                connection,
                patient_id,
                profile_id,
                pathway,
                trial[
                    "trial_id"
                ]
            )

            review = get_personalization_review(
                connection,
                pathway,
                trial[
                    "trial_id"
                ]
            )

        status = (
            review[0]
            if review is not None
            else "Pending Review"
        )

        # ---------------------------------------------
        # RESULT SUMMARY
        # ---------------------------------------------

        r1, r2, r3 = st.columns(
            3
        )

        r1.metric(
            "Candidate",
            trial[
                "stimulus_type"
            ]
            or
            "Not recorded"
        )

        r2.metric(
            "Frequency",
            (
                "Broadband"
                if trial[
                    "stimulus_type"
                ] == "White Noise"
                else (
                    f"{int(round(float(trial['frequency_hz'])))} Hz"
                    if trial[
                        "frequency_hz"
                    ] is not None
                    else "N/A"
                )
            )
        )

        r3.metric(
            "Observed Response",
            (
                f"{float(trial['observed_score']):.1f} / 100"
                if trial[
                    "observed_score"
                ] is not None
                else "N/A"
            )
        )

        if pathway == "ai":

            st.metric(
                "AI Predicted Response",
                (
                    f"{float(trial['predicted_score']):.1f} / 100"
                    if trial[
                        "predicted_score"
                    ] is not None
                    else "N/A"
                )
            )

        detail_1, detail_2, detail_3 = st.columns(
            3
        )

        detail_1.write(
            "**Relative level:** "
            + (
                f"{float(trial['relative_level']):.02f}"
                if trial[
                    "relative_level"
                ] is not None
                else "N/A"
            )
        )

        detail_2.write(
            "**Comfort:** "
            + (
                f"{int(trial['comfort_score'])}/10"
                if trial[
                    "comfort_score"
                ] is not None
                else "N/A"
            )
        )

        detail_3.write(
            "**Noticeability:** "
            + (
                f"{int(trial['noticeability'])}/10"
                if trial[
                    "noticeability"
                ] is not None
                else "N/A"
            )
        )

        if (
            trial[
                "stimulus_type"
            ] == "Narrow-Band Noise"
        ):

            st.write(
                "**Bandwidth:**",
                (
                    f"{int(round(float(trial['bandwidth_hz'])))} Hz"
                    if trial[
                        "bandwidth_hz"
                    ] is not None
                    else "N/A"
                )
            )

        st.caption(
            "Relative level is an uncalibrated digital level, not dB SPL or dB HL."
        )

        # ---------------------------------------------
        # LISTEN BEFORE DECIDING
        # ---------------------------------------------

        st.markdown(
            "#### Listen to the candidate before review"
        )

        st.write(
            "Use the player below to inspect the exact candidate associated "
            "with this pathway result before making the professional decision."
        )

        try:

            candidate_audio = generate_stimulus(
                stimulus_type=trial[
                    "stimulus_type"
                ],
                frequency=_safe_sound_frequency(
                    trial[
                        "stimulus_type"
                    ],
                    trial[
                        "frequency_hz"
                    ]
                ),
                duration=2,
                volume=_safe_sound_level(
                    trial[
                        "relative_level"
                    ]
                ),
                bandwidth=_safe_sound_bandwidth(
                    trial[
                        "bandwidth_hz"
                    ]
                )
            )

            st.audio(
                candidate_audio,
                format="audio/wav"
            )

        except Exception as error:

            st.error(
                "Could not generate this review sound: "
                f"{error}"
            )

        # ---------------------------------------------
        # CURRENT PROFESSIONAL STATUS
        # ---------------------------------------------

        if status == "Approved":

            st.success(
                "✅ Professional Status: Approved"
            )

        elif status == "Modified":

            st.warning(
                "✏️ Professional Status: Modified"
            )

        elif status == "Rejected":

            st.error(
                "❌ Professional Status: Rejected"
            )

        else:

            st.info(
                "⏳ Professional Status: Pending Review"
            )

        st.caption(
            "This decision belongs to the selected professional recommendation method. "
            "Manual and AI-assisted evidence remain traceable independently."
        )

        note_key = (
            f"review_note_{pathway}_{trial['trial_id']}"
        )

        default_note = (
            review[6]
            if (
                review is not None
                and
                review[6] is not None
            )
            else ""
        )

        clinician_note = st.text_area(
            "Professional note",
            value=default_note,
            key=note_key,
            placeholder=(
                "Optional rationale or note visible in the professional review history."
            )
        )

        approve_col, reject_col = st.columns(
            2
        )

        with approve_col:

            if st.button(
                "✅ Approve Candidate",
                use_container_width=True,
                key=(
                    f"approve_{pathway}_{trial['trial_id']}"
                )
            ):

                save_independent_personalization_review(
                    connection,
                    patient_id,
                    profile_id,
                    pathway,
                    trial,
                    status="Approved",
                    decision="Approve",
                    clinician_note=clinician_note
                )

                st.session_state[
                    "audiologist_review_flash"
                ] = (
                    f"{pathway_title} approved. "
                    "The patient-side professional status is now Approved."
                )

                st.rerun()

        with reject_col:

            if st.button(
                "❌ Reject Candidate",
                use_container_width=True,
                key=(
                    f"reject_{pathway}_{trial['trial_id']}"
                )
            ):

                save_independent_personalization_review(
                    connection,
                    patient_id,
                    profile_id,
                    pathway,
                    trial,
                    status="Rejected",
                    decision="Reject",
                    clinician_note=clinician_note
                )

                st.session_state[
                    "audiologist_review_flash"
                ] = (
                    f"{pathway_title} rejected. "
                    "The patient-side professional status is now Rejected."
                )

                st.rerun()

        # ---------------------------------------------
        # OPTIONAL MODIFICATION
        # ---------------------------------------------

        with st.expander(
            "✏️ Modify candidate before accepting"
        ):

            st.write(
                "Use this only when the candidate is useful but the final "
                "clinician-reviewed configuration should be adjusted."
            )

            stimulus_options = [
                "Pure Tone",
                "Narrow-Band Noise",
                "White Noise"
            ]

            current_type = (
                trial[
                    "stimulus_type"
                ]
                if trial[
                    "stimulus_type"
                ] in stimulus_options
                else "Pure Tone"
            )

            modified_type = st.selectbox(
                "Final stimulus type",
                stimulus_options,
                index=stimulus_options.index(
                    current_type
                ),
                key=(
                    f"modify_type_{pathway}_{trial['trial_id']}"
                )
            )

            modified_frequency = None

            if modified_type != "White Noise":

                modified_frequency = st.number_input(
                    "Final frequency (Hz)",
                    min_value=250.0,
                    max_value=10000.0,
                    value=float(
                        trial[
                            "frequency_hz"
                        ]
                        if trial[
                            "frequency_hz"
                        ] is not None
                        else 4000.0
                    ),
                    step=50.0,
                    key=(
                        f"modify_freq_{pathway}_{trial['trial_id']}"
                    )
                )

            modified_bandwidth = None

            if modified_type == "Narrow-Band Noise":

                modified_bandwidth = st.number_input(
                    "Final bandwidth (Hz)",
                    min_value=50.0,
                    max_value=4000.0,
                    value=float(
                        trial[
                            "bandwidth_hz"
                        ]
                        if trial[
                            "bandwidth_hz"
                        ] is not None
                        else 500.0
                    ),
                    step=50.0,
                    key=(
                        f"modify_bw_{pathway}_{trial['trial_id']}"
                    )
                )

            modified_level = st.slider(
                "Final relative level",
                0.01,
                0.20,
                _safe_sound_level(
                    trial[
                        "relative_level"
                    ]
                ),
                0.01,
                key=(
                    f"modify_level_{pathway}_{trial['trial_id']}"
                )
            )

            st.caption(
                "Preview of the modified configuration:"
            )

            try:

                modified_audio = generate_stimulus(
                    stimulus_type=modified_type,
                    frequency=_safe_sound_frequency(
                        modified_type,
                        modified_frequency
                    ),
                    duration=2,
                    volume=modified_level,
                    bandwidth=_safe_sound_bandwidth(
                        modified_bandwidth
                    )
                )

                st.audio(
                    modified_audio,
                    format="audio/wav"
                )

            except Exception as error:

                st.error(
                    "Could not generate the modified preview: "
                    f"{error}"
                )

            if st.button(
                "💾 Save Modified Configuration",
                use_container_width=True,
                key=(
                    f"save_modify_{pathway}_{trial['trial_id']}"
                )
            ):

                save_independent_personalization_review(
                    connection,
                    patient_id,
                    profile_id,
                    pathway,
                    trial,
                    status="Modified",
                    decision="Modify",
                    clinician_note=clinician_note,
                    final_stimulus_type=modified_type,
                    final_frequency_hz=modified_frequency,
                    final_bandwidth_hz=modified_bandwidth,
                    final_relative_level=modified_level
                )

                st.session_state[
                    "audiologist_review_flash"
                ] = (
                    f"{pathway_title} modified and saved. "
                    "The patient side will show Professional Status: Modified."
                )

                st.rerun()

        # ---------------------------------------------
        # CURRENT FINAL CLINICIAN CONFIGURATION
        # ---------------------------------------------

        refreshed_review = get_personalization_review(
            connection,
            pathway,
            trial[
                "trial_id"
            ]
        )

        if (
            refreshed_review is not None
            and
            refreshed_review[0] in (
                "Approved",
                "Modified"
            )
            and
            refreshed_review[2] is not None
        ):

            st.markdown(
                "#### Final clinician-reviewed configuration"
            )

            final_type = refreshed_review[2]
            final_frequency = refreshed_review[3]
            final_bandwidth = refreshed_review[4]
            final_level = refreshed_review[5]

            final_1, final_2, final_3 = st.columns(
                3
            )

            final_1.metric(
                "Final sound",
                final_type
            )

            final_2.metric(
                "Final frequency",
                (
                    "Broadband"
                    if final_type == "White Noise"
                    else (
                        f"{int(round(float(final_frequency)))} Hz"
                        if final_frequency is not None
                        else "N/A"
                    )
                )
            )

            final_3.metric(
                "Final relative level",
                (
                    f"{float(final_level):.02f}"
                    if final_level is not None
                    else "N/A"
                )
            )

            try:

                st.audio(
                    generate_stimulus(
                        stimulus_type=final_type,
                        frequency=_safe_sound_frequency(
                            final_type,
                            final_frequency
                        ),
                        duration=2,
                        volume=_safe_sound_level(
                            final_level
                        ),
                        bandwidth=_safe_sound_bandwidth(
                            final_bandwidth
                        )
                    ),
                    format="audio/wav"
                )

            except Exception:

                pass


def navigate_to_audiologist_patient_page(
    target_page,
    patient_id,
    patient_code
):
    """Navigate while preserving the selected professional patient context."""

    st.session_state[
        "audiologist_active_patient_id"
    ] = patient_id

    st.session_state[
        "audiologist_active_patient_code"
    ] = patient_code

    # Prime the known patient selectors used by existing professional pages.
    selector_keys = {
        "Care Coordination": [
            "care_coordination_patient"
        ],
        "Audiometry": [
            "audiometry_patient"
        ],
        "Acoustic Personalization": [
            "audiologist_acoustic_patient"
        ],
        "Clinical Report": [
            "clinical_report_patient"
        ],
    }

    for widget_key in selector_keys.get(
        target_page,
        []
    ):

        st.session_state[
            widget_key
        ] = patient_code

    st.session_state[
        "main_navigation"
    ] = target_page


# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="SmartTinnitus | Medico International Services",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

APP_DIR = Path(__file__).resolve().parent
MEDICO_LOGO_PATH = APP_DIR / "assets" / "medico_logo.png"



# =====================================================
# CACHE AI MODEL
# =====================================================

@st.cache_resource
def get_cached_ai_model():

    return load_model()


# =====================================================
# DESIGN
# =====================================================

st.html("""
<style>
:root {
    --medico-green: #008f5a;
    --medico-green-dark: #006f47;
    --medico-green-soft: #e9f7f1;
    --medico-red: #d8262f;
    --medico-ink: #16222a;
    --medico-muted: #6b747b;
    --medico-line: #e5e8e6;
    --medico-bg: #f7f8f6;
}

.stApp {
    background: var(--medico-bg);
}

header[data-testid="stHeader"] {
    background: rgba(255,255,255,0.97) !important;
    border-bottom: 1px solid var(--medico-line);
}

.block-container {
    max-width: 1240px !important;
    padding-top: 2.4rem !important;
    padding-bottom: 3.5rem !important;
}

h1, [data-testid="stMainBlockContainer"] h1 {
    color: var(--medico-ink) !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em;
}

h2, h3, [data-testid="stMainBlockContainer"] h2, [data-testid="stMainBlockContainer"] h3 {
    color: #163c34 !important;
    font-weight: 800 !important;
}

p, li, label, .stCaption {
    line-height: 1.55 !important;
}

/* MEDICO-STYLE SIDEBAR */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--medico-line);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.25rem;
}

section[data-testid="stSidebar"] * {
    color: var(--medico-ink) !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 0.18rem;
}

section[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 10px;
    padding: 0.48rem 0.58rem;
    margin-bottom: 0.08rem;
    transition: all .15s ease;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: var(--medico-green-soft);
}

section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #e8f6ef !important;
    border-left: 3px solid var(--medico-green);
    font-weight: 800 !important;
}

section[data-testid="stSidebar"] input[type="radio"] {
    accent-color: var(--medico-green);
}

.sidebar-brand-shell {
    text-align: center;
    padding: 0 6px 8px 6px;
}
.sidebar-brand-title {
    color: #17382f;
    font-size: 1.18rem;
    font-weight: 850;
    margin-top: 8px;
}
.sidebar-brand-subtitle {
    color: #718079;
    font-size: .76rem;
    line-height: 1.45;
    margin-top: 3px;
}
.sidebar-section-label {
    color: #008f5a;
    font-size: .68rem;
    font-weight: 850;
    letter-spacing: .10em;
    text-transform: uppercase;
    margin: 8px 0 6px 2px;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {
    border-color: var(--medico-line) !important;
}

/* BUTTONS */
.stButton > button, .stFormSubmitButton > button {
    background: var(--medico-green) !important;
    color: #ffffff !important;
    border: 1px solid var(--medico-green) !important;
    border-radius: 7px !important;
    min-height: 40px;
    font-weight: 700 !important;
    box-shadow: none !important;
}

.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: var(--medico-green-dark) !important;
    border-color: var(--medico-green-dark) !important;
    color: #ffffff !important;
}

.stDownloadButton > button {
    background: #ffffff !important;
    color: var(--medico-green-dark) !important;
    border: 1px solid var(--medico-green) !important;
    border-radius: 7px !important;
    font-weight: 700 !important;
}

/* METRICS + CARDS */
div[data-testid="stMetric"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid var(--medico-line) !important;
    border-radius: 12px !important;
    box-shadow: 0 3px 12px rgba(16, 55, 39, 0.045) !important;
}

div[data-testid="stMetric"] {
    padding: 14px 16px;
}

div[data-testid="stMetricLabel"] {
    color: var(--medico-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricValue"] {
    color: #153d33 !important;
    font-weight: 800 !important;
}

div[data-testid="stAlert"] {
    border-radius: 9px !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--medico-line);
}

hr {
    margin-top: 1.4rem !important;
    margin-bottom: 1.4rem !important;
    border-color: var(--medico-line) !important;
}

/* GLOBAL MEDICO BRAND HEADER */
.medico-topbar {
    background: #ffffff;
    border: 1px solid var(--medico-line);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 22px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.025);
}

.medico-top-kicker {
    color: var(--medico-green);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

.medico-top-title {
    color: var(--medico-ink);
    font-size: 1.18rem;
    font-weight: 800;
    margin-top: 2px;
}

.medico-top-subtitle {
    color: var(--medico-muted);
    font-size: 0.82rem;
    margin-top: 2px;
}

.hero-box {
    background: linear-gradient(100deg, #eef8f3 0%, #ffffff 62%);
    border: 1px solid #dcece5;
    border-left: 5px solid var(--medico-green);
    border-radius: 14px;
    padding: 34px 38px;
    margin-bottom: 28px;
    box-shadow: 0 5px 18px rgba(20,70,50,0.05);
}

.hero-title {
    color: var(--medico-ink);
    font-size: 40px;
    font-weight: 800;
}

.hero-text {
    color: #59645f;
    font-size: 17px;
    line-height: 1.6;
    max-width: 820px;
}

.feature-card, .ui-panel, .patient-header, .workspace-status {
    background: #ffffff;
    border: 1px solid var(--medico-line);
    border-radius: 12px;
    box-shadow: 0 3px 12px rgba(16,55,39,0.04);
}

.feature-card { padding: 22px; margin-bottom: 15px; }
.feature-title, .workspace-section-title, .workspace-card-title, .workspace-status-title { color: #163c34; }
.feature-text, .workspace-section-text, .workspace-card-text, .workspace-status-text, .ui-subtitle { color: var(--medico-muted); }
.workspace-eyebrow, .ui-kicker { color: var(--medico-green); }

.medico-section-kicker {
    color: var(--medico-green);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* FOOTER */
.medico-footer {
    margin-top: 46px;
    padding: 26px 10px 14px 10px;
    border-top: 1px solid var(--medico-line);
    color: #343b38;
    font-size: 0.86rem;
}

.medico-footer-muted { color: #727a76; }
.medico-footer-contact { line-height: 1.75; }



/* ==========================================================
   MEDICO WEBSITE-LIKE APPLICATION SHELL
   ========================================================== */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

.block-container {
    max-width: 1320px !important;
    padding-top: 1.0rem !important;
}

.medico-app-header {
    background: #ffffff;
    border-bottom: 1px solid #e4e9e6;
    padding: 6px 0 14px 0;
    margin-bottom: 0;
}
.medico-app-brand {
    color: #18392f;
    font-size: 1.18rem;
    font-weight: 850;
    line-height: 1.15;
}
.medico-app-subbrand {
    color: #7a8580;
    font-size: .78rem;
    margin-top: 3px;
}
.medico-user-chip {
    background: #f2f8f5;
    border: 1px solid #dce9e2;
    color: #176348;
    border-radius: 999px;
    padding: 8px 13px;
    font-size: .78rem;
    font-weight: 800;
    text-align: center;
}

.medico-nav-shell {
    background: #ffffff;
    border-top: 1px solid #f0f2f1;
    border-bottom: 1px solid #e2e8e4;
    margin: 0 -1rem 24px -1rem;
    padding: 8px 1rem 5px 1rem;
    box-shadow: 0 3px 10px rgba(28,63,49,.025);
}
.medico-nav-label {
    color: #7a8580;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .09em;
    text-transform: uppercase;
    margin-bottom: 2px;
}

/* Make the top navigation read more like a website menu. */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] [role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: .25rem .45rem !important;
    align-items: center !important;
}
div[data-testid="stRadio"] [role="radiogroup"] > label {
    background: transparent;
    border-radius: 8px;
    padding: .35rem .52rem !important;
    margin: 0 !important;
}
div[data-testid="stRadio"] [role="radiogroup"] > label:hover {
    background: #eef7f2;
}

.medico-page-shell {
    background: #ffffff;
    border: 1px solid #e5e9e7;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 20px;
}
</style>
""")


# =====================================================
# DATABASE
# =====================================================

initialize_database()

initialize_auth_state()

if not st.session_state[
    "authenticated"
]:

    show_login_page()

    st.stop()


conn = get_connection()
cursor = conn.cursor()

# New clinician-coordinated care workflow.
ensure_care_coordination_table(
    conn
)


# =====================================================
# PATIENT ACCOUNT LINK CHECK
# =====================================================

if (
    st.session_state[
        "auth_role"
    ]
    ==
    "patient"
    and
    st.session_state[
        "auth_patient_id"
    ]
    is None
):

    st.error(
        "This patient login is not linked to a patient record. "
        "Please ask the audiologist to correct the account."
    )

    st.stop()


# =====================================================
# MEDICO WEBSITE-STYLE HEADER + TOP NAVIGATION
# =====================================================

current_role = st.session_state["auth_role"]
current_display_name = (
    st.session_state.get("auth_display_name")
    or st.session_state.get("auth_username")
    or "User"
)

if current_role == "audiologist":
    navigation_pages = [
        "Home",
        "Patients",
        "Acoustic Personalization",
        "Care Coordination",
        "Audiometry",
        "Patient Acoustic Digital Twin",
        "Clinical Report",
        "SmartTinnitus Assistant",
    ]
else:
    navigation_pages = [
        "Home",
        "Tinnitus Assessment",
        "My Tinnitus Profile",
        "Appointments & Care Requests",
        "Assigned Acoustic Test",
        "SmartTinnitus Assistant",
    ]

if st.session_state.get("main_navigation") not in navigation_pages:
    st.session_state["main_navigation"] = "Home"

# Top brand row: logo + product identity + logged-in user.
# Company contact details are intentionally kept on the public login page only.
brand_logo_col, brand_text_col, brand_user_col = st.columns(
    [1.15, 5.15, 1.55],
    vertical_alignment="center"
)

with brand_logo_col:
    if MEDICO_LOGO_PATH.exists():
        st.image(str(MEDICO_LOGO_PATH), width=165)
    else:
        st.markdown("**Medico International Services**")

with brand_text_col:
    st.markdown(
        "<div class='medico-app-header'>"
        "<div class='medico-top-kicker'>MEDICO INTERNATIONAL SERVICES</div>"
        "<div class='medico-app-brand'>SmartTinnitus</div>"
        "<div class='medico-app-subbrand'>Clinician-supervised digital audiology platform</div>"
        "</div>",
        unsafe_allow_html=True,
    )

with brand_user_col:
    portal_name = "Audiologist Portal" if current_role == "audiologist" else "Patient Portal"
    st.markdown(
        f"<div class='medico-user-chip'>{portal_name}<br>"
        f"<span style='font-weight:600;color:#6f7c76'>{current_display_name}</span></div>",
        unsafe_allow_html=True,
    )
    if st.button("Log out", use_container_width=True, key="top_logout"):
        from login import logout_user
        logout_user()
        st.rerun()

# Website-like navigation directly under the header.
st.markdown("<div class='medico-nav-label'>Navigation</div>", unsafe_allow_html=True)
page = st.radio(
    "Navigation",
    navigation_pages,
    key="main_navigation",
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# -----------------------------------------------------
# GLOBAL AUDIOLOGIST PATIENT CONTEXT
# -----------------------------------------------------
# The audiologist opens a patient once from Patients. Professional pages then
# reuse that same patient automatically. No duplicate selector is shown here.
_PATIENT_CONTEXT_PAGES = {
    "Audiometry",
    "Care Coordination",
    "Acoustic Personalization",
    "Patient Acoustic Digital Twin",
    "Clinical Report",
}

if current_role == "audiologist" and page in _PATIENT_CONTEXT_PAGES:
    _active_patient_id = get_active_patient_id()
    _active_patient_code = get_active_patient_code()

    if _active_patient_id is None or _active_patient_code is None:
        st.warning(
            "No patient is currently open. Go to Patients and click Open Patient first."
        )
        st.button(
            "Go to Patients",
            use_container_width=True,
            on_click=navigate_to_page,
            args=("Patients",),
            key=f"go_to_patients_from_{page}",
        )
        st.stop()

    st.markdown(
        f"<div class='ui-panel' style='padding:10px 14px;margin-bottom:14px'>"
        f"<b>Active patient:</b> {_active_patient_code}"
        f"</div>",
        unsafe_allow_html=True,
    )


# =====================================================
# PATIENT PAGE-ENTRY STATE
# =====================================================

_previous_main_page = st.session_state.get(
    "_smarttinnitus_previous_main_page"
)

if (
    False
    and
    st.session_state.get("auth_role") == "patient"
    and
    page == "Acoustic Personalization"
    and
    _previous_main_page != "Acoustic Personalization"
):
    # Always open patient Acoustic Personalization on the pathway-choice screen.
    st.session_state[
        "patient_personalization_path"
    ] = None

if (
    st.session_state.get("auth_role") == "audiologist"
    and
    page == "Acoustic Personalization"
    and
    _previous_main_page != "Acoustic Personalization"
):
    # Start the professional workspace from the first technical view each time
    # the audiologist enters the page.
    st.session_state[
        "audiologist_acoustic_mode"
    ] = "🎛️ Candidate & Assignment"

st.session_state[
    "_smarttinnitus_previous_main_page"
] = page


# =====================================================
# HARD ROLE-BASED PAGE ACCESS GUARD
# =====================================================

patient_allowed_pages = {
    "Home",
    "Tinnitus Assessment",
    "My Tinnitus Profile",
    "Appointments & Care Requests",
    "Assigned Acoustic Test",
    "SmartTinnitus Assistant"
}

audiologist_allowed_pages = {
    "Home",
    "Patients",
    "Acoustic Personalization",
    "Care Coordination",
    "Audiometry",
    "Patient Acoustic Digital Twin",
    "Clinical Report",
    "SmartTinnitus Assistant"
}

current_role = st.session_state[
    "auth_role"
]

allowed_pages = (
    patient_allowed_pages
    if current_role == "patient"
    else audiologist_allowed_pages
)

if page not in allowed_pages:

    st.error(
        "You do not have permission to access this page."
    )

    st.stop()


# =====================================================
# HOME
# =====================================================

if page == "Home":

    current_role = (
        st.session_state[
            "auth_role"
        ]
    )

    current_display_name = (
        st.session_state.get(
            "auth_display_name"
        )
        or
        st.session_state.get(
            "auth_username"
        )
        or
        "User"
    )

    # =================================================
    # PATIENT HOME
    # =================================================

    if (
        current_role
        ==
        "patient"
    ):

        linked_patient_id = (
            st.session_state[
                "auth_patient_id"
            ]
        )

        cursor.execute(
            """
            SELECT
                patient_code,
                age,
                sex
            FROM patients
            WHERE id = ?
            LIMIT 1
            """,
            (
                linked_patient_id,
            )
        )

        patient_row = (
            cursor.fetchone()
        )

        if patient_row is not None:

            (
                patient_code,
                patient_age,
                patient_sex
            ) = patient_row

        else:

            patient_code = (
                current_display_name
            )

            patient_age = None
            patient_sex = None

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM sessions
            WHERE patient_id = ?
            """,
            (
                linked_patient_id,
            )
        )

        session_count = int(
            cursor.fetchone()[
                0
            ]
        )

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM stimulus_trials
            WHERE patient_id = ?
            """,
            (
                linked_patient_id,
            )
        )

        trial_count = int(
            cursor.fetchone()[
                0
            ]
        )

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM audiologist_decisions ad
            WHERE ad.patient_id = ?
              AND ad.decision IN ('Approve', 'Modify')
              AND NOT EXISTS (
                    SELECT 1
                    FROM stimulus_trials st
                    WHERE st.decision_id = ad.id
              )
            """,
            (
                linked_patient_id,
            )
        )

        pending_approved_count = int(
            cursor.fetchone()[
                0
            ]
        )

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM tinnitus_profiles
            WHERE patient_id = ?
            """,
            (
                linked_patient_id,
            )
        )

        tinnitus_profile_count = int(
            cursor.fetchone()[0]
        )

        assessment_button_label = (
            "Start My First Assessment"
            if tinnitus_profile_count == 0
            else "Open Tinnitus Assessment"
        )

        st.html(
            f"""
            <div class="hero-box">
                <div class="hero-title">
                    🎧 Welcome to SmartTinnitus
                </div>

                <div class="hero-text">
                    Your personalized workspace for tinnitus assessment,
                    communication with your audiologist, care coordination
                    and assigned acoustic testing.
                </div>
            </div>
            """
        )

        st.subheader(
            f"Welcome, {current_display_name}"
        )

        st.caption(
            f"Linked patient record: {patient_code}"
        )

        m1, m2, m3 = st.columns(
            3
        )

        m1.metric(
            "Completed Sessions",
            session_count
        )

        m2.metric(
            "Supervised Acoustic Trials",
            trial_count
        )

        home_care_summary = get_patient_care_summary(
            linked_patient_id
        )

        m3.metric(
            "Care Actions",
            home_care_summary["action_required"]
        )

        st.markdown("---")

        st.html(
            """
            <div class="ui-kicker">Personal workspace</div>
            <div class="workspace-section-title">Your SmartTinnitus Journey</div>
            <div class="ui-subtitle">
                Follow your assessment, care requests, tinnitus profile,
                assigned acoustic tests and assistant in one place.
            </div>
            """
        )

        st.write(
            "Use the steps below in order. Each card takes you "
            "directly to the corresponding part of your workflow."
        )

        c1, c2 = st.columns(
            2
        )

        with c1:

            st.markdown(
                "### 🎯 1. Tinnitus Assessment"
            )

            st.caption(
                "Describe your tinnitus and complete the "
                "psychoacoustic pitch and loudness assessment."
            )

            st.button(
                assessment_button_label,
                use_container_width=True,
                on_click=navigate_to_page,
                args=(
                    "Tinnitus Assessment",
                ),
                key="patient_home_assessment"
            )

        with c2:

            st.markdown(
                "### 📅 2. Appointments & Care Requests"
            )

            st.caption(
                "Request an in-person appointment, video call or phone call, and respond "
                "to requests from your audiologist."
            )

            st.button(
                "Open Appointments & Care",
                use_container_width=True,
                on_click=navigate_to_page,
                args=(
                    "Appointments & Care Requests",
                ),
                key="patient_home_care"
            )

        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("### 🎧 3. My Tinnitus Profile")
            st.caption("View your latest tinnitus characteristics, matched pitch, relative loudness and impact scores.")
            st.button(
                "Open My Tinnitus Profile",
                use_container_width=True,
                on_click=navigate_to_page,
                args=("My Tinnitus Profile",),
                key="patient_home_tinnitus_profile"
            )

        with c2:
            st.markdown("### 🔊 4. Assigned Acoustic Test")
            st.caption("Listen to the sound assigned by your audiologist and submit your feedback.")
            st.button(
                "Open Assigned Acoustic Test",
                use_container_width=True,
                on_click=navigate_to_page,
                args=("Assigned Acoustic Test",),
                key="patient_home_assigned_acoustic_test"
            )

        with c3:
            st.markdown("### 💬 5. SmartTinnitus Assistant")
            st.caption("Ask questions about information already available in your SmartTinnitus record.")
            st.button(
                "Open SmartTinnitus Assistant",
                use_container_width=True,
                on_click=navigate_to_page,
                args=("SmartTinnitus Assistant",),
                key="patient_home_chatbot"
            )

        st.markdown("---")

        care_summary = home_care_summary

        if care_summary["action_required"] > 0:

            st.warning(
                "Your audiologist has requested information, confirmation, audiometry, "
                "a consultation or another professional next step."
            )

            st.button(
                "Review Care Request",
                use_container_width=True,
                on_click=navigate_to_page,
                args=(
                    "Appointments & Care Requests",
                ),
                key="patient_home_care_action"
            )

        elif care_summary["scheduled"] > 0:

            st.success(
                "You have a scheduled care activity with your audiologist."
            )


    # =================================================
    # AUDIOLOGIST HOME
    # =================================================

    else:

        cursor.execute("SELECT COUNT(*) FROM patients")
        patient_count = int(cursor.fetchone()[0])

        cursor.execute("SELECT COUNT(*) FROM stimulus_trials")
        total_trials = int(cursor.fetchone()[0])

        active_code = get_active_patient_code()

        st.html(
            f"""
            <div class="audio-home-shell">
                <div class="audio-home-kicker">Audiologist workspace</div>
                <div class="audio-home-title">Welcome back, {current_display_name}</div>
                <div class="audio-home-subtitle">
                    Open a patient once, then continue through personalization, care coordination,
                    audiometry, the patient acoustic Digital Twin, reporting and the assistant.
                </div>
            </div>
            """
        )

        st.write("")

        m1, m2, m3 = st.columns(3, gap="medium")
        m1.metric("Patients", patient_count)
        m2.metric("Recorded Acoustic Trials", total_trials)
        m3.metric("Active Patient", active_code or "None")

        st.write("")
        st.markdown("### Main workflow")

        c1, c2, c3 = st.columns(3, gap="large")

        with c1:
            with st.container(border=True):
                st.markdown("### 👤 Patients")
                st.caption("Create a patient, review the patient directory and open one patient record.")
                st.button(
                    "Open Patients",
                    use_container_width=True,
                    on_click=navigate_to_page,
                    args=("Patients",),
                    key="audio_home_patients"
                )

        with c2:
            with st.container(border=True):
                st.markdown("### 🔊 Acoustic Personalization")
                st.caption("Configure a sound manually or use AI-assisted generation, then send the approved candidate to the patient.")
                st.button(
                    "Open Acoustic Personalization",
                    type="primary",
                    use_container_width=True,
                    on_click=navigate_to_page,
                    args=("Acoustic Personalization",),
                    key="audio_home_personalization"
                )

        with c3:
            with st.container(border=True):
                st.markdown("### 🤝 Care Coordination")
                st.caption("Coordinate appointments, requests, referrals and supervised clinical follow-up actions.")
                st.button(
                    "Open Care Coordination",
                    use_container_width=True,
                    on_click=navigate_to_page,
                    args=("Care Coordination",),
                    key="audio_home_care"
                )

        c4, c5, c6 = st.columns(3, gap="large")

        with c4:
            with st.container(border=True):
                st.markdown("### 🎧 Audiometry")
                st.caption("Record and review externally measured hearing thresholds for the active patient.")
                st.button(
                    "Open Audiometry",
                    use_container_width=True,
                    on_click=navigate_to_page,
                    args=("Audiometry",),
                    key="audio_home_audiometry"
                )

        with c5:
            with st.container(border=True):
                st.markdown("### 🧠 Patient Acoustic Digital Twin")
                st.caption("Review the response map, observed acoustic evidence and evidence evolution for the active patient.")
                st.button(
                    "Open Patient Acoustic Digital Twin",
                    use_container_width=True,
                    on_click=navigate_to_page,
                    args=("Patient Acoustic Digital Twin",),
                    key="audio_home_digital_twin"
                )

        with c6:
            with st.container(border=True):
                st.markdown("### 📄 Clinical Report")
                st.caption("Generate the clinical summary for the active patient.")
                st.button(
                    "Open Clinical Report",
                    use_container_width=True,
                    on_click=navigate_to_page,
                    args=("Clinical Report",),
                    key="audio_home_report"
                )

        with st.container(border=True):
            st.markdown("### 💬 SmartTinnitus Assistant")
            st.caption("Ask questions about the stored SmartTinnitus record and workflow.")
            st.button(
                "Open SmartTinnitus Assistant",
                use_container_width=True,
                on_click=navigate_to_page,
                args=("SmartTinnitus Assistant",),
                key="audio_home_chatbot"
            )


# =====================================================
# PATIENTS
# =====================================================

elif page == "Patients":

    st.title(
        "Patients"
    )


    overview_tab, registration_tab = st.tabs(
        [
            "📋 Patient Overview & Review",
            "➕ Create Patient"
        ]
    )

    # =================================================
    # CREATE PATIENT + LOGIN
    # =================================================

    with registration_tab:

        st.subheader(
            "Create New Patient"
        )

        st.caption(
            "Register a patient and automatically generate their "
            "SmartTinnitus patient login."
        )

        c1, c2 = st.columns(
            2
        )

        with c1:

            patient_code = st.text_input(
                "Patient ID",
                placeholder="P001",
                key="new_patient_code"
            )

            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                step=1,
                key="new_patient_age"
            )

        with c2:

            sex = st.selectbox(
                "Sex",
                [
                    "Select",
                    "Female",
                    "Male",
                    "Other"
                ],
                key="new_patient_sex"
            )

        st.caption(
            "A patient login will be generated automatically."
        )

        if st.button(
            "Create Patient & Generate Login",
            use_container_width=True,
            key="create_patient_and_login"
        ):

            if not patient_code.strip():

                st.error(
                    "Enter a Patient ID."
                )

            elif sex == "Select":

                st.error(
                    "Select sex."
                )

            else:

                try:

                    cursor.execute(
                        """
                        INSERT INTO patients
                        (
                            patient_code,
                            age,
                            sex,
                            created_at
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            patient_code.strip(),
                            age,
                            sex,
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        )
                    )

                    new_patient_id = (
                        cursor.lastrowid
                    )

                    conn.commit()

                    (
                        account_success,
                        account_message,
                        generated_username,
                        generated_password
                    ) = create_patient_login(
                        patient_id=new_patient_id,
                        patient_code=patient_code.strip()
                    )

                    if account_success:

                        st.session_state[
                            "new_patient_credentials"
                        ] = {
                            "patient_id": new_patient_id,
                            "patient_code": patient_code.strip(),
                            "username": generated_username,
                            "password": generated_password
                        }

                        st.success(
                            "Patient and login account created successfully."
                        )

                    else:

                        cursor.execute(
                            """
                            DELETE FROM patients
                            WHERE id = ?
                            """,
                            (
                                new_patient_id,
                            )
                        )

                        conn.commit()

                        st.error(
                            "The patient account could not be created."
                        )

                        st.error(
                            account_message
                        )

                except sqlite3.IntegrityError:

                    st.error(
                        "Patient ID already exists."
                    )

        new_credentials = (
            st.session_state.get(
                "new_patient_credentials"
            )
        )

        if new_credentials:

            st.markdown("---")

            st.markdown(
                "### 🔐 Patient Login Credentials"
            )

            st.warning(
                "Give these credentials to the patient securely."
            )

            credential_col1, credential_col2 = (
                st.columns(2)
            )

            with credential_col1:

                st.text_input(
                    "Username",
                    value=new_credentials[
                        "username"
                    ],
                    disabled=True,
                    key="generated_patient_username"
                )

            with credential_col2:

                st.text_input(
                    "Temporary Password",
                    value=new_credentials[
                        "password"
                    ],
                    disabled=True,
                    key="generated_patient_password"
                )

            st.code(
                "Username: "
                f"{new_credentials['username']}\n"
                "Temporary password: "
                f"{new_credentials['password']}",
                language=None
            )

            if st.button(
                "Clear displayed credentials",
                key="clear_new_patient_credentials"
            ):

                del st.session_state[
                    "new_patient_credentials"
                ]

                st.rerun()


    # =================================================
    # PATIENT OVERVIEW + PROFESSIONAL REVIEW
    # =================================================

    with overview_tab:

        # Patient identity is now supplied by the backend service layer.
        # Keep the tuple shape temporarily so the existing detailed patient
        # record below can be migrated without changing behavior.
        patient_rows = [
            (
                item["id"],
                item["patient_code"],
                item["age"],
                item["sex"],
                item["created_at"],
            )
            for item in backend_list_patients()
        ]

        if not patient_rows:

            st.warning(
                "No patients are registered yet."
            )

        else:

            ensure_personalization_review_table(
                conn
            )

            ensure_patient_followup_progress_columns(
                conn
            )

            # -----------------------------------------
            # PATIENT DIRECTORY — BACKEND-SUPPLIED
            # -----------------------------------------

            st.markdown("## Patient Directory")

            all_patient_summaries = backend_list_patient_summaries()

            patient_search = st.text_input(
                "Search patients",
                placeholder="Search by patient code...",
                key="patients_directory_search",
            ).strip().lower()

            filtered_summaries = [
                summary
                for summary in all_patient_summaries
                if not patient_search
                or patient_search in str(summary["patient_code"]).lower()
            ]

            if not filtered_summaries:
                st.info("No patient matches this search.")

            for summary in filtered_summaries:
                with st.container(border=True):
                    c1, c2, c3, c4, c5, c6 = st.columns(
                        [1.2, 0.7, 0.8, 1.5, 1.4, 1.0],
                        vertical_alignment="center",
                    )

                    c1.markdown(f"**{summary['patient_code']}**")
                    c2.caption(f"Age\n\n**{summary['age'] if summary['age'] is not None else '—'}**")
                    c3.caption(f"Sex\n\n**{summary['sex'] or '—'}**")
                    c4.caption(
                        f"Assessment\n\n**{summary['assessment_status']}**"
                    )
                    c5.caption(
                        "Audiogram\n\n**"
                        + ("Available" if summary["audiogram_available"] else "Not available")
                        + "**"
                    )

                    if c6.button(
                        "Open Patient",
                        key=f"open_patient_{summary['id']}",
                        use_container_width=True,
                    ):
                        set_active_patient(
                            summary["id"],
                            summary["patient_code"],
                        )
                        st.rerun()


            st.markdown("---")

            # -----------------------------------------
            # OPENED PATIENT RECORD
            # -----------------------------------------

            patient_options = {
                code: {
                    "id": patient_id,
                    "age": age,
                    "sex": sex,
                    "created_at": created_at,
                }
                for (patient_id, code, age, sex, created_at) in patient_rows
            }

            selected_patient_code = get_active_patient_code()

            if selected_patient_code not in patient_options:
                st.info(
                    "Open a patient from the directory above to view the complete patient record."
                )
                st.stop()

            selected_patient = patient_options[selected_patient_code]
            patient_id = selected_patient["id"]

            st.markdown(
                f"### Open Patient Record · {selected_patient_code}"
            )

            # -----------------------------------------
            # IDENTITY + ACCOUNT
            # -----------------------------------------

            st.markdown(
                f"## Patient {selected_patient_code}"
            )

            account_row = conn.execute(
                """
                SELECT
                    username,
                    is_active
                FROM users
                WHERE patient_id = ?
                  AND role = 'patient'
                ORDER BY id DESC
                LIMIT 1
                """,
                (
                    patient_id,
                )
            ).fetchone()

            id1, id2, id3, id4 = st.columns(
                4
            )

            id1.metric(
                "Patient Code",
                selected_patient_code
            )

            id2.metric(
                "Age",
                selected_patient[
                    "age"
                ]
            )

            id3.metric(
                "Sex",
                selected_patient[
                    "sex"
                ]
            )

            id4.metric(
                "Patient Account",
                (
                    "Active"
                    if (
                        account_row is not None
                        and
                        bool(
                            account_row[1]
                        )
                    )
                    else "Inactive / Not linked"
                )
            )

            if account_row is not None:

                st.caption(
                    f"Linked username: {account_row[0]}"
                )

            flash_message = st.session_state.pop(
                "audiologist_review_flash",
                None
            )

            if flash_message:

                st.success(
                    flash_message
                )

            # -----------------------------------------
            # AUDIOMETRY SUMMARY
            # -----------------------------------------

            st.markdown("---")
            st.markdown(
                "## Audiometry Summary"
            )

            selected_audiogram_summary = conn.execute(
                """
                SELECT
                    COUNT(*) AS threshold_rows,
                    COUNT(DISTINCT created_at) AS saved_audiogram_sets,
                    MAX(created_at) AS latest_audiogram
                FROM audiograms
                WHERE patient_id = ?
                """,
                (
                    patient_id,
                )
            ).fetchone()

            selected_has_audiogram = (
                selected_audiogram_summary is not None
                and
                int(
                    selected_audiogram_summary[0]
                    or
                    0
                ) > 0
            )

            selected_audiogram_count = (
                int(
                    selected_audiogram_summary[1]
                    or
                    0
                )
                if selected_audiogram_summary is not None
                else 0
            )

            selected_latest_audiogram = (
                str(
                    selected_audiogram_summary[2]
                )[:10]
                if (
                    selected_has_audiogram
                    and
                    selected_audiogram_summary[2] is not None
                )
                else "—"
            )

            audio_1, audio_2, audio_3 = st.columns(
                3
            )

            audio_1.metric(
                "Audiogram Status",
                (
                    "✅ Available"
                    if selected_has_audiogram
                    else "⚪ Not available"
                )
            )

            audio_2.metric(
                "Latest Audiogram",
                selected_latest_audiogram
            )

            audio_3.metric(
                "Saved Audiograms",
                selected_audiogram_count
            )

            if selected_has_audiogram:

                st.caption(
                    "Audiometric threshold data are available for this patient. "
                    "Open Audiometry to review or plot the full right- and left-ear audiogram."
                )

            else:

                st.info(
                    "No audiogram has been saved for this patient yet."
                )

            # -----------------------------------------
            # LATEST TINNITUS PROFILE
            # -----------------------------------------

            st.markdown("---")
            st.markdown(
                "## Latest Tinnitus Profile"
            )

            latest_profile = conn.execute(
                """
                SELECT
                    id,
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
                """,
                (
                    patient_id,
                )
            ).fetchone()

            profile_id = (
                latest_profile[0]
                if latest_profile is not None
                else None
            )

            if latest_profile is None:

                st.info(
                    "No tinnitus profile has been recorded yet."
                )

            else:

                (
                    profile_id,
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
                    profile_created_at
                ) = latest_profile

                p1, p2, p3, p4 = st.columns(
                    4
                )

                p1.metric(
                    "Location",
                    laterality
                    or
                    "N/A"
                )

                p2.metric(
                    "Sound Type",
                    tinnitus_type
                    or
                    "N/A"
                )

                p3.metric(
                    "Pattern",
                    temporal_pattern
                    or
                    "N/A"
                )

                p4.metric(
                    "Duration",
                    (
                        f"{duration_months} months"
                        if duration_months is not None
                        else "N/A"
                    )
                )

                p5, p6, p7 = st.columns(
                    3
                )

                p5.metric(
                    "Matched Pitch",
                    (
                        f"{int(round(float(estimated_pitch_hz)))} Hz"
                        if estimated_pitch_hz is not None
                        else "Not completed"
                    )
                )

                p6.metric(
                    "Relative Loudness",
                    (
                        f"{float(loudness_relative):.02f}"
                        if loudness_relative is not None
                        else "Not completed"
                    )
                )

                p7.metric(
                    "Assessment Date",
                    profile_created_at
                    or
                    "N/A"
                )

                impact_df = pd.DataFrame(
                    [
                        {
                            "Measure": "Annoyance",
                            "Score": annoyance_score
                        },
                        {
                            "Measure": "Sleep Impact",
                            "Score": sleep_score
                        },
                        {
                            "Measure": "Concentration Impact",
                            "Score": concentration_score
                        },
                        {
                            "Measure": "Stress",
                            "Score": stress_score
                        },
                    ]
                )

                st.dataframe(
                    impact_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Pitch and loudness values are subjective profiling values. "
                    "Relative loudness is not a calibrated dB measurement."
                )

            # -----------------------------------------
            # CARE COORDINATION SUMMARY
            # -----------------------------------------

            st.markdown("---")
            st.markdown(
                "## Care Coordination"
            )

            selected_care_summary = get_patient_care_summary(
                patient_id
            )

            care_1, care_2, care_3 = st.columns(
                3
            )

            care_1.metric(
                "Open Requests",
                selected_care_summary["pending"]
            )

            care_2.metric(
                "Scheduled",
                selected_care_summary["scheduled"]
            )

            care_3.metric(
                "Action Required",
                selected_care_summary["action_required"]
            )

            st.caption(
                "Open the dedicated Care Coordination workspace to request information, "
                "schedule consultations or audiometry, document referrals, and arrange supervised acoustic testing."
            )

            st.button(
                "Open Care Coordination",
                use_container_width=True,
                on_click=navigate_to_audiologist_patient_page,
                args=(
                    "Care Coordination",
                    patient_id,
                    selected_patient_code
                ),
                key=f"patients_open_care_{patient_id}"
            )

            # -----------------------------------------
            # ACOUSTIC PERSONALIZATION STATUS
            # -----------------------------------------

            st.markdown("---")
            st.markdown(
                "## Audiologist-Supervised Acoustic Testing Status"
            )

            st.write(
                "Current audiologist-manual and AI-assisted supervised results are summarized here. "
                "The audiologist remains responsible for the configuration tested by the patient."
            )

            standard_trial = get_latest_personalization_trial(
                conn,
                patient_id,
                "standard"
            )

            ai_trial = get_latest_personalization_trial(
                conn,
                patient_id,
                "ai"
            )

            standard_col, ai_col = st.columns(
                2
            )

            with standard_col:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "### 🎧 Audiologist Manual"
                    )

                    if standard_trial is None:

                        st.info(
                            "No completed audiologist-manual result yet."
                        )

                    else:

                        standard_review = get_personalization_review(
                            conn,
                            "standard",
                            standard_trial[
                                "trial_id"
                            ]
                        )

                        standard_status = (
                            standard_review[0]
                            if standard_review is not None
                            else "Pending Review"
                        )

                        st.metric(
                            "Current Status",
                            standard_status
                        )

                        s1, s2 = st.columns(
                            2
                        )

                        s1.metric(
                            "Candidate",
                            standard_trial[
                                "stimulus_type"
                            ]
                            or
                            "Not recorded"
                        )

                        s2.metric(
                            "Observed Response",
                            (
                                f"{float(standard_trial['observed_score']):.1f} / 100"
                                if standard_trial[
                                    "observed_score"
                                ] is not None
                                else "N/A"
                            )
                        )

            with ai_col:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "### 🤖 AI-Assisted Recommendation"
                    )

                    if ai_trial is None:

                        st.info(
                            "No completed supervised AI-assisted result yet."
                        )

                    else:

                        ai_review = get_personalization_review(
                            conn,
                            "ai",
                            ai_trial[
                                "trial_id"
                            ]
                        )

                        ai_status = (
                            ai_review[0]
                            if ai_review is not None
                            else "Pending Review"
                        )

                        st.metric(
                            "Current Status",
                            ai_status
                        )

                        a1, a2 = st.columns(
                            2
                        )

                        a1.metric(
                            "Candidate",
                            ai_trial[
                                "stimulus_type"
                            ]
                            or
                            "Not recorded"
                        )

                        a2.metric(
                            "Observed Response",
                            (
                                f"{float(ai_trial['observed_score']):.1f} / 100"
                                if ai_trial[
                                    "observed_score"
                                ] is not None
                                else "N/A"
                            )
                        )

                        if ai_trial[
                            "predicted_score"
                        ] is not None:

                            st.caption(
                                "AI predicted response: "
                                f"{float(ai_trial['predicted_score']):.1f} / 100"
                            )

            st.caption(
                "Manual and AI-assisted records remain traceable independently. "
                "Open Acoustic Personalization to select the recommendation method, supervise the patient test, "
                "and Approve, Modify, or Reject the resulting configuration."
            )

            # -----------------------------------------
            # FOLLOW-UP SUMMARY
            # -----------------------------------------

            st.markdown("---")
            st.markdown(
                "## Latest Follow-up Summary"
            )

            total_followups = conn.execute(
                """
                SELECT COUNT(*)
                FROM sessions
                WHERE patient_id = ?
                """,
                (
                    patient_id,
                )
            ).fetchone()[0]

            latest_followup = conn.execute(
                """
                SELECT
                    session_date,
                    annoyance_after,
                    sleep_impact,
                    concentration_impact,
                    stress_score,
                    tinnitus_noticeability_after,
                    change_since_last
                FROM sessions
                WHERE patient_id = ?
                ORDER BY session_date DESC, id DESC
                LIMIT 1
                """,
                (
                    patient_id,
                )
            ).fetchone()

            f1, f2 = st.columns(
                2
            )

            f1.metric(
                "Total Follow-ups",
                int(
                    total_followups
                    or
                    0
                )
            )

            f2.metric(
                "Last Follow-up",
                (
                    latest_followup[0]
                    if latest_followup is not None
                    else "None yet"
                )
            )

            if latest_followup is None:

                st.caption(
                    "No follow-up information has been recorded for this patient."
                )

            else:

                (
                    followup_date,
                    followup_annoyance,
                    followup_sleep,
                    followup_concentration,
                    followup_stress,
                    followup_noticeability,
                    followup_change
                ) = latest_followup

                ff1, ff2, ff3, ff4, ff5 = st.columns(
                    5
                )

                ff1.metric(
                    "Annoyance",
                    (
                        f"{followup_annoyance}/10"
                        if followup_annoyance is not None
                        else "N/A"
                    )
                )

                ff2.metric(
                    "Sleep",
                    (
                        f"{followup_sleep}/10"
                        if followup_sleep is not None
                        else "N/A"
                    )
                )

                ff3.metric(
                    "Concentration",
                    (
                        f"{followup_concentration}/10"
                        if followup_concentration is not None
                        else "N/A"
                    )
                )

                ff4.metric(
                    "Stress",
                    (
                        f"{followup_stress}/10"
                        if followup_stress is not None
                        else "N/A"
                    )
                )

                ff5.metric(
                    "Noticeability",
                    (
                        f"{followup_noticeability}/10"
                        if followup_noticeability is not None
                        else "N/A"
                    )
                )

                st.write(
                    "**Patient comparison with previous follow-up:**",
                    followup_change
                    or
                    "Not recorded"
                )




# =====================================================
# TINNITUS ASSESSMENT
# =====================================================

elif page == "Tinnitus Assessment":

    st.title(
        "Tinnitus Assessment"
    )

    current_role = (
        st.session_state.get(
            "auth_role"
        )
    )

    # =================================================
    # PATIENT GUIDED ASSESSMENT
    # =================================================

    if current_role == "patient":

        patient_id = (
            st.session_state.get(
                "auth_patient_id"
            )
        )

        if patient_id is None:

            st.error(
                "This patient account is not linked to a patient record."
            )

            st.stop()

        if (
            "patient_assessment_step"
            not in st.session_state
        ):

            st.session_state[
                "patient_assessment_step"
            ] = 1

        if (
            "patient_assessment_profile_id"
            not in st.session_state
        ):

            st.session_state[
                "patient_assessment_profile_id"
            ] = None

        if (
            "guided_loudness_level"
            not in st.session_state
        ):

            st.session_state[
                "guided_loudness_level"
            ] = 0.03

        if (
            "guided_safety_level"
            not in st.session_state
        ):

            st.session_state[
                "guided_safety_level"
            ] = 0.03

        if (
            "guided_safety_confirmed"
            not in st.session_state
        ):

            st.session_state[
                "guided_safety_confirmed"
            ] = False

        assessment_step = int(
            st.session_state[
                "patient_assessment_step"
            ]
        )

        step_names = {
            1: "About Your Tinnitus",
            2: "Audio Safety Check",
            3: "Pitch Matching",
            4: "Loudness Matching",
            5: "Assessment Complete"
        }

        st.progress(
            assessment_step / 5
        )

        st.caption(
            f"Step {assessment_step} of 5 · "
            f"{step_names[assessment_step]}"
        )

        st.markdown("---")

        # =============================================
        # STEP 1 — QUESTIONNAIRE
        # =============================================

        if assessment_step == 1:

            st.subheader(
                "1. About Your Tinnitus"
            )

            st.write(
                "Describe what you hear and how it affects you. "
                "These answers create your tinnitus profile."
            )

            c1, c2 = st.columns(
                2
            )

            with c1:

                laterality = st.radio(
                    "Where do you perceive your tinnitus?",
                    [
                        "Left ear",
                        "Right ear",
                        "Both ears",
                        "Inside the head",
                        "Not sure"
                    ],
                    key="guided_laterality"
                )

                temporal_pattern = st.radio(
                    "Is it continuous or intermittent?",
                    [
                        "Continuous",
                        "Intermittent"
                    ],
                    horizontal=True,
                    key="guided_temporal_pattern"
                )

            with c2:

                tinnitus_type = st.selectbox(
                    "What does it sound like?",
                    [
                        "Tonal",
                        "Buzzing",
                        "Hissing",
                        "Clicking",
                        "Mixed",
                        "Other"
                    ],
                    key="guided_tinnitus_type"
                )

                duration_months = st.number_input(
                    "How long have you had it? (months)",
                    min_value=0,
                    max_value=1000,
                    step=1,
                    key="guided_duration_months"
                )

            st.markdown(
                "### How much does it affect you?"
            )

            st.caption(
                "0 = no impact · 10 = maximum impact"
            )

            c1, c2 = st.columns(
                2
            )

            with c1:

                annoyance = st.slider(
                    "Annoyance",
                    0,
                    10,
                    0,
                    key="guided_annoyance"
                )

                sleep = st.slider(
                    "Sleep impact",
                    0,
                    10,
                    0,
                    key="guided_sleep"
                )

            with c2:

                concentration = st.slider(
                    "Concentration impact",
                    0,
                    10,
                    0,
                    key="guided_concentration"
                )

                stress = st.slider(
                    "Stress associated with tinnitus",
                    0,
                    10,
                    0,
                    key="guided_stress"
                )

            if st.button(
                "Save & Continue",
                use_container_width=True,
                key="guided_questionnaire_continue"
            ):

                cursor.execute(
                    """
                    INSERT INTO tinnitus_profiles
                    (
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
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patient_id,
                        laterality,
                        tinnitus_type,
                        temporal_pattern,
                        int(duration_months),
                        int(annoyance),
                        int(sleep),
                        int(concentration),
                        int(stress),
                        None,
                        None,
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    )
                )

                conn.commit()

                st.session_state[
                    "patient_assessment_profile_id"
                ] = cursor.lastrowid

                reset_pitch_search(
                    st.session_state
                )

                st.session_state[
                    "guided_loudness_level"
                ] = 0.03

                st.session_state[
                    "guided_safety_level"
                ] = 0.03

                st.session_state[
                    "guided_safety_confirmed"
                ] = False

                st.session_state[
                    "patient_assessment_step"
                ] = 2

                st.rerun()

        # =============================================
        # STEP 2 — AUDIO SAFETY
        # =============================================

        elif assessment_step == 2:

            st.subheader(
                "2. Audio Safety Check"
            )

            st.warning(
                "Audio levels are relative and uncalibrated. "
                "They are not dB SPL or dB HL."
            )

            st.write(
                "Listen to the test sound. SmartTinnitus will change the "
                "relative test level according to your response."
            )

            safety_level = float(
                st.session_state.get(
                    "guided_safety_level",
                    0.03
                )
            )

            safety_confirmed = bool(
                st.session_state.get(
                    "guided_safety_confirmed",
                    False
                )
            )

            st.markdown(
                "### Test sound"
            )

            st.metric(
                "Current relative test level",
                f"{safety_level:.2f}"
            )

            st.audio(
                generate_pure_tone(
                    frequency=1000,
                    duration=1.0,
                    volume=safety_level
                ),
                format="audio/wav"
            )

            st.caption(
                "Play the sound, then choose how it feels. "
                "If you select Too quiet or Too loud, SmartTinnitus "
                "changes the level automatically."
            )

            c1, c2, c3 = st.columns(
                3
            )

            with c1:

                if st.button(
                    "🔉 Too quiet",
                    use_container_width=True,
                    key="guided_safety_too_quiet"
                ):

                    new_level = min(
                        0.10,
                        round(
                            safety_level + 0.01,
                            2
                        )
                    )

                    st.session_state[
                        "guided_safety_level"
                    ] = new_level

                    st.session_state[
                        "guided_safety_confirmed"
                    ] = False

                    st.rerun()

            with c2:

                if st.button(
                    "✅ Comfortable",
                    use_container_width=True,
                    key="guided_safety_comfortable"
                ):

                    st.session_state[
                        "guided_safety_confirmed"
                    ] = True

                    st.session_state[
                        "guided_loudness_level"
                    ] = safety_level

                    st.session_state[
                        "guided_manual_loudness"
                    ] = safety_level

                    st.rerun()

            with c3:

                if st.button(
                    "🔊 Too loud",
                    use_container_width=True,
                    key="guided_safety_too_loud"
                ):

                    new_level = max(
                        0.01,
                        round(
                            safety_level - 0.01,
                            2
                        )
                    )

                    st.session_state[
                        "guided_safety_level"
                    ] = new_level

                    st.session_state[
                        "guided_safety_confirmed"
                    ] = False

                    st.rerun()

            if safety_confirmed:

                st.success(
                    "Comfortable level confirmed. "
                    "You can continue to Pitch Matching."
                )

            else:

                st.info(
                    "Confirm a comfortable listening level before continuing."
                )

            st.markdown("---")

            c1, c2 = st.columns(
                2
            )

            with c1:

                if st.button(
                    "← Back",
                    use_container_width=True,
                    key="guided_audio_back"
                ):

                    profile_id = (
                        st.session_state.get(
                            "patient_assessment_profile_id"
                        )
                    )

                    if profile_id is not None:

                        cursor.execute(
                            """
                            DELETE FROM tinnitus_profiles
                            WHERE id = ?
                              AND patient_id = ?
                            """,
                            (
                                profile_id,
                                patient_id
                            )
                        )

                        conn.commit()

                    st.session_state[
                        "patient_assessment_profile_id"
                    ] = None

                    st.session_state[
                        "patient_assessment_step"
                    ] = 1

                    st.session_state[
                        "guided_safety_confirmed"
                    ] = False

                    st.rerun()

            with c2:

                if st.button(
                    "Continue to Pitch Matching",
                    use_container_width=True,
                    key="guided_audio_continue"
                ):

                    if not bool(
                        st.session_state.get(
                            "guided_safety_confirmed",
                            False
                        )
                    ):

                        st.error(
                            "Please confirm that the test sound is comfortable "
                            "before continuing."
                        )

                    else:

                        st.session_state[
                            "patient_assessment_step"
                        ] = 3

                        st.rerun()


        # =============================================
        # STEP 3 — PITCH MATCHING
        # =============================================

        elif assessment_step == 3:

            st.subheader(
                "3. Pitch Matching"
            )

            st.info(
                "**Pitch** means how high or low the sound is. "
                "For example, a whistle has a high pitch and a deep hum has a low pitch."
            )

            st.write(
                "Move the frequency slider, then play the tone. "
                "Keep changing the frequency until the tone feels closest "
                "to the pitch of your tinnitus."
            )

            st.warning(
                "Keep the sound at a comfortable level. "
                "This is an experimental matching tool, not a diagnostic test."
            )

            pitch_frequency = st.slider(
                "Frequency (Hz)",
                min_value=250,
                max_value=10000,
                value=int(
                    st.session_state.get(
                        "guided_manual_pitch_hz",
                        4000
                    )
                ),
                step=50,
                key="guided_manual_pitch_slider"
            )

            st.session_state[
                "guided_manual_pitch_hz"
            ] = pitch_frequency

            st.metric(
                "Current frequency",
                f"{pitch_frequency} Hz"
            )

            st.markdown(
                "### Listen to this frequency"
            )

            st.audio(
                generate_pure_tone(
                    frequency=pitch_frequency,
                    duration=1.5,
                    volume=float(
                        st.session_state.get(
                            "guided_safety_level",
                            0.03
                        )
                    )
                ),
                format="audio/wav"
            )

            st.caption(
                "Move the slider again, then press play again. "
                "Repeat until the pitch is the closest match you can find."
            )

            c1, c2 = st.columns(
                2
            )

            with c1:

                if st.button(
                    "← Back to Audio Check",
                    use_container_width=True,
                    key="guided_pitch_back"
                ):

                    st.session_state[
                        "patient_assessment_step"
                    ] = 2

                    st.rerun()

            with c2:

                if st.button(
                    "This Frequency Matches My Tinnitus",
                    use_container_width=True,
                    key="guided_manual_pitch_save"
                ):

                    profile_id = (
                        st.session_state.get(
                            "patient_assessment_profile_id"
                        )
                    )

                    if profile_id is None:

                        st.error(
                            "The current assessment profile could not be found."
                        )

                    else:

                        cursor.execute(
                            """
                            UPDATE tinnitus_profiles
                            SET estimated_pitch_hz = ?
                            WHERE id = ?
                              AND patient_id = ?
                            """,
                            (
                                float(
                                    pitch_frequency
                                ),
                                profile_id,
                                patient_id
                            )
                        )

                        conn.commit()

                        st.session_state[
                            "guided_manual_loudness"
                        ] = float(
                            st.session_state.get(
                                "guided_safety_level",
                                0.03
                            )
                        )

                        st.session_state[
                            "patient_assessment_step"
                        ] = 4

                        st.rerun()


        # =============================================
        # STEP 4 — LOUDNESS MATCHING
        # =============================================

        elif assessment_step == 4:

            st.subheader(
                "4. Loudness Matching"
            )

            st.info(
                "**Loudness** means how quiet or loud the sound seems. "
                "Here you adjust the level until the test tone feels similar "
                "to the perceived loudness of your tinnitus."
            )

            st.warning(
                "The level shown here is only a relative digital level. "
                "It is NOT a calibrated dB SPL or dB HL measurement."
            )

            profile_id = (
                st.session_state.get(
                    "patient_assessment_profile_id"
                )
            )

            cursor.execute(
                """
                SELECT estimated_pitch_hz
                FROM tinnitus_profiles
                WHERE id = ?
                  AND patient_id = ?
                LIMIT 1
                """,
                (
                    profile_id,
                    patient_id
                )
            )

            pitch_row = (
                cursor.fetchone()
            )

            if (
                pitch_row is None
                or pitch_row[0] is None
            ):

                st.error(
                    "No saved pitch was found for this assessment."
                )

            else:

                pitch_hz = float(
                    pitch_row[0]
                )

                st.metric(
                    "Matched pitch",
                    f"{pitch_hz:.0f} Hz"
                )

                relative_level = st.slider(
                    "Relative volume",
                    min_value=0.01,
                    max_value=0.20,
                    value=float(
                        st.session_state.get(
                            "guided_manual_loudness",
                            0.03
                        )
                    ),
                    step=0.01,
                    key="guided_manual_loudness_slider"
                )

                st.session_state[
                    "guided_manual_loudness"
                ] = relative_level

                st.markdown(
                    "### Listen at this level"
                )

                st.audio(
                    generate_pure_tone(
                        frequency=pitch_hz,
                        duration=1.5,
                        volume=relative_level
                    ),
                    format="audio/wav"
                )

                st.caption(
                    "Move the volume slider, then press play again. "
                    "Keep the sound comfortable at all times."
                )

                c1, c2 = st.columns(
                    2
                )

                with c1:

                    if st.button(
                        "← Back to Pitch",
                        use_container_width=True,
                        key="guided_loudness_back"
                    ):

                        st.session_state[
                            "patient_assessment_step"
                        ] = 3

                        st.rerun()

                with c2:

                    if st.button(
                        "This Loudness Matches My Tinnitus",
                        use_container_width=True,
                        key="guided_manual_loudness_save"
                    ):

                        cursor.execute(
                            """
                            UPDATE tinnitus_profiles
                            SET loudness_relative = ?
                            WHERE id = ?
                              AND patient_id = ?
                            """,
                            (
                                float(
                                    relative_level
                                ),
                                profile_id,
                                patient_id
                            )
                        )

                        conn.commit()

                        st.session_state[
                            "patient_assessment_step"
                        ] = 5

                        st.rerun()


        # =============================================
        # STEP 5 — ASSESSMENT COMPLETE
        # =============================================

        elif assessment_step == 5:

            st.subheader(
                "5. Assessment Complete"
            )

            st.success(
                "Your tinnitus assessment has been saved."
            )

            st.write(
                "Your questionnaire answers, matched pitch and relative "
                "loudness are now available in **My Tinnitus Profile**."
            )

            st.info(
                "You can go back to Loudness Matching if you want to adjust "
                "the last value, or open your saved tinnitus profile."
            )

            st.markdown("---")

            c1, c2 = st.columns(
                2
            )

            with c1:

                if st.button(
                    "← Back to Loudness Matching",
                    use_container_width=True,
                    key="guided_complete_back_loudness"
                ):

                    st.session_state[
                        "patient_assessment_step"
                    ] = 4

                    st.rerun()

            with c2:

                st.button(
                    "🎧 Open My Tinnitus Profile",
                    use_container_width=True,
                    on_click=navigate_to_page,
                    args=(
                        "My Tinnitus Profile",
                    ),
                    key="guided_open_profile"
                )

            c1, c2 = st.columns(
                2
            )

            with c1:

                st.button(
                    "📅 Contact / Book Audiologist",
                    use_container_width=True,
                    on_click=navigate_to_page,
                    args=(
                        "Appointments & Care Requests",
                    ),
                    key="guided_go_care"
                )

            with c2:

                st.button(
                    "🏠 Return Home",
                    use_container_width=True,
                    on_click=navigate_to_page,
                    args=(
                        "Home",
                    ),
                    key="guided_return_home"
                )

            st.markdown("---")

            if st.button(
                "↻ Start a New Assessment",
                use_container_width=True,
                key="guided_start_new_assessment"
            ):

                st.session_state[
                    "patient_assessment_step"
                ] = 1

                st.session_state[
                    "patient_assessment_profile_id"
                ] = None

                st.session_state[
                    "guided_manual_pitch_hz"
                ] = 4000

                st.session_state[
                    "guided_manual_loudness"
                ] = 0.03

                st.session_state[
                    "guided_safety_level"
                ] = 0.03

                st.session_state[
                    "guided_safety_confirmed"
                ] = False

                reset_pitch_search(
                    st.session_state
                )

                st.rerun()

        # Stop here for patient users so the professional
        # Pure Tone / Pitch / Loudness technical selector below
        # is not displayed.
        st.stop()

    assessment_mode = st.radio(
        "Assessment",
        [
            "Pure Tone",
            "Pitch Matching",
            "Loudness Matching"
        ],
        horizontal=True
    )

    if assessment_mode == "Pure Tone":

        st.warning(
            "Relative audio level is uncalibrated. "
            "It is not dB SPL or dB HL."
        )

        c1, c2, c3 = st.columns(3)

        frequency = c1.slider(
            "Frequency (Hz)",
            250,
            10000,
            4000,
            50
        )

        duration = c2.slider(
            "Duration",
            1,
            5,
            2
        )

        volume = c3.slider(
            "Relative volume",
            0.01,
            0.20,
            0.05,
            0.01
        )

        if st.button(
            "Generate Tone"
        ):

            st.audio(
                generate_pure_tone(
                    frequency=frequency,
                    duration=duration,
                    volume=volume
                ),
                format="audio/wav"
            )

    elif assessment_mode == "Pitch Matching":

        initialize_pitch_state(
            st.session_state
        )

        options_pitch = (
            get_pitch_options(
                st.session_state
            )
        )

        st.info(
            f"Round "
            f"{st.session_state.pitch_round}"
        )

        cols = st.columns(3)

        for index, freq in enumerate(
            options_pitch
        ):

            with cols[index]:

                st.markdown(
                    f"### {int(freq)} Hz"
                )

                st.audio(
                    generate_pure_tone(
                        frequency=freq,
                        duration=1.5,
                        volume=0.05
                    )
                )

        selected_pitch = st.radio(
            "Closest tone",
            options_pitch,
            format_func=lambda x:
                f"{int(x)} Hz",
            horizontal=True,
            key=(
                f"pitch_choice_"
                f"{st.session_state.pitch_round}"
            )
        )

        if (
            not
            st.session_state.pitch_finished
        ):

            if st.button(
                "Next Round"
            ):

                update_pitch_search(
                    st.session_state,
                    selected_pitch
                )

                st.rerun()

        else:

            final_pitch = (
                get_final_pitch(
                    st.session_state
                )
            )

            st.success(
                f"Estimated pitch: "
                f"{final_pitch} Hz"
            )

            cursor.execute(
                """
                SELECT id, patient_code
                FROM patients
                ORDER BY patient_code
                """
            )

            patient_rows = cursor.fetchall()

            if patient_rows:

                patient_options = {
                    code: pid
                    for pid, code
                    in patient_rows
                }

                patient_code = (
                    select_patient_for_current_user(
                        patient_options,
                        label="Save result to",
                        key="pitch_save_patient"
                    )
                )

                if st.button(
                    "Save Pitch Result"
                ):

                    cursor.execute(
                        """
                        SELECT id
                        FROM tinnitus_profiles
                        WHERE patient_id = ?
                        ORDER BY id DESC
                        LIMIT 1
                        """,
                        (
                            patient_options[
                                patient_code
                            ],
                        )
                    )

                    latest = cursor.fetchone()

                    if latest:

                        cursor.execute(
                            """
                            UPDATE tinnitus_profiles
                            SET estimated_pitch_hz = ?
                            WHERE id = ?
                            """,
                            (
                                final_pitch,
                                latest[0]
                            )
                        )

                        conn.commit()

                        st.success(
                            "Pitch saved."
                        )

            if st.button(
                "Restart Pitch Test"
            ):

                reset_pitch_search(
                    st.session_state
                )

                st.rerun()

    elif assessment_mode == "Loudness Matching":

        cursor.execute(
            """
            SELECT
                p.id,
                p.patient_code,
                tp.estimated_pitch_hz
            FROM patients p

            JOIN tinnitus_profiles tp
            ON tp.id = (
                SELECT t2.id
                FROM tinnitus_profiles t2
                WHERE t2.patient_id = p.id
                AND t2.estimated_pitch_hz IS NOT NULL
                ORDER BY t2.id DESC
                LIMIT 1
            )

            ORDER BY p.patient_code
            """
        )

        rows = cursor.fetchall()

        if not rows:

            st.warning(
                "No patient with saved pitch."
            )

        else:

            options = {
                code: {
                    "id": pid,
                    "pitch": pitch
                }
                for pid, code, pitch
                in rows
            }

            selected = select_patient_for_current_user(
                options,
                label="Patient",
                key="loudness_patient"
            )

            selected_data = (
                options[selected]
            )

            st.metric(
                "Pitch",
                f"{int(selected_data['pitch'])} Hz"
            )

            level = st.slider(
                "Relative Level",
                0.01,
                0.20,
                0.03,
                0.01
            )

            if st.button(
                "Generate Loudness Tone"
            ):

                st.audio(
                    generate_pure_tone(
                        frequency=(
                            selected_data[
                                "pitch"
                            ]
                        ),
                        duration=1.5,
                        volume=level
                    )
                )

            if st.button(
                "Save Loudness Match"
            ):

                cursor.execute(
                    """
                    SELECT id
                    FROM tinnitus_profiles
                    WHERE patient_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (
                        selected_data[
                            "id"
                        ],
                    )
                )

                latest = cursor.fetchone()

                if latest:

                    cursor.execute(
                        """
                        UPDATE tinnitus_profiles
                        SET loudness_relative = ?
                        WHERE id = ?
                        """,
                        (
                            level,
                            latest[0]
                        )
                    )

                    conn.commit()

                    st.success(
                        "Loudness saved."
                    )


# =====================================================
# APPOINTMENTS & CARE REQUESTS
# =====================================================

elif page == "Appointments & Care Requests":

    show_patient_care_page(
        st.session_state.get(
            "auth_patient_id"
        )
    )


# =====================================================
# ASSIGNED ACOUSTIC TEST
# =====================================================

elif page == "Assigned Acoustic Test":

    st.title("Assigned Acoustic Test")
    st.caption(
        "Only sounds prepared and sent by your audiologist appear here. "
        "You do not generate acoustic configurations yourself."
    )

    patient_id = st.session_state.get("auth_patient_id")
    if patient_id is None:
        st.error("This patient account is not linked to a patient record.")
        st.stop()

    assignment = backend_get_pending_acoustic_assignment(patient_id)

    if assignment is None:
        st.info(
            "There is no acoustic test waiting for your feedback right now. "
            "Your audiologist will send a configuration when one is ready."
        )
    else:
        pathway_label = (
            "AI-Assisted" if assignment["pathway"] == "ai"
            else "Audiologist Manual"
        )
        st.success(f"New {pathway_label} acoustic test assigned")

        c1, c2, c3 = st.columns(3)
        c1.metric("Stimulus", assignment["stimulus_type"])
        c2.metric(
            "Frequency",
            "Broadband" if assignment["stimulus_type"] == "White Noise"
            else f"{float(assignment['center_frequency_hz']):.0f} Hz"
        )
        c3.metric("Duration", f"{int(assignment['duration_seconds'])} s")

        st.caption(
            f"Relative digital level: {float(assignment['relative_level']):.02f}. "
            "This is not a calibrated dB SPL or dB HL value."
        )
        if assignment.get("audiologist_note"):
            st.info(f"Audiologist note: {assignment['audiologist_note']}")

        try:
            st.audio(
                generate_stimulus(
                    stimulus_type=assignment["stimulus_type"],
                    frequency=(float(assignment["center_frequency_hz"])
                               if assignment["center_frequency_hz"] is not None else 1000.0),
                    duration=int(assignment["duration_seconds"]),
                    volume=float(assignment["relative_level"]),
                    bandwidth=(float(assignment["bandwidth_hz"])
                               if assignment["bandwidth_hz"] is not None else 500.0),
                ),
                format="audio/wav"
            )
        except Exception as error:
            st.error(f"Could not generate the assigned sound: {error}")

        st.markdown("### Your feedback")
        st.caption("Listen to the assigned sound first, then record how it felt.")
        f1, f2 = st.columns(2)
        comfort = f1.slider("Comfort", 0, 10, 5, key=f"assigned_comfort_{assignment['id']}")
        noticeability = f2.slider("Tinnitus noticeability", 0, 10, 5, key=f"assigned_noticeability_{assignment['id']}")
        listen_again = st.radio(
            "Would you listen to this sound again?", ["Yes", "No"],
            horizontal=True, key=f"assigned_again_{assignment['id']}"
        )
        patient_comment = st.text_area(
            "Optional comment",
            placeholder="Example: more comfortable than the previous sound.",
            key=f"assigned_comment_{assignment['id']}"
        )

        if st.button("Submit Feedback", use_container_width=True, key=f"submit_assigned_feedback_{assignment['id']}"):
            saved = backend_submit_acoustic_feedback(
                assignment_id=assignment["id"], patient_id=patient_id,
                comfort_score=comfort, tinnitus_noticeability=noticeability,
                listen_again=listen_again, patient_comment=patient_comment,
            )
            if saved:
                st.success(
                    "Feedback sent to your audiologist. They can now review it and decide "
                    "whether another sound should be prepared."
                )
                st.rerun()
            else:
                st.error("This assignment could not be updated.")


# =====================================================
# MY TINNITUS PROFILE
# =====================================================

elif page == "My Tinnitus Profile":

    st.title(
        "My Tinnitus Profile"
    )

    patient_id = (
        st.session_state.get(
            "auth_patient_id"
        )
    )

    if patient_id is None:

        st.error(
            "This patient account is not linked to a patient record."
        )

        st.stop()

    cursor.execute(
        """
        SELECT
            patient_code,
            age,
            sex
        FROM patients
        WHERE id = ?
        LIMIT 1
        """,
        (
            patient_id,
        )
    )

    patient_info = (
        cursor.fetchone()
    )

    cursor.execute(
        """
        SELECT
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
        """,
        (
            patient_id,
        )
    )

    profile = (
        cursor.fetchone()
    )

    if patient_info is not None:

        patient_code, patient_age, patient_sex = (
            patient_info
        )

        st.caption(
            f"Patient {patient_code}"
        )

    if profile is None:

        st.info(
            "You do not have a saved tinnitus assessment yet."
        )

        st.button(
            "Start My First Assessment",
            use_container_width=True,
            on_click=navigate_to_page,
            args=(
                "Tinnitus Assessment",
            ),
            key="profile_start_assessment"
        )

    else:

        (
            laterality,
            tinnitus_type,
            temporal_pattern,
            duration_months,
            annoyance,
            sleep,
            concentration,
            stress,
            estimated_pitch,
            loudness_relative,
            profile_created_at
        ) = profile

        st.success(
            "Latest saved tinnitus assessment"
        )

        if profile_created_at:

            st.caption(
                f"Assessment recorded: {profile_created_at}"
            )

        st.markdown(
            "### Tinnitus characteristics"
        )

        c1, c2, c3 = st.columns(
            3
        )

        c1.metric(
            "Location",
            laterality
        )

        c2.metric(
            "Sound type",
            tinnitus_type
        )

        c3.metric(
            "Pattern",
            temporal_pattern
        )

        c1, c2, c3 = st.columns(
            3
        )

        c1.metric(
            "Duration",
            f"{int(duration_months)} months"
        )

        c2.metric(
            "Matched pitch",
            (
                f"{float(estimated_pitch):.0f} Hz"
                if estimated_pitch is not None
                else "Not completed"
            )
        )

        c3.metric(
            "Relative loudness",
            (
                f"{float(loudness_relative):.2f}"
                if loudness_relative is not None
                else "Not completed"
            )
        )

        st.caption(
            "Pitch = how high or low the matched sound is. "
            "Loudness = the relative digital level matched by the patient; "
            "it is not a calibrated dB value."
        )

        st.markdown("---")

        st.markdown(
            "### Patient-reported impact"
        )

        c1, c2, c3, c4 = st.columns(
            4
        )

        c1.metric(
            "Annoyance",
            f"{int(annoyance)}/10"
        )

        c2.metric(
            "Sleep impact",
            f"{int(sleep)}/10"
        )

        c3.metric(
            "Concentration",
            f"{int(concentration)}/10"
        )

        c4.metric(
            "Stress",
            f"{int(stress)}/10"
        )


        st.markdown("---")

        c1, c2 = st.columns(
            2
        )

        with c1:

            st.button(
                "Start a New Assessment",
                use_container_width=True,
                on_click=start_new_patient_assessment,
                key="profile_new_assessment"
            )

        with c2:

            st.button(
                "Contact / Book Audiologist",
                use_container_width=True,
                on_click=navigate_to_page,
                args=(
                    "Appointments & Care Requests",
                ),
                key="profile_to_care"
            )


# =====================================================
# ACOUSTIC PERSONALIZATION
# =====================================================

elif page == "Acoustic Personalization":

    # Patient self-service acoustic personalization has been removed.
    # Acoustic generation and testing are controlled by the audiologist.
    if st.session_state.get("auth_role") == "patient":

        st.title("Audiologist-Supervised Acoustic Care")
        st.warning(
            "Acoustic sounds are no longer generated or tested independently on the patient side. "
            "Please contact your audiologist to arrange supervised acoustic testing."
        )
        st.button(
            "Open Appointments & Care Requests",
            use_container_width=True,
            on_click=navigate_to_page,
            args=("Appointments & Care Requests",),
            key="patient_acoustic_redirect_to_care"
        )
        st.stop()

    # =================================================
    # LEGACY PATIENT PERSONALIZATION CODE (UNREACHABLE)
    # retained temporarily for database/backward compatibility.
    # =================================================

    if st.session_state[
        "auth_role"
    ] == "patient":

        st.title(
            "Acoustic Personalization"
        )

        st.success(
            "✅ TWO-PATHWAY WORKFLOW ACTIVE — choose Standard or AI-Assisted Personalization."
        )

        st.caption(
            "Choose the standard feedback-driven pathway or the optional AI-assisted "
            "pathway. You may complete one or both. Their results and professional "
            "decisions remain independent."
        )

        st.warning(
            "Playback levels are relative digital amplitudes and are not "
            "calibrated dB SPL or dB HL. Stop playback if a sound is uncomfortable."
        )

        patient_id = st.session_state.get(
            "auth_patient_id"
        )

        ensure_personalization_review_table(
            conn
        )

        # ---------------------------------------------
        # Latest tinnitus profile
        # ---------------------------------------------

        cursor.execute(
            """
            SELECT
                id,
                laterality,
                tinnitus_type,
                temporal_pattern,
                duration_months,
                annoyance_score,
                sleep_score,
                concentration_score,
                stress_score,
                estimated_pitch_hz,
                loudness_relative
            FROM tinnitus_profiles
            WHERE patient_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                patient_id,
            )
        )

        profile_row = cursor.fetchone()

        if profile_row is None:

            st.info(
                "Complete your Tinnitus Assessment before starting "
                "Acoustic Personalization."
            )

            st.button(
                "Go to Tinnitus Assessment",
                use_container_width=True,
                on_click=navigate_to_page,
                args=(
                    "Tinnitus Assessment",
                ),
                key="patient_personalization_go_assessment"
            )

        else:

            (
                profile_id,
                profile_laterality,
                profile_type,
                profile_pattern,
                profile_duration,
                profile_annoyance,
                profile_sleep,
                profile_concentration,
                profile_stress,
                profile_pitch,
                profile_loudness
            ) = profile_row

            if (
                profile_pitch is None
                or
                profile_loudness is None
            ):

                st.info(
                    "Your latest tinnitus profile is not complete yet. "
                    "Finish pitch and loudness matching first."
                )

                st.button(
                    "Continue Tinnitus Assessment",
                    use_container_width=True,
                    on_click=navigate_to_page,
                    args=(
                        "Tinnitus Assessment",
                    ),
                    key="patient_personalization_finish_assessment"
                )

            else:

                profile_pitch = float(
                    profile_pitch
                )

                profile_loudness = float(
                    profile_loudness
                )

                safe_level = max(
                    0.01,
                    min(
                        0.10,
                        profile_loudness
                    )
                )

                # -------------------------------------
                # Profile summary shared by both paths
                # -------------------------------------

                st.markdown(
                    "### Your tinnitus profile"
                )

                p1, p2, p3, p4 = st.columns(
                    4
                )

                p1.metric(
                    "Matched Pitch",
                    f"{int(round(profile_pitch))} Hz"
                )

                p2.metric(
                    "Type",
                    profile_type or "Not recorded"
                )

                p3.metric(
                    "Location",
                    profile_laterality or "Not recorded"
                )

                p4.metric(
                    "Relative Loudness",
                    f"{profile_loudness:.2f}"
                )

                st.markdown("---")

                if (
                    "patient_personalization_path"
                    not in st.session_state
                ):
                    st.session_state[
                        "patient_personalization_path"
                    ] = None

                personalization_path = st.session_state.get(
                    "patient_personalization_path"
                )

                if personalization_path is None:

                    st.markdown(
                        "## Choose ONE pathway to start"
                    )

                    st.write(
                        "Choose either pathway. You may complete only one, or complete "
                        "both. Each pathway keeps its own result and its own independent "
                        "audiologist review/status."
                    )

                    standard_landing_review = get_latest_patient_pathway_review(
                        conn,
                        patient_id,
                        "standard"
                    )

                    ai_landing_review = get_latest_patient_pathway_review(
                        conn,
                        patient_id,
                        "ai"
                    )

                    standard_landing_status = (
                        standard_landing_review[0]
                        if standard_landing_review is not None
                        else "Not submitted yet"
                    )

                    ai_landing_status = (
                        ai_landing_review[0]
                        if ai_landing_review is not None
                        else "Not submitted yet"
                    )

                    path_col1, path_col2 = st.columns(
                        2
                    )

                    with path_col1:

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                "### 🎧 Standard Personalization"
                            )

                            st.write(
                                "Personalize sounds using your tinnitus profile, "
                                "your real listening feedback, observed response ranking "
                                "and transparent rule-based refinement."
                            )

                            st.caption(
                                "No AI model is used in this pathway."
                            )

                            st.markdown("---")

                            st.markdown(
                                "#### 👨‍⚕️ Standard Professional Status"
                            )

                            if standard_landing_status == "Approved":
                                st.success(
                                    "✅ Approved"
                                )
                            elif standard_landing_status == "Modified":
                                st.warning(
                                    "✏️ Modified"
                                )
                            elif standard_landing_status == "Rejected":
                                st.error(
                                    "❌ Rejected"
                                )
                            elif standard_landing_status == "Pending Review":
                                st.info(
                                    "⏳ Pending audiologist review"
                                )
                            else:
                                st.caption(
                                    "No Standard result has been submitted for professional review yet."
                                )

                            st.button(
                                "Start Standard Personalization",
                                use_container_width=True,
                                on_click=set_patient_personalization_path,
                                args=(
                                    "Standard Personalization",
                                ),
                                key="patient_choose_standard_path"
                            )

                    with path_col2:

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                "### 🤖 AI-Assisted Personalization"
                            )

                            st.write(
                                "Use an optional machine-learning model to predict and rank "
                                "candidate sounds, then test the AI-recommended candidate "
                                "and record your real response."
                            )

                            st.caption(
                                "AI is optional. You can use SmartTinnitus without it."
                            )

                            st.markdown("---")

                            st.markdown(
                                "#### 👨‍⚕️ AI Professional Status"
                            )

                            if ai_landing_status == "Approved":
                                st.success(
                                    "✅ Approved"
                                )
                            elif ai_landing_status == "Modified":
                                st.warning(
                                    "✏️ Modified"
                                )
                            elif ai_landing_status == "Rejected":
                                st.error(
                                    "❌ Rejected"
                                )
                            elif ai_landing_status == "Pending Review":
                                st.info(
                                    "⏳ Pending audiologist review"
                                )
                            else:
                                st.caption(
                                    "No AI result has been submitted for professional review yet."
                                )

                            st.button(
                                "Start AI-Assisted Personalization",
                                use_container_width=True,
                                on_click=set_patient_personalization_path,
                                args=(
                                    "AI Personalization (Optional)",
                                ),
                                key="patient_choose_ai_path"
                            )

                    st.markdown("---")

                    st.caption(
                        "Standard and AI professional decisions are independent. "
                        "For example, Standard may be Approved while AI is Modified or Rejected."
                    )

                    st.button(
                        "Compare My Standard & AI Results",
                        use_container_width=True,
                        on_click=set_patient_personalization_path,
                        args=(
                            "Results Comparison",
                        ),
                        key="patient_open_personalization_results"
                    )

                # =========================================================
                # PATH 1 — STANDARD PERSONALIZATION (NON-AI)
                # =========================================================

                if personalization_path == "Standard Personalization":

                    st.button(
                        "← Back to Personalization Choices",
                        on_click=set_patient_personalization_path,
                        args=(
                            None,
                        ),
                        key="patient_standard_back_to_choices"
                    )

                    st.markdown(
                        "## Standard Personalization"
                    )

                    st.write(
                        "This pathway uses your actual listening feedback. "
                        "It tests initial sound types, ranks the observed responses, "
                        "then applies transparent rule-based refinement around the "
                        "best-rated candidate. No AI model is required."
                    )

                    st.info(
                        "Standard pathway: Candidate testing → observed response ranking "
                        "→ rule-based refinement → best-rated candidate → professional review."
                    )

                    if (
                        "patient_personalization_step"
                        not in st.session_state
                    ):
                        reset_patient_personalization_state()

                    personalization_step = int(
                        st.session_state.get(
                            "patient_personalization_step",
                            1
                        )
                    )

                    initial_bandwidth = max(
                        200.0,
                        min(
                            1000.0,
                            profile_pitch * 0.12
                        )
                    )

                    initial_candidates = [
                        {
                            "label": "Candidate 1",
                            "stimulus_type": "Pure Tone",
                            "frequency": profile_pitch,
                            "bandwidth": None,
                            "level": safe_level,
                            "reason": "A tone anchored to your matched tinnitus pitch."
                        },
                        {
                            "label": "Candidate 2",
                            "stimulus_type": "Narrow-Band Noise",
                            "frequency": profile_pitch,
                            "bandwidth": initial_bandwidth,
                            "level": safe_level,
                            "reason": "A narrow noise band centered around your matched pitch."
                        },
                        {
                            "label": "Candidate 3",
                            "stimulus_type": "White Noise",
                            "frequency": profile_pitch,
                            "bandwidth": None,
                            "level": safe_level,
                            "reason": "A broadband comparison sound at the same relative level."
                        }
                    ]

                    # ---------------------------------
                    # Standard Step 1 — Start
                    # ---------------------------------

                    if personalization_step == 1:

                        st.markdown(
                            "### Start Standard Personalization"
                        )

                        st.write(
                            "You will test three initial sounds, then three refined "
                            "variations of the best-rated initial sound."
                        )

                        st.button(
                            "▶ Start Standard Personalization",
                            use_container_width=True,
                            key="patient_standard_start",
                            on_click=lambda: st.session_state.update(
                                {
                                    "patient_personalization_step": 2,
                                    "patient_personalization_initial_index": 0,
                                    "patient_personalization_refine_index": 0,
                                    "patient_personalization_initial_results": [],
                                    "patient_personalization_refined_results": [],
                                    "patient_personalization_refine_candidates": [],
                                    "patient_personalization_standard_result_trial_id": None
                                }
                            )
                        )

                    # ---------------------------------
                    # Standard Step 2 — Initial sounds
                    # ---------------------------------

                    elif personalization_step == 2:

                        initial_index = int(
                            st.session_state.get(
                                "patient_personalization_initial_index",
                                0
                            )
                        )

                        if initial_index >= len(initial_candidates):

                            initial_results = list(
                                st.session_state.get(
                                    "patient_personalization_initial_results",
                                    []
                                )
                            )

                            if not initial_results:

                                st.error(
                                    "No standard-path responses were recorded. "
                                    "Please restart the session."
                                )

                                st.button(
                                    "Restart Standard Personalization",
                                    on_click=reset_patient_personalization_state,
                                    use_container_width=True,
                                    key="patient_standard_restart_empty"
                                )

                            else:

                                best_initial = max(
                                    initial_results,
                                    key=lambda item: item[
                                        "score"
                                    ]
                                )

                                best_frequency = float(
                                    best_initial[
                                        "frequency"
                                    ]
                                )

                                best_level = float(
                                    best_initial[
                                        "level"
                                    ]
                                )

                                best_type = best_initial[
                                    "stimulus_type"
                                ]

                                best_bandwidth = best_initial.get(
                                    "bandwidth"
                                )

                                # Transparent rule-based refinement.
                                if best_type == "White Noise":

                                    refine_candidates = [
                                        {
                                            "label": "Refined Candidate A",
                                            "stimulus_type": best_type,
                                            "frequency": profile_pitch,
                                            "bandwidth": None,
                                            "level": max(0.01, best_level * 0.80)
                                        },
                                        {
                                            "label": "Refined Candidate B",
                                            "stimulus_type": best_type,
                                            "frequency": profile_pitch,
                                            "bandwidth": None,
                                            "level": best_level
                                        },
                                        {
                                            "label": "Refined Candidate C",
                                            "stimulus_type": best_type,
                                            "frequency": profile_pitch,
                                            "bandwidth": None,
                                            "level": min(0.10, best_level * 1.20)
                                        }
                                    ]

                                elif best_type == "Narrow-Band Noise":

                                    reference_bandwidth = float(
                                        best_bandwidth
                                        if best_bandwidth is not None
                                        else initial_bandwidth
                                    )

                                    refine_candidates = [
                                        {
                                            "label": "Refined Candidate A",
                                            "stimulus_type": best_type,
                                            "frequency": max(250.0, best_frequency * 0.90),
                                            "bandwidth": min(2000.0, reference_bandwidth * 1.20),
                                            "level": best_level
                                        },
                                        {
                                            "label": "Refined Candidate B",
                                            "stimulus_type": best_type,
                                            "frequency": best_frequency,
                                            "bandwidth": max(100.0, reference_bandwidth * 0.80),
                                            "level": best_level
                                        },
                                        {
                                            "label": "Refined Candidate C",
                                            "stimulus_type": best_type,
                                            "frequency": min(10000.0, best_frequency * 1.10),
                                            "bandwidth": reference_bandwidth,
                                            "level": best_level
                                        }
                                    ]

                                else:

                                    refine_candidates = [
                                        {
                                            "label": "Refined Candidate A",
                                            "stimulus_type": best_type,
                                            "frequency": max(250.0, best_frequency * 0.90),
                                            "bandwidth": None,
                                            "level": best_level
                                        },
                                        {
                                            "label": "Refined Candidate B",
                                            "stimulus_type": best_type,
                                            "frequency": best_frequency,
                                            "bandwidth": None,
                                            "level": best_level
                                        },
                                        {
                                            "label": "Refined Candidate C",
                                            "stimulus_type": best_type,
                                            "frequency": min(10000.0, best_frequency * 1.10),
                                            "bandwidth": None,
                                            "level": best_level
                                        }
                                    ]

                                st.session_state[
                                    "patient_personalization_refine_candidates"
                                ] = refine_candidates

                                st.session_state[
                                    "patient_personalization_refine_index"
                                ] = 0

                                st.session_state[
                                    "patient_personalization_step"
                                ] = 3

                                st.rerun()

                        else:

                            candidate = initial_candidates[
                                initial_index
                            ]

                            st.markdown(
                                f"### Initial Sound {initial_index + 1} of {len(initial_candidates)}"
                            )

                            st.progress(
                                initial_index / len(initial_candidates)
                            )

                            st.markdown(
                                f"**{candidate['label']} — {candidate['stimulus_type']}**"
                            )

                            st.caption(
                                candidate[
                                    "reason"
                                ]
                            )

                            if candidate[
                                "stimulus_type"
                            ] == "White Noise":

                                st.write(
                                    f"Relative level: **{candidate['level']:.2f}**"
                                )

                            else:

                                st.write(
                                    f"Frequency: **{int(round(candidate['frequency']))} Hz**"
                                )

                                if candidate.get(
                                    "bandwidth"
                                ) is not None:

                                    st.write(
                                        f"Bandwidth: **{int(round(candidate['bandwidth']))} Hz**"
                                    )

                                st.write(
                                    f"Relative level: **{candidate['level']:.2f}**"
                                )

                            try:

                                st.audio(
                                    generate_stimulus(
                                        stimulus_type=candidate[
                                            "stimulus_type"
                                        ],
                                        frequency=float(
                                            candidate[
                                                "frequency"
                                            ]
                                        ),
                                        duration=2,
                                        volume=float(
                                            candidate[
                                                "level"
                                            ]
                                        ),
                                        bandwidth=float(
                                            candidate.get(
                                                "bandwidth"
                                            )
                                            or 500.0
                                        )
                                    ),
                                    format="audio/wav"
                                )

                            except Exception as error:

                                st.error(
                                    f"Could not generate this sound: {error}"
                                )

                            st.markdown(
                                "#### Your feedback"
                            )

                            f1, f2 = st.columns(
                                2
                            )

                            comfort = f1.slider(
                                "Comfort",
                                0,
                                10,
                                5,
                                key=f"patient_standard_initial_comfort_{initial_index}"
                            )

                            noticeability = f2.slider(
                                "Tinnitus noticeability",
                                0,
                                10,
                                5,
                                key=f"patient_standard_initial_noticeability_{initial_index}"
                            )

                            listen_again = st.radio(
                                "Would you listen to this sound again?",
                                [
                                    "Yes",
                                    "No"
                                ],
                                horizontal=True,
                                key=f"patient_standard_initial_again_{initial_index}"
                            )

                            if st.button(
                                "Save Feedback & Next Sound",
                                use_container_width=True,
                                key=f"patient_standard_initial_save_{initial_index}"
                            ):

                                score = calculate_response_score(
                                    comfort,
                                    noticeability,
                                    listen_again
                                )

                                saved_bandwidth = (
                                    float(
                                        candidate[
                                            "bandwidth"
                                        ]
                                    )
                                    if candidate[
                                        "stimulus_type"
                                    ] == "Narrow-Band Noise"
                                    else None
                                )

                                cursor.execute(
                                    """
                                    INSERT INTO stimulus_trials
                                    (
                                        patient_id,
                                        session_id,
                                        profile_id,
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
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        patient_id,
                                        None,
                                        profile_id,
                                        "patient_personalization_initial",
                                        now_timestamp(),
                                        candidate[
                                            "stimulus_type"
                                        ],
                                        float(
                                            candidate[
                                                "frequency"
                                            ]
                                        ),
                                        saved_bandwidth,
                                        float(
                                            candidate[
                                                "level"
                                            ]
                                        ),
                                        int(
                                            comfort
                                        ),
                                        int(
                                            noticeability
                                        ),
                                        listen_again,
                                        now_timestamp()
                                    )
                                )

                                conn.commit()

                                results = list(
                                    st.session_state.get(
                                        "patient_personalization_initial_results",
                                        []
                                    )
                                )

                                results.append(
                                    {
                                        **candidate,
                                        "comfort": int(comfort),
                                        "noticeability": int(noticeability),
                                        "listen_again": listen_again,
                                        "score": float(score)
                                    }
                                )

                                st.session_state[
                                    "patient_personalization_initial_results"
                                ] = results

                                st.session_state[
                                    "patient_personalization_initial_index"
                                ] = initial_index + 1

                                st.rerun()

                    # ---------------------------------
                    # Standard Step 3 — Rule refinement
                    # ---------------------------------

                    elif personalization_step == 3:

                        refine_candidates = list(
                            st.session_state.get(
                                "patient_personalization_refine_candidates",
                                []
                            )
                        )

                        refine_index = int(
                            st.session_state.get(
                                "patient_personalization_refine_index",
                                0
                            )
                        )

                        if not refine_candidates:

                            st.error(
                                "The rule-based refinement candidates are missing. "
                                "Please restart Standard Personalization."
                            )

                            st.button(
                                "Restart Standard Personalization",
                                on_click=reset_patient_personalization_state,
                                use_container_width=True,
                                key="patient_standard_restart_refine"
                            )

                        elif refine_index >= len(refine_candidates):

                            st.session_state[
                                "patient_personalization_step"
                            ] = 4

                            st.rerun()

                        else:

                            candidate = refine_candidates[
                                refine_index
                            ]

                            st.markdown(
                                "### Rule-Based Refinement"
                            )

                            st.info(
                                "The next sounds are transparent variations around the "
                                "best-rated initial candidate. This is not AI."
                            )

                            st.markdown(
                                f"### Refined Sound {refine_index + 1} of {len(refine_candidates)}"
                            )

                            st.progress(
                                refine_index / len(refine_candidates)
                            )

                            st.markdown(
                                f"**{candidate['label']} — {candidate['stimulus_type']}**"
                            )

                            if candidate[
                                "stimulus_type"
                            ] == "White Noise":

                                st.write(
                                    f"Relative level: **{candidate['level']:.2f}**"
                                )

                            else:

                                st.write(
                                    f"Frequency: **{int(round(candidate['frequency']))} Hz**"
                                )

                                if candidate.get(
                                    "bandwidth"
                                ) is not None:

                                    st.write(
                                        f"Bandwidth: **{int(round(candidate['bandwidth']))} Hz**"
                                    )

                                st.write(
                                    f"Relative level: **{candidate['level']:.2f}**"
                                )

                            try:

                                st.audio(
                                    generate_stimulus(
                                        stimulus_type=candidate[
                                            "stimulus_type"
                                        ],
                                        frequency=float(
                                            candidate[
                                                "frequency"
                                            ]
                                        ),
                                        duration=2,
                                        volume=float(
                                            candidate[
                                                "level"
                                            ]
                                        ),
                                        bandwidth=float(
                                            candidate.get(
                                                "bandwidth"
                                            )
                                            or 500.0
                                        )
                                    ),
                                    format="audio/wav"
                                )

                            except Exception as error:

                                st.error(
                                    f"Could not generate this refined sound: {error}"
                                )

                            st.markdown(
                                "#### Your feedback"
                            )

                            f1, f2 = st.columns(
                                2
                            )

                            comfort = f1.slider(
                                "Comfort",
                                0,
                                10,
                                5,
                                key=f"patient_standard_refined_comfort_{refine_index}"
                            )

                            noticeability = f2.slider(
                                "Tinnitus noticeability",
                                0,
                                10,
                                5,
                                key=f"patient_standard_refined_noticeability_{refine_index}"
                            )

                            listen_again = st.radio(
                                "Would you listen to this sound again?",
                                [
                                    "Yes",
                                    "No"
                                ],
                                horizontal=True,
                                key=f"patient_standard_refined_again_{refine_index}"
                            )

                            if st.button(
                                "Save Feedback & Continue",
                                use_container_width=True,
                                key=f"patient_standard_refined_save_{refine_index}"
                            ):

                                score = calculate_response_score(
                                    comfort,
                                    noticeability,
                                    listen_again
                                )

                                saved_bandwidth = (
                                    float(
                                        candidate[
                                            "bandwidth"
                                        ]
                                    )
                                    if candidate[
                                        "stimulus_type"
                                    ] == "Narrow-Band Noise"
                                    else None
                                )

                                cursor.execute(
                                    """
                                    INSERT INTO stimulus_trials
                                    (
                                        patient_id,
                                        session_id,
                                        profile_id,
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
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        patient_id,
                                        None,
                                        profile_id,
                                        "patient_personalization_refined",
                                        now_timestamp(),
                                        candidate[
                                            "stimulus_type"
                                        ],
                                        float(
                                            candidate[
                                                "frequency"
                                            ]
                                        ),
                                        saved_bandwidth,
                                        float(
                                            candidate[
                                                "level"
                                            ]
                                        ),
                                        int(
                                            comfort
                                        ),
                                        int(
                                            noticeability
                                        ),
                                        listen_again,
                                        now_timestamp()
                                    )
                                )

                                conn.commit()

                                results = list(
                                    st.session_state.get(
                                        "patient_personalization_refined_results",
                                        []
                                    )
                                )

                                results.append(
                                    {
                                        **candidate,
                                        "comfort": int(comfort),
                                        "noticeability": int(noticeability),
                                        "listen_again": listen_again,
                                        "score": float(score)
                                    }
                                )

                                st.session_state[
                                    "patient_personalization_refined_results"
                                ] = results

                                st.session_state[
                                    "patient_personalization_refine_index"
                                ] = refine_index + 1

                                st.rerun()

                    # ---------------------------------
                    # Standard Step 4 — Final result
                    # ---------------------------------

                    else:

                        all_results = (
                            list(
                                st.session_state.get(
                                    "patient_personalization_initial_results",
                                    []
                                )
                            )
                            +
                            list(
                                st.session_state.get(
                                    "patient_personalization_refined_results",
                                    []
                                )
                            )
                        )

                        st.markdown(
                            "### Standard Personalization Result"
                        )

                        if not all_results:

                            st.info(
                                "No completed Standard Personalization result is available yet."
                            )

                        else:

                            best_result = max(
                                all_results,
                                key=lambda item: item[
                                    "score"
                                ]
                            )

                            standard_result_trial_id = st.session_state.get(
                                "patient_personalization_standard_result_trial_id"
                            )

                            if standard_result_trial_id is None:

                                saved_bandwidth = (
                                    float(
                                        best_result[
                                            "bandwidth"
                                        ]
                                    )
                                    if best_result[
                                        "stimulus_type"
                                    ] == "Narrow-Band Noise"
                                    and best_result.get(
                                        "bandwidth"
                                    ) is not None
                                    else None
                                )

                                cursor.execute(
                                    """
                                    INSERT INTO stimulus_trials
                                    (
                                        patient_id,
                                        session_id,
                                        profile_id,
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
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        patient_id,
                                        None,
                                        profile_id,
                                        "patient_standard_result",
                                        now_timestamp(),
                                        best_result[
                                            "stimulus_type"
                                        ],
                                        float(
                                            best_result[
                                                "frequency"
                                            ]
                                        ),
                                        saved_bandwidth,
                                        float(
                                            best_result[
                                                "level"
                                            ]
                                        ),
                                        int(
                                            best_result[
                                                "comfort"
                                            ]
                                        ),
                                        int(
                                            best_result[
                                                "noticeability"
                                            ]
                                        ),
                                        best_result[
                                            "listen_again"
                                        ],
                                        now_timestamp()
                                    )
                                )

                                conn.commit()

                                standard_result_trial_id = int(
                                    cursor.lastrowid
                                )

                                st.session_state[
                                    "patient_personalization_standard_result_trial_id"
                                ] = standard_result_trial_id

                                create_personalization_review_request(
                                    conn,
                                    patient_id,
                                    profile_id,
                                    "standard",
                                    standard_result_trial_id
                                )

                            st.success(
                                "Your best-rated Standard candidate has been saved separately."
                            )

                            r1, r2, r3, r4 = st.columns(
                                4
                            )

                            r1.metric(
                                "Best-Rated Sound",
                                best_result[
                                    "stimulus_type"
                                ]
                            )

                            r2.metric(
                                "Frequency",
                                (
                                    "Broadband"
                                    if best_result[
                                        "stimulus_type"
                                    ] == "White Noise"
                                    else f"{int(round(best_result['frequency']))} Hz"
                                )
                            )

                            r3.metric(
                                "Comfort",
                                f"{best_result['comfort']}/10"
                            )

                            r4.metric(
                                "Observed Response",
                                f"{best_result['score']:.1f} / 100"
                            )

                            if (
                                best_result[
                                    "stimulus_type"
                                ] == "Narrow-Band Noise"
                                and best_result.get(
                                    "bandwidth"
                                ) is not None
                            ):

                                st.metric(
                                    "Bandwidth",
                                    f"{int(round(best_result['bandwidth']))} Hz"
                                )

                            review_row = get_personalization_review(
                                conn,
                                "standard",
                                standard_result_trial_id
                            )

                            standard_status = (
                                review_row[0]
                                if review_row is not None
                                else "Pending Review"
                            )

                            st.markdown("---")
                            st.markdown(
                                "### 👨‍⚕️ Standard Path — Professional Review"
                            )

                            if standard_status == "Approved":
                                st.success(
                                    "✅ Standard Professional Status: Approved"
                                )
                            elif standard_status == "Modified":
                                st.warning(
                                    "✏️ Standard Professional Status: Modified"
                                )
                            elif standard_status == "Rejected":
                                st.error(
                                    "❌ Standard Professional Status: Rejected"
                                )
                            else:
                                st.info(
                                    "⏳ Standard Professional Status: Pending Review"
                                )

                            st.caption(
                                "This status belongs only to the Standard pathway. "
                                "It does not determine the AI pathway status."
                            )

                            try:

                                st.audio(
                                    generate_stimulus(
                                        stimulus_type=best_result[
                                            "stimulus_type"
                                        ],
                                        frequency=float(
                                            best_result[
                                                "frequency"
                                            ]
                                        ),
                                        duration=2,
                                        volume=float(
                                            best_result[
                                                "level"
                                            ]
                                        ),
                                        bandwidth=float(
                                            best_result.get(
                                                "bandwidth"
                                            )
                                            or 500.0
                                        )
                                    ),
                                    format="audio/wav"
                                )

                            except Exception as error:

                                st.error(
                                    f"Could not replay the Standard result: {error}"
                                )

                        b1, b2 = st.columns(
                            2
                        )

                        with b1:

                            st.button(
                                "↻ Start Another Standard Session",
                                use_container_width=True,
                                on_click=reset_patient_personalization_state,
                                key="patient_standard_new_session"
                            )

                        with b2:

                            st.button(
                                "Compare Standard & AI Results",
                                use_container_width=True,
                                on_click=set_patient_personalization_path,
                                args=(
                                    "Results Comparison",
                                ),
                                key="patient_standard_view_status"
                            )

                # =========================================================
                # PATH 2 — OPTIONAL AI PERSONALIZATION
                # =========================================================

                elif personalization_path == "AI Personalization (Optional)":

                    st.button(
                        "← Back to Personalization Choices",
                        on_click=set_patient_personalization_path,
                        args=(
                            None,
                        ),
                        key="patient_ai_back_to_choices"
                    )

                    st.markdown(
                        "## AI-Assisted Personalization"
                    )

                    st.write(
                        "This is a separate optional pathway. The model predicts and "
                        "ranks candidate sounds from your profile and previous acoustic "
                        "trials. You then test the AI-recommended candidate and provide "
                        "real feedback."
                    )

                    st.info(
                        "AI pathway: ML predicted ranking → recommended candidate → "
                        "patient test → observed response → professional review."
                    )

                    st.caption(
                        "AI predictions are experimental decision-support outputs, not "
                        "diagnoses or validated treatment recommendations."
                    )

                    if (
                        "patient_ai_recommendation"
                        not in st.session_state
                    ):
                        reset_patient_ai_personalization_state()

                    if not model_exists():

                        st.warning(
                            "No trained AI model is currently available. You can still "
                            "use Standard Personalization without AI."
                        )

                    else:

                        patient_features = {
                            "tinnitus_type": profile_type,
                            "laterality": profile_laterality,
                            "temporal_pattern": profile_pattern,
                            "duration_months": profile_duration,
                            "annoyance_score": profile_annoyance,
                            "sleep_score": profile_sleep,
                            "concentration_score": profile_concentration,
                            "stress_score": profile_stress,
                            "estimated_pitch_hz": profile_pitch,
                            "loudness_relative": profile_loudness
                        }

                        ai_recommendation = st.session_state.get(
                            "patient_ai_recommendation"
                        )

                        if ai_recommendation is None:

                            st.markdown(
                                "### Generate an AI recommendation"
                            )

                            st.write(
                                "The AI pathway is not automatic. Start it only if you "
                                "want an AI-assisted candidate recommendation."
                            )

                            if st.button(
                                "🤖 Generate AI Recommendation",
                                use_container_width=True,
                                key="patient_generate_ai_recommendation"
                            ):

                                try:

                                    patient_trial_history = (
                                        get_patient_trial_history_for_adaptation(
                                            conn,
                                            patient_id
                                        )
                                    )

                                    candidate_stimuli, adaptive_metadata = (
                                        generate_adaptive_candidate_stimuli(
                                            tinnitus_pitch=profile_pitch,
                                            trials_df=patient_trial_history
                                        )
                                    )

                                    ai_model = get_cached_ai_model()

                                    predictions = predict_responses_batch(
                                        patient_features,
                                        candidate_stimuli,
                                        model=ai_model
                                    )

                                    ranked_candidates = []

                                    for candidate, prediction in zip(
                                        candidate_stimuli,
                                        predictions
                                    ):

                                        ranked_candidates.append(
                                            {
                                                "stimulus_type": candidate[
                                                    "stimulus_type"
                                                ],
                                                "frequency": float(
                                                    candidate[
                                                        "center_frequency_hz"
                                                    ]
                                                ),
                                                "bandwidth": (
                                                    float(
                                                        candidate[
                                                            "bandwidth_hz"
                                                        ]
                                                    )
                                                    if candidate.get(
                                                        "bandwidth_hz"
                                                    ) is not None
                                                    else None
                                                ),
                                                "level": float(
                                                    candidate[
                                                        "relative_level"
                                                    ]
                                                ),
                                                "predicted_score": float(
                                                    prediction
                                                )
                                            }
                                        )

                                    ranked_candidates.sort(
                                        key=lambda item: item[
                                            "predicted_score"
                                        ],
                                        reverse=True
                                    )

                                    if not ranked_candidates:
                                        raise ValueError(
                                            "The AI did not produce any candidate sounds."
                                        )

                                    st.session_state[
                                        "patient_ai_recommendation"
                                    ] = ranked_candidates[0]

                                    st.session_state[
                                        "patient_ai_top_candidates"
                                    ] = ranked_candidates[:5]

                                    st.session_state[
                                        "patient_ai_recommendation_created_at"
                                    ] = now_timestamp()

                                    st.session_state[
                                        "patient_ai_result_trial_id"
                                    ] = None

                                    st.session_state[
                                        "patient_ai_observed_score"
                                    ] = None

                                    st.rerun()

                                except Exception as error:

                                    st.error(
                                        f"AI personalization could not generate a recommendation: {error}"
                                    )

                        else:

                            st.markdown(
                                "### AI-Recommended Candidate"
                            )

                            a1, a2, a3, a4 = st.columns(
                                4
                            )

                            a1.metric(
                                "Stimulus",
                                ai_recommendation[
                                    "stimulus_type"
                                ]
                            )

                            a2.metric(
                                "Frequency",
                                (
                                    "Broadband"
                                    if ai_recommendation[
                                        "stimulus_type"
                                    ] == "White Noise"
                                    else f"{int(round(ai_recommendation['frequency']))} Hz"
                                )
                            )

                            a3.metric(
                                "Relative Level",
                                f"{ai_recommendation['level']:.2f}"
                            )

                            a4.metric(
                                "AI Predicted Response",
                                f"{ai_recommendation['predicted_score']:.1f} / 100"
                            )

                            if (
                                ai_recommendation[
                                    "stimulus_type"
                                ] == "Narrow-Band Noise"
                                and ai_recommendation.get(
                                    "bandwidth"
                                ) is not None
                            ):

                                st.metric(
                                    "Bandwidth",
                                    f"{int(round(ai_recommendation['bandwidth']))} Hz"
                                )

                            top_candidates = st.session_state.get(
                                "patient_ai_top_candidates",
                                []
                            )

                            if top_candidates:

                                with st.expander(
                                    "View AI predicted ranking"
                                ):

                                    ranking_rows = []

                                    for index, candidate in enumerate(
                                        top_candidates,
                                        start=1
                                    ):

                                        ranking_rows.append(
                                            {
                                                "Rank": index,
                                                "Stimulus": candidate[
                                                    "stimulus_type"
                                                ],
                                                "Frequency": (
                                                    "Broadband"
                                                    if candidate[
                                                        "stimulus_type"
                                                    ] == "White Noise"
                                                    else f"{int(round(candidate['frequency']))} Hz"
                                                ),
                                                "Bandwidth": (
                                                    f"{int(round(candidate['bandwidth']))} Hz"
                                                    if candidate.get(
                                                        "bandwidth"
                                                    ) is not None
                                                    else "N/A"
                                                ),
                                                "Relative Level": round(
                                                    candidate[
                                                        "level"
                                                    ],
                                                    2
                                                ),
                                                "Predicted Score": round(
                                                    candidate[
                                                        "predicted_score"
                                                    ],
                                                    1
                                                )
                                            }
                                        )

                                    st.dataframe(
                                        pd.DataFrame(
                                            ranking_rows
                                        ),
                                        use_container_width=True,
                                        hide_index=True
                                    )

                            try:

                                st.audio(
                                    generate_stimulus(
                                        stimulus_type=ai_recommendation[
                                            "stimulus_type"
                                        ],
                                        frequency=float(
                                            ai_recommendation[
                                                "frequency"
                                            ]
                                        ),
                                        duration=2,
                                        volume=float(
                                            ai_recommendation[
                                                "level"
                                            ]
                                        ),
                                        bandwidth=float(
                                            ai_recommendation.get(
                                                "bandwidth"
                                            )
                                            or 500.0
                                        )
                                    ),
                                    format="audio/wav"
                                )

                            except Exception as error:

                                st.error(
                                    f"Could not generate the AI-recommended sound: {error}"
                                )

                            ai_result_trial_id = st.session_state.get(
                                "patient_ai_result_trial_id"
                            )

                            if ai_result_trial_id is None:

                                st.markdown(
                                    "#### Test the AI candidate and give real feedback"
                                )

                                f1, f2 = st.columns(
                                    2
                                )

                                ai_comfort = f1.slider(
                                    "Comfort",
                                    0,
                                    10,
                                    5,
                                    key="patient_ai_comfort"
                                )

                                ai_noticeability = f2.slider(
                                    "Tinnitus noticeability",
                                    0,
                                    10,
                                    5,
                                    key="patient_ai_noticeability"
                                )

                                ai_again = st.radio(
                                    "Would you listen to this AI-recommended sound again?",
                                    [
                                        "Yes",
                                        "No"
                                    ],
                                    horizontal=True,
                                    key="patient_ai_again"
                                )

                                if st.button(
                                    "Save AI-Path Feedback",
                                    use_container_width=True,
                                    key="patient_save_ai_feedback"
                                ):

                                    observed_score = calculate_response_score(
                                        ai_comfort,
                                        ai_noticeability,
                                        ai_again
                                    )

                                    provenance = get_model_provenance(
                                        MODEL_FILE,
                                        get_saved_model_metrics()
                                    )

                                    saved_bandwidth = (
                                        float(
                                            ai_recommendation[
                                                "bandwidth"
                                            ]
                                        )
                                        if ai_recommendation[
                                            "stimulus_type"
                                        ] == "Narrow-Band Noise"
                                        and ai_recommendation.get(
                                            "bandwidth"
                                        ) is not None
                                        else None
                                    )

                                    cursor.execute(
                                        """
                                        INSERT INTO stimulus_trials
                                        (
                                            patient_id,
                                            session_id,
                                            profile_id,
                                            source_type,
                                            model_version,
                                            model_trained_at,
                                            model_predicted_score,
                                            recommendation_created_at,
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
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        """,
                                        (
                                            patient_id,
                                            None,
                                            profile_id,
                                            "patient_ai_result",
                                            provenance[
                                                "model_version"
                                            ],
                                            provenance[
                                                "model_trained_at"
                                            ],
                                            float(
                                                ai_recommendation[
                                                    "predicted_score"
                                                ]
                                            ),
                                            st.session_state.get(
                                                "patient_ai_recommendation_created_at"
                                            )
                                            or now_timestamp(),
                                            now_timestamp(),
                                            ai_recommendation[
                                                "stimulus_type"
                                            ],
                                            float(
                                                ai_recommendation[
                                                    "frequency"
                                                ]
                                            ),
                                            saved_bandwidth,
                                            float(
                                                ai_recommendation[
                                                    "level"
                                                ]
                                            ),
                                            int(
                                                ai_comfort
                                            ),
                                            int(
                                                ai_noticeability
                                            ),
                                            ai_again,
                                            now_timestamp()
                                        )
                                    )

                                    conn.commit()

                                    ai_result_trial_id = int(
                                        cursor.lastrowid
                                    )

                                    st.session_state[
                                        "patient_ai_result_trial_id"
                                    ] = ai_result_trial_id

                                    st.session_state[
                                        "patient_ai_observed_score"
                                    ] = float(
                                        observed_score
                                    )

                                    create_personalization_review_request(
                                        conn,
                                        patient_id,
                                        profile_id,
                                        "ai",
                                        ai_result_trial_id
                                    )

                                    st.rerun()

                            else:

                                observed_score = st.session_state.get(
                                    "patient_ai_observed_score"
                                )

                                if observed_score is None:

                                    cursor.execute(
                                        """
                                        SELECT
                                            comfort_score,
                                            tinnitus_noticeability,
                                            listen_again
                                        FROM stimulus_trials
                                        WHERE id = ?
                                        """,
                                        (
                                            ai_result_trial_id,
                                        )
                                    )

                                    observed_row = cursor.fetchone()

                                    if observed_row is not None:

                                        observed_score = calculate_response_score(
                                            observed_row[0],
                                            observed_row[1],
                                            observed_row[2]
                                        )

                                st.success(
                                    "The AI candidate has now been tested. The prediction "
                                    "and your observed response remain separate."
                                )

                                c1, c2 = st.columns(
                                    2
                                )

                                c1.metric(
                                    "AI Predicted Response",
                                    f"{ai_recommendation['predicted_score']:.1f} / 100"
                                )

                                c2.metric(
                                    "Observed Patient Response",
                                    (
                                        f"{float(observed_score):.1f} / 100"
                                        if observed_score is not None
                                        else "N/A"
                                    )
                                )

                                review_row = get_personalization_review(
                                    conn,
                                    "ai",
                                    ai_result_trial_id
                                )

                                ai_status = (
                                    review_row[0]
                                    if review_row is not None
                                    else "Pending Review"
                                )

                                st.markdown("---")
                                st.markdown(
                                    "### 👨‍⚕️ AI Path — Professional Review"
                                )

                                if ai_status == "Approved":
                                    st.success(
                                        "✅ AI Professional Status: Approved"
                                    )
                                elif ai_status == "Modified":
                                    st.warning(
                                        "✏️ AI Professional Status: Modified"
                                    )
                                elif ai_status == "Rejected":
                                    st.error(
                                        "❌ AI Professional Status: Rejected"
                                    )
                                else:
                                    st.info(
                                        "⏳ AI Professional Status: Pending Review"
                                    )

                                st.caption(
                                    "This status belongs only to the AI pathway. "
                                    "It does not determine the Standard pathway status."
                                )

                            c1, c2 = st.columns(
                                2
                            )

                            with c1:

                                st.button(
                                    "↻ Start a New AI Recommendation",
                                    use_container_width=True,
                                    on_click=reset_patient_ai_personalization_state,
                                    key="patient_ai_new_recommendation"
                                )

                            with c2:

                                st.button(
                                    "Compare Standard & AI Results",
                                    use_container_width=True,
                                    on_click=set_patient_personalization_path,
                                    args=(
                                        "Results Comparison",
                                    ),
                                    key="patient_ai_view_status"
                                )

                # =========================================================
                # RESULTS + PROFESSIONAL STATUS FOR BOTH PATHWAYS
                # =========================================================

                elif personalization_path == "Results Comparison":

                    st.button(
                        "← Back to Personalization Choices",
                        on_click=set_patient_personalization_path,
                        args=(
                            None,
                        ),
                        key="patient_results_back_to_choices"
                    )

                    st.markdown(
                        "## Compare Standard & AI Results"
                    )

                    st.write(
                        "This page only compares the two pathways side by side. "
                        "It does not create one shared professional status. "
                        "Standard keeps its own professional decision and AI keeps its own."
                    )

                    # ---------------------------------
                    # Latest Standard result
                    # ---------------------------------

                    cursor.execute(
                        """
                        SELECT
                            id,
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
                          AND source_type = 'patient_standard_result'
                        ORDER BY id DESC
                        LIMIT 1
                        """,
                        (
                            patient_id,
                        )
                    )

                    standard_row = cursor.fetchone()

                    # ---------------------------------
                    # Latest AI result
                    # ---------------------------------

                    cursor.execute(
                        """
                        SELECT
                            id,
                            stimulus_type,
                            center_frequency_hz,
                            bandwidth_hz,
                            relative_level,
                            model_predicted_score,
                            comfort_score,
                            tinnitus_noticeability,
                            listen_again,
                            created_at
                        FROM stimulus_trials
                        WHERE patient_id = ?
                          AND source_type IN ('audiologist_ai_result', 'patient_ai_result')
                        ORDER BY id DESC
                        LIMIT 1
                        """,
                        (
                            patient_id,
                        )
                    )

                    ai_row = cursor.fetchone()

                    c1, c2 = st.columns(
                        2
                    )

                    with c1:

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                "### Standard Personalization"
                            )

                            if standard_row is None:

                                st.caption(
                                    "No completed Standard result yet."
                                )

                            else:

                                (
                                    standard_trial_id,
                                    standard_type,
                                    standard_frequency,
                                    standard_bandwidth,
                                    standard_level,
                                    standard_comfort,
                                    standard_noticeability,
                                    standard_again,
                                    standard_created_at
                                ) = standard_row

                                standard_observed = calculate_response_score(
                                    standard_comfort,
                                    standard_noticeability,
                                    standard_again
                                )

                                st.metric(
                                    "Best-Rated Candidate",
                                    standard_type
                                )

                                st.write(
                                    "**Frequency:**",
                                    (
                                        "Broadband"
                                        if standard_type == "White Noise"
                                        else f"{int(round(float(standard_frequency)))} Hz"
                                    )
                                )

                                if (
                                    standard_type == "Narrow-Band Noise"
                                    and standard_bandwidth is not None
                                ):

                                    st.write(
                                        "**Bandwidth:**",
                                        f"{int(round(float(standard_bandwidth)))} Hz"
                                    )

                                st.write(
                                    "**Observed response:**",
                                    f"{float(standard_observed):.1f} / 100"
                                )

                                standard_review = get_personalization_review(
                                    conn,
                                    "standard",
                                    standard_trial_id
                                )

                                if standard_review is None:

                                    create_personalization_review_request(
                                        conn,
                                        patient_id,
                                        profile_id,
                                        "standard",
                                        standard_trial_id
                                    )

                                    standard_review = get_personalization_review(
                                        conn,
                                        "standard",
                                        standard_trial_id
                                    )

                                standard_status = (
                                    standard_review[0]
                                    if standard_review is not None
                                    else "Pending Review"
                                )

                                st.info(
                                    f"Standard Professional Status: **{standard_status}**"
                                )

                                if (
                                    standard_review is not None
                                    and standard_review[2] is not None
                                ):

                                    st.success(
                                        "A professional final configuration has been recorded."
                                    )

                                    st.write(
                                        "**Final sound:**",
                                        standard_review[2]
                                    )

                    with c2:

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                "### AI Personalization"
                            )

                            if ai_row is None:

                                st.caption(
                                    "No completed AI result yet. AI remains optional."
                                )

                            else:

                                (
                                    ai_trial_id,
                                    ai_type,
                                    ai_frequency,
                                    ai_bandwidth,
                                    ai_level,
                                    ai_predicted,
                                    ai_comfort,
                                    ai_noticeability,
                                    ai_again,
                                    ai_created_at
                                ) = ai_row

                                ai_observed = calculate_response_score(
                                    ai_comfort,
                                    ai_noticeability,
                                    ai_again
                                )

                                st.metric(
                                    "AI-Recommended Candidate",
                                    ai_type
                                )

                                st.write(
                                    "**Frequency:**",
                                    (
                                        "Broadband"
                                        if ai_type == "White Noise"
                                        else f"{int(round(float(ai_frequency)))} Hz"
                                    )
                                )

                                if (
                                    ai_type == "Narrow-Band Noise"
                                    and ai_bandwidth is not None
                                ):

                                    st.write(
                                        "**Bandwidth:**",
                                        f"{int(round(float(ai_bandwidth)))} Hz"
                                    )

                                st.write(
                                    "**AI predicted response:**",
                                    (
                                        f"{float(ai_predicted):.1f} / 100"
                                        if ai_predicted is not None
                                        else "N/A"
                                    )
                                )

                                st.write(
                                    "**Observed patient response:**",
                                    f"{float(ai_observed):.1f} / 100"
                                )

                                ai_review = get_personalization_review(
                                    conn,
                                    "ai",
                                    ai_trial_id
                                )

                                if ai_review is None:

                                    create_personalization_review_request(
                                        conn,
                                        patient_id,
                                        profile_id,
                                        "ai",
                                        ai_trial_id
                                    )

                                    ai_review = get_personalization_review(
                                        conn,
                                        "ai",
                                        ai_trial_id
                                    )

                                ai_status = (
                                    ai_review[0]
                                    if ai_review is not None
                                    else "Pending Review"
                                )

                                st.info(
                                    f"AI Professional Status: **{ai_status}**"
                                )

                                if (
                                    ai_review is not None
                                    and ai_review[2] is not None
                                ):

                                    st.success(
                                        "A professional final configuration has been recorded."
                                    )

                                    st.write(
                                        "**Final sound:**",
                                        ai_review[2]
                                    )

                    st.caption(
                        "Observed response scores are prototype weighted feedback scores. "
                        "AI predicted scores and observed patient scores are deliberately "
                        "shown separately. Neither is a validated clinical outcome measure."
                    )

                    st.markdown("---")
                    st.markdown(
                        "### Independent Professional Reviews"
                    )

                    st.write(
                        "The Standard result and the AI result are reviewed independently. "
                        "Each one can separately be Pending Review, Approved, Modified or "
                        "Rejected, and each can have a different clinician-reviewed final "
                        "configuration."
                    )

                    st.info(
                        "If both pathways are completed, the professional can compare the "
                        "Standard observed result with the AI prediction + observed result "
                        "before making a decision."
                    )

                    # ---------------------------------
                    # Preserve existing clinician-reviewed sound
                    # from the current audiologist workflow.
                    # ---------------------------------

                    cursor.execute(
                        """
                        SELECT
                            ad.id,
                            COALESCE(ad.final_stimulus_type, ad.ai_stimulus_type),
                            COALESCE(ad.final_frequency_hz, ad.ai_frequency_hz),
                            COALESCE(ad.final_bandwidth_hz, ad.ai_bandwidth_hz),
                            COALESCE(ad.final_relative_level, ad.ai_relative_level),
                            ad.decision
                        FROM audiologist_decisions ad
                        WHERE ad.patient_id = ?
                          AND ad.decision IN ('Approve', 'Modify')
                        ORDER BY ad.id DESC
                        LIMIT 1
                        """,
                        (
                            patient_id,
                        )
                    )

                    existing_clinician_row = cursor.fetchone()

                    if existing_clinician_row is not None:

                        (
                            existing_decision_id,
                            existing_type,
                            existing_frequency,
                            existing_bandwidth,
                            existing_level,
                            existing_decision
                        ) = existing_clinician_row

                        st.success(
                            "An existing clinician-reviewed configuration is also available "
                            "from the current professional workflow."
                        )

                        e1, e2, e3 = st.columns(
                            3
                        )

                        e1.metric(
                            "Clinician Decision",
                            existing_decision
                        )

                        e2.metric(
                            "Sound",
                            existing_type
                        )

                        e3.metric(
                            "Frequency",
                            (
                                "Broadband"
                                if existing_type == "White Noise"
                                else f"{int(round(float(existing_frequency)))} Hz"
                            )
                        )

                        try:

                            st.audio(
                                generate_stimulus(
                                    stimulus_type=existing_type,
                                    frequency=float(
                                        existing_frequency
                                    ),
                                    duration=2,
                                    volume=float(
                                        existing_level
                                    ),
                                    bandwidth=float(
                                        existing_bandwidth
                                        if existing_bandwidth is not None
                                        else 500.0
                                    )
                                ),
                                format="audio/wav"
                            )

                        except Exception as error:

                            st.error(
                                f"Could not generate the clinician-reviewed sound: {error}"
                            )

                    nav1, nav2 = st.columns(
                        2
                    )

                    with nav1:

                        st.button(
                            "🎧 My Tinnitus Profile",
                            use_container_width=True,
                            on_click=navigate_to_page,
                            args=(
                                "My Tinnitus Profile",
                            ),
                            key="patient_personalization_results_profile"
                        )

                    with nav2:

                        st.button(
                            "🏠 Return Home",
                            use_container_width=True,
                            on_click=navigate_to_page,
                            args=(
                                "Home",
                            ),
                            key="patient_personalization_results_home"
                        )

    # =================================================
    # AUDIOLOGIST — TECHNICAL PERSONALIZATION WORKSPACE
    # =================================================

    else:

        st.title(
            "Acoustic Personalization"
        )

        acoustic_review_flash = st.session_state.pop(
            "audiologist_review_flash",
            None
        )

        if acoustic_review_flash:

            st.success(
                acoustic_review_flash
            )


        # -------------------------------------------------
        # SELECT ONE PATIENT FOR THE ENTIRE WORKSPACE
        # -------------------------------------------------

        cursor.execute(
            """
            SELECT id, patient_code
            FROM patients
            ORDER BY patient_code
            """
        )

        acoustic_patient_rows = cursor.fetchall()

        if not acoustic_patient_rows:

            st.warning(
                "No patients are available."
            )

        else:

            acoustic_patient_id = get_active_patient_id()
            acoustic_patient_code = get_active_patient_code()

            if acoustic_patient_id is None:
                st.warning("Open a patient from Patients first.")
                st.stop()

            # -------------------------------------------------
            # PATIENT CONTEXT
            # -------------------------------------------------

            acoustic_profile = conn.execute(
                """
                SELECT
                    id,
                    estimated_pitch_hz,
                    loudness_relative,
                    tinnitus_type,
                    laterality,
                    temporal_pattern,
                    duration_months,
                    annoyance_score,
                    sleep_score,
                    concentration_score,
                    stress_score,
                    created_at
                FROM tinnitus_profiles
                WHERE patient_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (
                    acoustic_patient_id,
                )
            ).fetchone()

            acoustic_profile_id = (
                acoustic_profile[0]
                if acoustic_profile is not None
                else None
            )

            acoustic_pitch = (
                float(acoustic_profile[1])
                if (
                    acoustic_profile is not None
                    and
                    acoustic_profile[1] is not None
                )
                else None
            )

            acoustic_loudness = (
                float(acoustic_profile[2])
                if (
                    acoustic_profile is not None
                    and
                    acoustic_profile[2] is not None
                )
                else None
            )

            acoustic_type = acoustic_profile[3] if acoustic_profile is not None else None
            acoustic_laterality = acoustic_profile[4] if acoustic_profile is not None else None
            acoustic_pattern = acoustic_profile[5] if acoustic_profile is not None else None
            acoustic_duration = acoustic_profile[6] if acoustic_profile is not None else None
            acoustic_annoyance = acoustic_profile[7] if acoustic_profile is not None else None
            acoustic_sleep = acoustic_profile[8] if acoustic_profile is not None else None
            acoustic_concentration = acoustic_profile[9] if acoustic_profile is not None else None
            acoustic_stress = acoustic_profile[10] if acoustic_profile is not None else None
            acoustic_profile_date = acoustic_profile[11] if acoustic_profile is not None else None

            audiogram_row = conn.execute(
                """
                SELECT
                    COUNT(*),
                    MAX(created_at)
                FROM audiograms
                WHERE patient_id = ?
                """,
                (
                    acoustic_patient_id,
                )
            ).fetchone()

            has_audiogram = (
                audiogram_row is not None
                and
                int(
                    audiogram_row[0]
                    or
                    0
                ) > 0
            )

            trial_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM stimulus_trials
                WHERE patient_id = ?
                """,
                (
                    acoustic_patient_id,
                )
            ).fetchone()[0]

            standard_context_trial = get_latest_personalization_trial(
                conn,
                acoustic_patient_id,
                "standard"
            )

            ai_context_trial = get_latest_personalization_trial(
                conn,
                acoustic_patient_id,
                "ai"
            )

            context_1, context_2, context_3, context_4 = st.columns(
                4
            )

            context_1.metric(
                "Patient",
                acoustic_patient_code
            )

            context_2.metric(
                "Matched Tinnitus Pitch",
                (
                    f"{int(round(acoustic_pitch))} Hz"
                    if acoustic_pitch is not None
                    else "Not available"
                )
            )

            context_3.metric(
                "Relative Loudness",
                (
                    f"{acoustic_loudness:.02f}"
                    if acoustic_loudness is not None
                    else "Not available"
                )
            )

            context_4.metric(
                "Audiogram",
                (
                    "✅ Available"
                    if has_audiogram
                    else "⚪ Not available"
                )
            )

            with st.expander("📋 Review Full Patient Profile", expanded=True):

                if acoustic_profile is None:
                    st.warning("No tinnitus profile is available for this patient.")
                else:
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Laterality", acoustic_laterality or "N/A")
                    p2.metric("Tinnitus Type", acoustic_type or "N/A")
                    p3.metric("Pattern", acoustic_pattern or "N/A")
                    p4.metric(
                        "Duration",
                        f"{acoustic_duration} months" if acoustic_duration is not None else "N/A"
                    )

                    i1, i2, i3, i4 = st.columns(4)
                    i1.metric("Annoyance", f"{acoustic_annoyance}/10" if acoustic_annoyance is not None else "N/A")
                    i2.metric("Sleep Impact", f"{acoustic_sleep}/10" if acoustic_sleep is not None else "N/A")
                    i3.metric("Concentration", f"{acoustic_concentration}/10" if acoustic_concentration is not None else "N/A")
                    i4.metric("Stress", f"{acoustic_stress}/10" if acoustic_stress is not None else "N/A")

            st.markdown("---")

            if (
                "audiologist_acoustic_mode"
                not in st.session_state
            ):
                st.session_state[
                    "audiologist_acoustic_mode"
                ] = "🎛️ Candidate & Assignment"

            acoustic_mode = st.radio(
                "Technical view",
                [
                    "🎛️ Candidate & Assignment",
                    "💬 Patient Feedback"
                ],
                horizontal=True,
                key="audiologist_acoustic_mode"
            )

            # =================================================
            # 1. CANDIDATE GENERATOR
            # =================================================

            if acoustic_mode == "🎛️ Candidate & Assignment":

                st.subheader("Prepare and Send Acoustic Candidate")
                st.write(
                    "Choose how the sound should be prepared. A manually configured sound can be "
                    "previewed and sent directly. An AI-assisted proposal must be reviewed and "
                    "approved by the audiologist before it can be sent to the patient."
                )

                if acoustic_profile is None:
                    st.error(
                        "No tinnitus profile is available for this patient. Complete or request "
                        "the patient assessment before preparing an acoustic test."
                    )
                    st.stop()

                if acoustic_pitch is None:
                    st.warning(
                        "The patient has no saved matched pitch. A default frequency can be previewed, "
                        "but the configuration should be interpreted cautiously."
                    )

                recommendation_method = st.radio(
                    "Sound generation method",
                    ["🩺 Audiologist Manual Configuration", "🤖 AI-Assisted Generation"],
                    horizontal=True,
                    key=f"audiologist_recommendation_method_{acoustic_patient_id}"
                )

                default_frequency = float(acoustic_pitch if acoustic_pitch is not None else 4000.0)
                default_level = max(
                    0.01, min(0.10, float(acoustic_loudness if acoustic_loudness is not None else 0.03))
                )

                selected_pathway = "standard"
                candidate_predicted_score = None
                ai_recommendation_created_at = None
                provenance = None
                ai_approved = True

                if recommendation_method.startswith("🩺"):
                    st.markdown("### Audiologist-configured candidate")
                    st.caption("Configure the parameters, preview the sound, then send it to the patient.")

                    generator_type = st.selectbox(
                        "Stimulus Type", ["Pure Tone", "Narrow-Band Noise", "White Noise"],
                        key=f"audiologist_manual_type_{acoustic_patient_id}"
                    )
                    generator_frequency = None
                    if generator_type != "White Noise":
                        generator_frequency = st.slider(
                            "Frequency (Hz)", 250, 10000,
                            int(max(250, min(10000, round(default_frequency / 50.0) * 50))),
                            50, key=f"audiologist_manual_frequency_{acoustic_patient_id}"
                        )
                    generator_bandwidth = None
                    if generator_type == "Narrow-Band Noise":
                        generator_bandwidth = st.slider(
                            "Bandwidth (Hz)", 100, 2000, 500, 50,
                            key=f"audiologist_manual_bandwidth_{acoustic_patient_id}"
                        )
                    generator_level = st.slider(
                        "Relative Digital Level", 0.01, 0.10, default_level, 0.01,
                        key=f"audiologist_manual_level_{acoustic_patient_id}"
                    )

                else:
                    selected_pathway = "ai"
                    recommendation_key = f"audiologist_ai_recommendation_{acoustic_patient_id}"
                    ranking_key = f"audiologist_ai_ranking_{acoustic_patient_id}"
                    index_key = f"audiologist_ai_candidate_index_{acoustic_patient_id}"
                    created_key = f"audiologist_ai_created_at_{acoustic_patient_id}"
                    approval_key = f"audiologist_ai_approved_{acoustic_patient_id}"

                    st.markdown("### AI-assisted candidate")
                    st.caption(
                        "The AI ranks candidate configurations. Nothing is sent until the audiologist approves one."
                    )

                    if not model_exists():
                        st.warning("No trained AI model is available. Use Manual Configuration or train the model first.")
                        st.stop()

                    if st.button("🤖 Generate AI Candidates", use_container_width=True, key=f"audiologist_generate_ai_{acoustic_patient_id}"):
                        try:
                            full_profile = conn.execute(
                                """
                                SELECT tinnitus_type, laterality, temporal_pattern, duration_months,
                                       annoyance_score, sleep_score, concentration_score, stress_score,
                                       estimated_pitch_hz, loudness_relative
                                FROM tinnitus_profiles
                                WHERE patient_id = ?
                                ORDER BY id DESC LIMIT 1
                                """,
                                (acoustic_patient_id,)
                            ).fetchone()
                            if full_profile is None:
                                raise ValueError("No tinnitus profile is available for this patient.")

                            patient_features = {
                                "tinnitus_type": full_profile[0], "laterality": full_profile[1],
                                "temporal_pattern": full_profile[2], "duration_months": full_profile[3],
                                "annoyance_score": full_profile[4], "sleep_score": full_profile[5],
                                "concentration_score": full_profile[6], "stress_score": full_profile[7],
                                "estimated_pitch_hz": full_profile[8], "loudness_relative": full_profile[9],
                            }
                            history_df = get_patient_trial_history_for_adaptation(conn, acoustic_patient_id)
                            candidate_stimuli, _ = generate_adaptive_candidate_stimuli(
                                tinnitus_pitch=(float(full_profile[8]) if full_profile[8] is not None else default_frequency),
                                trials_df=history_df
                            )
                            predictions = predict_responses_batch(
                                patient_features, candidate_stimuli, model=get_cached_ai_model()
                            )
                            ranked_candidates = []
                            for candidate, prediction in zip(candidate_stimuli, predictions):
                                ranked_candidates.append({
                                    "stimulus_type": candidate["stimulus_type"],
                                    "frequency": float(candidate["center_frequency_hz"]),
                                    "bandwidth": (float(candidate["bandwidth_hz"]) if candidate.get("bandwidth_hz") is not None else None),
                                    "level": float(candidate["relative_level"]),
                                    "predicted_score": float(prediction),
                                })
                            ranked_candidates.sort(key=lambda item: item["predicted_score"], reverse=True)
                            if not ranked_candidates:
                                raise ValueError("The AI model did not produce candidate configurations.")
                            st.session_state[ranking_key] = ranked_candidates[:5]
                            st.session_state[index_key] = 0
                            st.session_state[recommendation_key] = ranked_candidates[0]
                            st.session_state[created_key] = now_timestamp()
                            st.session_state[approval_key] = False
                            st.rerun()
                        except Exception as error:
                            st.error(f"AI candidates could not be generated: {error}")

                    top_candidates = st.session_state.get(ranking_key, [])
                    if not top_candidates:
                        st.info("Generate AI candidates to start.")
                        st.stop()

                    current_index = int(st.session_state.get(index_key, 0))
                    current_index = max(0, min(current_index, len(top_candidates) - 1))
                    ai_recommendation = top_candidates[current_index]
                    st.session_state[recommendation_key] = ai_recommendation
                    ai_recommendation_created_at = st.session_state.get(created_key)
                    candidate_predicted_score = float(ai_recommendation["predicted_score"])

                    generator_type = ai_recommendation["stimulus_type"]
                    generator_frequency = None if generator_type == "White Noise" else float(ai_recommendation["frequency"])
                    generator_bandwidth = ai_recommendation.get("bandwidth")
                    generator_level = float(ai_recommendation["level"])

                    ai1, ai2, ai3, ai4 = st.columns(4)
                    ai1.metric("Candidate", generator_type)
                    ai2.metric("Frequency", "Broadband" if generator_type == "White Noise" else f"{generator_frequency:.0f} Hz")
                    ai3.metric("Relative Level", f"{generator_level:.02f}")
                    ai4.metric("Predicted Response", f"{candidate_predicted_score:.1f} / 100")

                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("✅ Approve AI Candidate", use_container_width=True, key=f"approve_ai_before_send_{acoustic_patient_id}_{current_index}"):
                            st.session_state[approval_key] = True
                            st.rerun()
                    with b2:
                        if st.button("🔄 Show Another AI Candidate", use_container_width=True, disabled=len(top_candidates) <= 1, key=f"next_ai_candidate_{acoustic_patient_id}_{current_index}"):
                            st.session_state[index_key] = (current_index + 1) % len(top_candidates)
                            st.session_state[approval_key] = False
                            st.rerun()

                    ai_approved = bool(st.session_state.get(approval_key, False))
                    if ai_approved:
                        st.success("Approved by audiologist — ready to send.")
                    else:
                        st.info("Approve this AI candidate or show another AI candidate before sending.")

                    provenance = get_model_provenance(MODEL_FILE, get_saved_model_metrics())

                st.markdown("### Preview")
                duration_seconds = st.slider(
                    "Patient test duration (seconds)", 5, 60, 20, 5,
                    key=f"assignment_duration_{acoustic_patient_id}_{selected_pathway}"
                )
                try:
                    st.audio(
                        generate_stimulus(
                            stimulus_type=generator_type,
                            frequency=(float(generator_frequency) if generator_frequency is not None else 1000.0),
                            duration=min(int(duration_seconds), 10),
                            volume=float(generator_level),
                            bandwidth=(float(generator_bandwidth) if generator_bandwidth is not None else 500.0)
                        ),
                        format="audio/wav"
                    )
                except Exception as error:
                    st.error(f"Could not generate this stimulus: {error}")

                audiologist_note = st.text_area(
                    "Optional note for the patient",
                    placeholder="Example: Please listen in a quiet room and stop if uncomfortable.",
                    key=f"assignment_note_{acoustic_patient_id}_{selected_pathway}"
                )
                send_disabled = selected_pathway == "ai" and not ai_approved
                if st.button(
                    "📨 Send to Patient", use_container_width=True, disabled=send_disabled,
                    key=f"send_candidate_to_patient_{acoustic_patient_id}_{selected_pathway}"
                ):
                    try:
                        assignment_id = backend_create_acoustic_assignment(
                            patient_id=acoustic_patient_id, profile_id=acoustic_profile_id,
                            pathway=selected_pathway, stimulus_type=generator_type,
                            center_frequency_hz=(float(generator_frequency) if generator_frequency is not None else None),
                            bandwidth_hz=(float(generator_bandwidth) if generator_bandwidth is not None else None),
                            relative_level=float(generator_level), duration_seconds=int(duration_seconds),
                            predicted_score=(float(candidate_predicted_score) if candidate_predicted_score is not None else None),
                            model_version=(provenance.get("model_version") if provenance else None),
                            model_trained_at=(provenance.get("model_trained_at") if provenance else None),
                            recommendation_created_at=(ai_recommendation_created_at if selected_pathway == "ai" else now_timestamp()),
                            audiologist_note=audiologist_note,
                        )
                        st.success(
                            f"Candidate #{assignment_id} sent to the patient. "
                            "It will appear in Assigned Acoustic Test."
                        )
                    except Exception as error:
                        st.error(f"Could not send the candidate: {error}")

            elif acoustic_mode == "💬 Patient Feedback":

                st.subheader("Patient Feedback")
                st.write(
                    "Review feedback returned after the patient listened to an audiologist-sent sound. "
                    "Use it to decide whether another candidate should be prepared."
                )
                feedback_rows = backend_list_acoustic_feedback(acoustic_patient_id, limit=20)
                if not feedback_rows:
                    st.info("No patient feedback has been returned for assigned acoustic tests yet.")
                else:
                    latest = feedback_rows[0]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Stimulus", latest["stimulus_type"])
                    c2.metric("Comfort", f"{latest['comfort_score']}/10")
                    c3.metric("Noticeability", f"{latest['tinnitus_noticeability']}/10")
                    c4.metric("Listen Again", latest["listen_again"])
                    if latest.get("patient_comment"):
                        st.info(f"Patient comment: {latest['patient_comment']}")
                    st.caption(
                        "There is no extra Approve / Modify / Reject step after feedback. "
                        "If adjustment is needed, return to Candidate & Assignment and prepare another sound."
                    )
                    feedback_df = pd.DataFrame(feedback_rows).rename(columns={
                        "pathway": "Method", "stimulus_type": "Stimulus",
                        "center_frequency_hz": "Frequency (Hz)", "relative_level": "Relative Level",
                        "comfort_score": "Comfort /10", "tinnitus_noticeability": "Noticeability /10",
                        "listen_again": "Listen Again", "patient_comment": "Patient Comment",
                        "responded_at": "Feedback Received",
                    })
                    st.dataframe(
                        feedback_df[["Method", "Stimulus", "Frequency (Hz)", "Relative Level",
                                     "Comfort /10", "Noticeability /10", "Listen Again",
                                     "Patient Comment", "Feedback Received"]],
                        use_container_width=True, hide_index=True
                    )

            # =================================================
            # 2. OBSERVED RESPONSES + RANKING
            # =================================================

            elif acoustic_mode == "📊 Observed Evidence":

                st.subheader(
                    "Observed Evidence"
                )

                st.write(
                    "This view contains the configurations the patient actually tested under supervision and the feedback that was recorded. "
                    "Observed evidence remains separate from AI predictions."
                )

                observed_rows = conn.execute(
                    """
                    SELECT
                        id,
                        source_type,
                        stimulus_type,
                        center_frequency_hz,
                        bandwidth_hz,
                        relative_level,
                        comfort_score,
                        tinnitus_noticeability,
                        listen_again,
                        COALESCE(tested_at, created_at)
                    FROM stimulus_trials
                    WHERE patient_id = ?
                      AND comfort_score IS NOT NULL
                      AND tinnitus_noticeability IS NOT NULL
                      AND source_type NOT IN ('patient_standard_result')
                    ORDER BY id ASC
                    """,
                    (
                        acoustic_patient_id,
                    )
                ).fetchall()

                if not observed_rows:

                    st.info(
                        "No observed acoustic responses are available for this patient yet."
                    )

                else:

                    observed_records = []

                    source_labels = {
                        "patient_personalization_initial": "Standard · Initial",
                        "patient_personalization_refined": "Standard · Refined",
                        "patient_ai_result": "Legacy AI · Patient-Tested",
                        "audiologist_ai_result": "Audiologist AI-Assisted · Supervised",
                        "audiologist_manual_result": "Audiologist Manual · Supervised",
                        "audiologist_supervised_manual": "Audiologist · Supervised",
                        "manual": "Manual Trial",
                    }

                    for (
                        trial_id,
                        source_type,
                        stimulus_type,
                        frequency_hz,
                        bandwidth_hz,
                        relative_level,
                        comfort_score,
                        noticeability,
                        listen_again,
                        trial_date
                    ) in observed_rows:

                        observed_score = calculate_response_score(
                            comfort_score,
                            noticeability,
                            listen_again
                        )

                        observed_records.append(
                            {
                                "Trial ID": trial_id,
                                "Source": source_labels.get(
                                    source_type,
                                    source_type
                                ),
                                "Stimulus": stimulus_type,
                                "Frequency (Hz)": frequency_hz,
                                "Bandwidth (Hz)": bandwidth_hz,
                                "Relative Level": relative_level,
                                "Comfort": comfort_score,
                                "Noticeability": noticeability,
                                "Listen Again": (
                                    "Yes"
                                    if str(listen_again).strip().lower() in ("yes", "1", "true")
                                    else "No"
                                ),
                                "Observed Response / 100": round(
                                    float(observed_score),
                                    1
                                ),
                                "Date": trial_date,
                            }
                        )

                    observed_df = pd.DataFrame(
                        observed_records
                    ).sort_values(
                        "Observed Response / 100",
                        ascending=False
                    )

                    rank_1, rank_2, rank_3 = st.columns(
                        3
                    )

                    best_row = observed_df.iloc[0]

                    rank_1.metric(
                        "Best Observed Response",
                        f"{float(best_row['Observed Response / 100']):.1f} / 100"
                    )

                    rank_2.metric(
                        "Best Stimulus",
                        str(
                            best_row[
                                "Stimulus"
                            ]
                        )
                    )

                    rank_3.metric(
                        "Observed Trials",
                        len(
                            observed_df
                        )
                    )

                    st.dataframe(
                        observed_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Ranking is computed from actual patient feedback. "
                        "The 0–100 response score is a prototype engineering score, "
                        "not a validated clinical outcome or treatment-efficacy percentage."
                    )

            # =================================================
            # 3. STANDARD RULE-BASED ANALYSIS
            # =================================================

            elif acoustic_mode == "✅ Professional Review":

                st.subheader(
                    "Professional Review"
                )

                st.write(
                    "Review the latest supervised result from each recommendation method. "
                    "The audiologist may approve, modify or reject the configuration after considering the patient profile, "
                    "audiometric context, observed response and, when used, the AI prediction."
                )

                st.info(
                    "Manual and AI-assisted recommendations are kept traceable as separate evidence. "
                    "The AI never replaces the audiologist's final professional decision."
                )

                manual_trial = get_latest_personalization_trial(
                    conn,
                    acoustic_patient_id,
                    "standard"
                )

                ai_trial = get_latest_personalization_trial(
                    conn,
                    acoustic_patient_id,
                    "ai"
                )

                manual_col, ai_col = st.columns(2)

                with manual_col:
                    render_audiologist_pathway_review(
                        conn,
                        acoustic_patient_id,
                        acoustic_profile_id,
                        "standard",
                        manual_trial
                    )

                with ai_col:
                    render_audiologist_pathway_review(
                        conn,
                        acoustic_patient_id,
                        acoustic_profile_id,
                        "ai",
                        ai_trial
                    )

                st.caption(
                    "A rejected or modified AI proposal does not invalidate the model; it records the professional decision for this patient and this specific configuration."
                )

            # =================================================
            # 4. ADAPTIVE TESTING
            # =================================================

            elif acoustic_mode == "🧭 Adaptive Testing":

                st.subheader(
                    "Adaptive Testing"
                )

                st.write(
                    "Run a supervised sequential acoustic-testing workflow in which "
                    "each recorded patient response can inform the next candidate."
                )

                st.info(
                    "This is the same adaptive-session functionality that was previously "
                    "a separate audiologist page. It is now integrated here so all acoustic "
                    "testing and personalization work stays in one place."
                )

                st.caption(
                    "Adaptive Testing is a research/prototype decision-support workflow. "
                    "Observed response scores are based on patient feedback and are not "
                    "validated treatment-efficacy measures."
                )

                show_adaptive_session_page(
                    current_role=st.session_state[
                        "auth_role"
                    ],
                    auth_patient_id=st.session_state.get(
                        "auth_patient_id"
                    )
                )

            # =================================================
            # 5. AI ANALYSIS
            # =================================================

            elif acoustic_mode == "🤖 AI Analysis":

                st.subheader(
                    "AI Analysis"
                )

                st.write(
                    "Review saved AI-tested recommendations and compare each model prediction "
                    "with the patient's actual observed response. Model evaluation and training "
                    "remain in AI Model Evaluation."
                )

                ai_rows = conn.execute(
                    """
                    SELECT
                        id,
                        stimulus_type,
                        center_frequency_hz,
                        bandwidth_hz,
                        relative_level,
                        model_predicted_score,
                        comfort_score,
                        tinnitus_noticeability,
                        listen_again,
                        model_version,
                        model_trained_at,
                        COALESCE(tested_at, created_at)
                    FROM stimulus_trials
                    WHERE patient_id = ?
                      AND source_type IN ('audiologist_ai_result', 'patient_ai_result')
                    ORDER BY id DESC
                    """,
                    (
                        acoustic_patient_id,
                    )
                ).fetchall()

                if not ai_rows:

                    st.info(
                        "🤖 No supervised AI-assisted result yet."
                    )

                    st.write(
                        "No AI-assisted candidate has yet been tested under audiologist supervision. "
                        "Use Recommendation & Supervised Test, choose AI-Assisted Recommendation, "
                        "review or modify the AI proposal, and record the patient's observed response."
                    )

                    st.caption(
                        "There is simply no supervised AI-assisted result available for this patient yet."
                    )

                else:

                    ai_records = []

                    for (
                        trial_id,
                        stimulus_type,
                        frequency_hz,
                        bandwidth_hz,
                        relative_level,
                        predicted_score,
                        comfort_score,
                        noticeability,
                        listen_again,
                        model_version,
                        model_trained_at,
                        trial_date
                    ) in ai_rows:

                        observed_score = (
                            calculate_response_score(
                                comfort_score,
                                noticeability,
                                listen_again
                            )
                            if (
                                comfort_score is not None
                                and
                                noticeability is not None
                            )
                            else None
                        )

                        prediction_error = (
                            abs(
                                float(predicted_score)
                                -
                                float(observed_score)
                            )
                            if (
                                predicted_score is not None
                                and
                                observed_score is not None
                            )
                            else None
                        )

                        ai_records.append(
                            {
                                "Trial ID": trial_id,
                                "Stimulus": stimulus_type,
                                "Frequency (Hz)": frequency_hz,
                                "Bandwidth (Hz)": bandwidth_hz,
                                "Relative Level": relative_level,
                                "Predicted Response / 100": (
                                    round(
                                        float(predicted_score),
                                        1
                                    )
                                    if predicted_score is not None
                                    else None
                                ),
                                "Observed Response / 100": (
                                    round(
                                        float(observed_score),
                                        1
                                    )
                                    if observed_score is not None
                                    else None
                                ),
                                "Absolute Difference": (
                                    round(
                                        float(prediction_error),
                                        1
                                    )
                                    if prediction_error is not None
                                    else None
                                ),
                                "Model Version": model_version,
                                "Model Trained At": model_trained_at,
                                "Test Date": trial_date,
                            }
                        )

                    ai_df = pd.DataFrame(
                        ai_records
                    )

                    latest_ai = ai_df.iloc[0]

                    ai_1, ai_2, ai_3 = st.columns(
                        3
                    )

                    ai_1.metric(
                        "Latest AI Prediction",
                        (
                            f"{float(latest_ai['Predicted Response / 100']):.1f} / 100"
                            if pd.notna(
                                latest_ai[
                                    "Predicted Response / 100"
                                ]
                            )
                            else "N/A"
                        )
                    )

                    ai_2.metric(
                        "Observed Response",
                        (
                            f"{float(latest_ai['Observed Response / 100']):.1f} / 100"
                            if pd.notna(
                                latest_ai[
                                    "Observed Response / 100"
                                ]
                            )
                            else "N/A"
                        )
                    )

                    ai_3.metric(
                        "Prediction–Observation Difference",
                        (
                            f"{float(latest_ai['Absolute Difference']):.1f}"
                            if pd.notna(
                                latest_ai[
                                    "Absolute Difference"
                                ]
                            )
                            else "N/A"
                        )
                    )

                    st.dataframe(
                        ai_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    latest_ai_trial = get_latest_personalization_trial(
                        conn,
                        acoustic_patient_id,
                        "ai"
                    )

                    st.caption(
                        "Predicted response is a model estimate. Observed response comes from "
                        "actual patient feedback. They are intentionally kept separate."
                    )

                    st.markdown(
                        "### Final AI Candidate & Professional Review"
                    )

                    if latest_ai_trial is None:

                        st.info(
                            "No final AI result is available for professional review yet."
                        )

                    else:

                        render_audiologist_pathway_review(
                            conn,
                            acoustic_patient_id,
                            acoustic_profile_id,
                            "ai",
                            latest_ai_trial
                        )

                        st.caption(
                            "This decision updates the same AI professional status "
                            "shown to the patient and summarized in Patients."
                        )

elif page == "Care Coordination":

    st.title(
        "Care Coordination"
    )

    selected_care_patient_id = get_active_patient_id()
    selected_care_patient_code = get_active_patient_code()

    if selected_care_patient_id is None:
        st.warning("Open a patient from Patients first.")
    else:
        _care_original_caption = st.caption

        def _care_clean_caption(body, *args, **kwargs):
            if isinstance(body, str) and body.startswith(
                "Coordinate the next step for "
            ):
                return None
            return _care_original_caption(body, *args, **kwargs)

        st.caption = _care_clean_caption
        try:
            show_audiologist_care_coordination(
                selected_care_patient_id,
                selected_care_patient_code
            )
        finally:
            st.caption = _care_original_caption


elif page == "Patient Acoustic Digital Twin":

    digital_twin_patient_id = (
        get_active_patient_id()
        if st.session_state.get("auth_role") == "audiologist"
        else st.session_state.get("auth_patient_id")
    )

    show_digital_twin_page(
        current_role=st.session_state[
            "auth_role"
        ],
        auth_patient_id=digital_twin_patient_id
    )


elif page == "Follow-up & Progress":

    if st.session_state[
        "auth_role"
    ] == "patient":

        show_patient_followup_progress_page(
            st.session_state.get(
                "auth_patient_id"
            )
        )

    else:

        show_audiologist_followup_progress_page()


elif page == "AI Model Evaluation":

    st.info(
        "This page evaluates the SmartTinnitus AI model itself — not a specific patient's "
        "clinical status. Use it to inspect dataset adequacy, patient-wise validation, "
        "prediction error and model retraining."
    )

    show_ml_evaluation_page(
        current_role=st.session_state[
            "auth_role"
        ]
    )


elif page == "SmartTinnitus Assistant":

    def _final_assistant_suggestions(role):
        if role == "patient":
            return [
                "Summarize my SmartTinnitus record",
                "What is my tinnitus pitch?",
                "What was my best recorded stimulus?",
                "What does my response score mean?",
                "What was my latest acoustic response?",
                "Explain my recorded acoustic evidence",
            ]
        return [
            "Summarize this patient",
            "What is the patient's tinnitus pitch?",
            "What was the patient's best recorded stimulus?",
            "Why did the AI recommend this sound?",
            "What was the latest acoustic response?",
            "Summarize the patient's recorded acoustic evidence",
        ]

    smarttinnitus_chatbot_module._suggested_questions = _final_assistant_suggestions

    _assistant_original_info = st.info
    _assistant_original_caption = st.caption

    def _assistant_clean_info(body, *args, **kwargs):
        if isinstance(body, str) and (
            body.startswith("The assistant explains SmartTinnitus measurements")
            or body.startswith("The Assistant can explain your SmartTinnitus record")
            or body.startswith("The Assistant supports explanation of stored patient evidence")
        ):
            return None
        return _assistant_original_info(body, *args, **kwargs)

    def _assistant_clean_caption(body, *args, **kwargs):
        if isinstance(body, str) and (
            body.startswith("Context-aware conversational access")
            or body.startswith("Assistant v2:")
        ):
            return None
        return _assistant_original_caption(body, *args, **kwargs)

    st.info = _assistant_clean_info
    st.caption = _assistant_clean_caption
    try:
        show_smarttinnitus_chatbot(
            current_role=st.session_state[
                "auth_role"
            ],
            auth_patient_id=(
                get_active_patient_id()
                if st.session_state.get("auth_role") == "audiologist"
                else st.session_state.get("auth_patient_id")
            )
        )
    finally:
        st.info = _assistant_original_info
        st.caption = _assistant_original_caption


elif page == "Audiometry":

    st.title(
        "Audiometry"
    )

    patient_id = get_active_patient_id()
    selected = get_active_patient_code()

    if patient_id is None:

        st.warning(
            "Open a patient from Patients first."
        )

    else:

        cursor.execute(
            """
            SELECT estimated_pitch_hz
            FROM tinnitus_profiles
            WHERE patient_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                patient_id,
            )
        )

        result = cursor.fetchone()

        pitch = (
            result[0]
            if result
            and result[0]
            is not None
            else None
        )

        if pitch:

            st.metric(
                "Tinnitus Pitch",
                f"{int(pitch)} Hz"
            )

        right = []
        left = []

        c1, c2 = st.columns(2)

        with c1:

            st.subheader(
                "Right Ear"
            )

            for freq in AUDIOGRAM_FREQUENCIES:

                right.append(
                    st.number_input(
                        f"{freq} Hz (dB HL)",
                        -10,
                        120,
                        20,
                        5,
                        key=f"right_{freq}"
                    )
                )

        with c2:

            st.subheader(
                "Left Ear"
            )

            for freq in AUDIOGRAM_FREQUENCIES:

                left.append(
                    st.number_input(
                        f"{freq} Hz (dB HL)",
                        -10,
                        120,
                        20,
                        5,
                        key=f"left_{freq}"
                    )
                )

        if st.button(
            "Plot Audiogram"
        ):

            st.pyplot(
                plot_audiogram(
                    right,
                    left,
                    tinnitus_pitch=pitch
                )
            )

        if st.button(
            "Save Audiogram"
        ):

            cursor.execute(
                """
                DELETE FROM audiograms
                WHERE patient_id = ?
                """,
                (
                    patient_id,
                )
            )

            now = (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            for frequency, value in zip(
                AUDIOGRAM_FREQUENCIES,
                right
            ):

                cursor.execute(
                    """
                    INSERT INTO audiograms
                    (
                        patient_id,
                        ear,
                        frequency_hz,
                        threshold_db_hl,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        patient_id,
                        "Right",
                        frequency,
                        value,
                        now
                    )
                )

            for frequency, value in zip(
                AUDIOGRAM_FREQUENCIES,
                left
            ):

                cursor.execute(
                    """
                    INSERT INTO audiograms
                    (
                        patient_id,
                        ear,
                        frequency_hz,
                        threshold_db_hl,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        patient_id,
                        "Left",
                        frequency,
                        value,
                        now
                    )
                )

            conn.commit()

            st.success(
                "Audiogram saved."
            )


# =====================================================
# CLINICAL REPORT
# =====================================================

elif page == "Clinical Report":

    st.title(
        "Clinical Report"
    )

    patient_id = get_active_patient_id()
    selected = get_active_patient_code()

    if patient_id is None:

        st.warning(
            "Open a patient from Patients first."
        )

    else:

        cursor.execute(
            """
            SELECT
                patient_code,
                age,
                sex
            FROM patients
            WHERE id = ?
            """,
            (
                patient_id,
            )
        )

        row = cursor.fetchone()

        patient_info = {
            "patient_code":
                row[0],

            "age":
                row[1],

            "sex":
                row[2]
        }

        cursor.execute(
            """
            SELECT
                laterality,
                tinnitus_type,
                temporal_pattern,
                duration_months,
                annoyance_score,
                sleep_score,
                concentration_score,
                stress_score,
                estimated_pitch_hz,
                loudness_relative
            FROM tinnitus_profiles
            WHERE patient_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                patient_id,
            )
        )

        row = cursor.fetchone()

        profile = None

        if row:

            profile = {

                "laterality":
                    row[0],

                "tinnitus_type":
                    row[1],

                "temporal_pattern":
                    row[2],

                "duration_months":
                    row[3],

                "annoyance_score":
                    row[4],

                "sleep_score":
                    row[5],

                "concentration_score":
                    row[6],

                "stress_score":
                    row[7],

                "estimated_pitch_hz":
                    row[8],

                "loudness_relative":
                    row[9]
            }

        cursor.execute(
            """
            SELECT
                session_id,
                decision_id,
                profile_id,
                source_type,
                model_version,
                model_trained_at,
                model_predicted_score,
                recommendation_created_at,
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
            ORDER BY created_at
            """,
            (
                patient_id,
            )
        )

        trials = []

        for row in cursor.fetchall():

            trials.append(
                {
                    "session_id":
                        row[0],

                    "decision_id":
                        row[1],

                    "profile_id":
                        row[2],

                    "source_type":
                        row[3],

                    "model_version":
                        row[4],

                    "model_trained_at":
                        row[5],

                    "model_predicted_score":
                        row[6],

                    "recommendation_created_at":
                        row[7],

                    "tested_at":
                        row[8],

                    "stimulus_type":
                        row[9],

                    "frequency":
                        row[10],

                    "bandwidth":
                        row[11],

                    "level":
                        row[12],

                    "comfort":
                        row[13],

                    "noticeability":
                        row[14],

                    "listen_again":
                        row[15],

                    "created_at":
                        row[16]
                }
            )

        cursor.execute(
            """
            SELECT
                id,
                session_date,
                annoyance_before,
                annoyance_after,
                tinnitus_noticeability_before,
                tinnitus_noticeability_after,
                notes
            FROM sessions
            WHERE patient_id = ?
            ORDER BY session_date, id
            """,
            (
                patient_id,
            )
        )

        sessions = []

        for row in cursor.fetchall():

            sessions.append(
                {
                    "session_id":
                        row[0],

                    "date":
                        row[1],

                    "annoyance_before":
                        row[2],

                    "annoyance_after":
                        row[3],

                    "noticeability_before":
                        row[4],

                    "noticeability_after":
                        row[5],

                    "notes":
                        row[6]
                }
            )

        cursor.execute(
            """
            SELECT
                ear,
                frequency_hz,
                threshold_db_hl
            FROM audiograms
            WHERE patient_id = ?
            ORDER BY frequency_hz
            """,
            (
                patient_id,
            )
        )

        aud_rows = cursor.fetchall()

        audiogram_report = None

        if aud_rows:

            right = []
            left = []

            for freq in AUDIOGRAM_FREQUENCIES:

                right.append(
                    next(
                        (
                            row[2]
                            for row
                            in aud_rows
                            if row[0]
                            == "Right"
                            and row[1]
                            == freq
                        ),
                        None
                    )
                )

                left.append(
                    next(
                        (
                            row[2]
                            for row
                            in aud_rows
                            if row[0]
                            == "Left"
                            and row[1]
                            == freq
                        ),
                        None
                    )
                )

            if (
                None not in right
                and
                None not in left
            ):

                audiogram_report = {

                    "frequencies":
                        AUDIOGRAM_FREQUENCIES,

                    "right_ear":
                        right,

                    "left_ear":
                        left,

                    "tinnitus_pitch":
                        (
                            profile[
                                "estimated_pitch_hz"
                            ]
                            if profile
                            else None
                        )
                }

        cursor.execute(
            """
            SELECT
                ad.id,
                ad.session_id,
                ad.profile_id,
                ad.model_version,
                ad.model_trained_at,
                ad.recommendation_created_at,
                s.session_date,
                ad.ai_stimulus_type,
                ad.ai_frequency_hz,
                ad.ai_bandwidth_hz,
                ad.ai_relative_level,
                ad.ai_predicted_score,
                ad.decision,
                ad.final_stimulus_type,
                ad.final_frequency_hz,
                ad.final_bandwidth_hz,
                ad.final_relative_level,
                ad.final_predicted_score,
                ad.clinician_notes,
                ad.created_at
            FROM audiologist_decisions ad
            LEFT JOIN sessions s
                ON ad.session_id = s.id
            WHERE ad.patient_id = ?
            ORDER BY ad.id DESC
            """,
            (
                patient_id,
            )
        )

        report_decisions = []

        for row in cursor.fetchall():

            report_decisions.append(
                {
                    "decision_id":
                        row[0],

                    "session_id":
                        row[1],

                    "profile_id":
                        row[2],

                    "model_version":
                        row[3],

                    "model_trained_at":
                        row[4],

                    "recommendation_created_at":
                        row[5],

                    "session_date":
                        row[6],

                    "ai_stimulus_type":
                        row[7],

                    "ai_frequency_hz":
                        row[8],

                    "ai_bandwidth_hz":
                        row[9],

                    "ai_relative_level":
                        row[10],

                    "ai_predicted_score":
                        row[11],

                    "decision":
                        row[12],

                    "final_stimulus_type":
                        row[13],

                    "final_frequency_hz":
                        row[14],

                    "final_bandwidth_hz":
                        row[15],

                    "final_relative_level":
                        row[16],

                    "final_predicted_score":
                        row[17],

                    "clinician_notes":
                        row[18],

                    "created_at":
                        row[19]
                }
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Patient",
            selected
        )

        c2.metric(
            "Sessions",
            len(sessions)
        )

        c3.metric(
            "Trials",
            len(trials)
        )

        if st.button(
            "Generate PDF Report"
        ):

            filename = (
                f"SmartTinnitus_Report_"
                f"{selected}.pdf"
            )

            try:

                generate_patient_report(
                    filename,
                    patient_info,
                    profile,
                    trials,
                    sessions,
                    audiogram_report,
                    acoustic_fingerprint=None,
                    audiologist_decisions=report_decisions
                )

                st.session_state[
                    "generated_report"
                ] = filename

                st.session_state[
                    "generated_report_patient"
                ] = selected

                st.success(
                    "Report generated."
                )

            except Exception as error:

                st.error(
                    f"Report failed: {error}"
                )

        if (
            st.session_state.get(
                "generated_report_patient"
            )
            ==
            selected
            and
            "generated_report"
            in
            st.session_state
        ):

            filename = (
                st.session_state[
                    "generated_report"
                ]
            )

            try:

                with open(
                    filename,
                    "rb"
                ) as file:

                    st.download_button(
                        "⬇ Download Report",
                        data=file.read(),
                        file_name=filename,
                        mime="application/pdf"
                    )

            except FileNotFoundError:

                st.warning(
                    "Generate the report again."
                )


# =====================================================
# MINIMAL FOOTER
# =====================================================

st.markdown(
    "<div style='margin-top:38px;padding:16px 0 4px 0;border-top:1px solid #e5e9e7;"
    "text-align:center;color:#909894;font-size:.73rem'>"
    "© ABC 2023 · Medico International Services · SmartTinnitus"
    "</div>",
    unsafe_allow_html=True,
)


# =====================================================
# CLOSE DATABASE
# =====================================================

conn.close()