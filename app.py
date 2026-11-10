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



from digital_twin import (
    show_digital_twin_page as _show_digital_twin_page
)

from streamlit.delta_generator import DeltaGenerator

from smarttinnitus_chatbot import (
    show_smarttinnitus_chatbot
)
import smarttinnitus_chatbot as smarttinnitus_chatbot_module


from backend.db import (
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
    create_patient_with_login as backend_create_patient_with_login,
)
from backend.patient_context import (
    get_active_patient_id,
    get_active_patient_code,
    set_active_patient,
)
from backend.acoustic_service import (
    get_acoustic_context as backend_get_acoustic_context,
    generate_ai_candidates as backend_generate_ai_candidates,
    send_candidate as backend_send_acoustic_candidate,
    get_patient_pending_assignment as backend_get_pending_acoustic_assignment,
    submit_patient_feedback as backend_submit_acoustic_feedback,
    get_patient_feedback as backend_list_acoustic_feedback,
)
from backend.audiometry_service import (
    get_latest_tinnitus_pitch as backend_get_latest_tinnitus_pitch,
    get_latest_audiogram as backend_get_latest_audiogram,
    get_audiogram_history as backend_get_audiogram_history,
    save_audiogram as backend_save_audiogram,
)
from backend.digital_twin_service import (
    resolve_patient_id as backend_resolve_digital_twin_patient_id,
    get_evidence_summary as backend_get_twin_evidence_summary,
    get_personalization_result_counts as backend_get_twin_result_counts,
    get_audiologist_twin_evolution as backend_get_audiologist_twin_evolution,
    get_patient_twin_evolution as backend_get_patient_twin_evolution,
)
from backend.report_service import (
    get_report_summary as backend_get_report_summary,
    generate_clinical_report as backend_generate_clinical_report,
    get_report_history as backend_get_report_history,
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


def _resolve_digital_twin_patient_id(selected_value):
    return backend_resolve_digital_twin_patient_id(selected_value)


def _get_audiologist_twin_evidence_summary(patient_id):
    return backend_get_twin_evidence_summary(patient_id)


def _render_audiologist_twin_evolution(patient_id):
    st.markdown("## Twin Evolution")

    if patient_id is None:
        st.info("Select a patient to review their Digital Twin evolution.")
        return

    payload = backend_get_audiologist_twin_evolution(patient_id)
    display_df = payload["rows"]
    summary = payload["summary"]

    if display_df.empty:
        st.info("No recorded acoustic evidence is available for this patient yet.")
        return

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Observed Acoustic Trials", summary["observed_trials"])
    e2.metric("Manual Results", summary["standard_results"])
    e3.metric("AI-Tested Results", summary["ai_results"])
    e4.metric("Audiogram", "Available" if summary["audiogram_available"] else "Not available")

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def _get_patient_personalization_result_counts(patient_id):
    return backend_get_twin_result_counts(patient_id)


def _render_patient_twin_evolution(patient_id):
    st.markdown("## Twin Evolution")

    payload = backend_get_patient_twin_evolution(patient_id)
    display_df = payload["rows"]

    if display_df.empty:
        st.info(
            "No completed supervised manual or AI-assisted acoustic results are available yet. "
            "Your audiologist must first complete a supervised acoustic test."
        )
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Manual Results", payload["manual_count"])
    c2.metric("AI-Assisted Results", payload["ai_count"])
    c3.metric("Total Supervised Results", payload["total_count"])

    st.markdown("### Your supervised acoustic history")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


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
            "Tinnitus Profile",
            "Available" if tinnitus_profile_count > 0 else "Not completed"
        )

        m2.metric(
            "Recorded Acoustic Trials",
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
            """
            <style>
            .audio-home-shell {
                padding: 4px 0 8px 0;
            }

            .audio-home-kicker {
                color: #2a8191;
                font-size: 0.74rem;
                font-weight: 800;
                letter-spacing: 0.11em;
                text-transform: uppercase;
                margin-bottom: 0.45rem;
            }

            .audio-home-title {
                color: #123f52;
                font-size: 2.25rem;
                line-height: 1.1;
                font-weight: 850;
                letter-spacing: -0.035em;
                margin-bottom: 0.4rem;
            }

            .audio-home-subtitle {
                color: #6b7f88;
                font-size: 0.98rem;
                line-height: 1.55;
                max-width: 760px;
            }

            div[data-testid="stMetric"] {
                min-height: 104px;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 18px !important;
            }
            </style>
            """
        )

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
            "📋 Patient Overview",
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

                result = backend_create_patient_with_login(
                    patient_code=patient_code,
                    age=age,
                    sex=sex,
                )

                if result["success"]:

                    st.session_state[
                        "new_patient_credentials"
                    ] = {
                        "patient_id": result["patient_id"],
                        "patient_code": result["patient_code"],
                        "username": result["username"],
                        "password": result["password"],
                    }

                    st.success(
                        result["message"]
                    )

                else:

                    st.error(
                        result["message"]
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

    # Patient self-service personalization has been removed.
    # The remaining workspace is audiologist-controlled.

    # =================================================
    # AUDIOLOGIST — TECHNICAL PERSONALIZATION WORKSPACE
    # =================================================

    if st.session_state.get("auth_role") == "audiologist":

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


        acoustic_patient_id = get_active_patient_id()
        acoustic_patient_code = get_active_patient_code()

        if acoustic_patient_id is None:
            st.warning("Open a patient from Patients first.")
            st.stop()

        acoustic_context = backend_get_acoustic_context(acoustic_patient_id)
        acoustic_profile = acoustic_context.get("profile")
        acoustic_profile_id = acoustic_profile.get("id") if acoustic_profile else None
        acoustic_pitch = float(acoustic_profile["estimated_pitch_hz"]) if acoustic_profile and acoustic_profile.get("estimated_pitch_hz") is not None else None
        acoustic_loudness = float(acoustic_profile["loudness_relative"]) if acoustic_profile and acoustic_profile.get("loudness_relative") is not None else None
        acoustic_type = acoustic_profile.get("tinnitus_type") if acoustic_profile else None
        acoustic_laterality = acoustic_profile.get("laterality") if acoustic_profile else None
        acoustic_pattern = acoustic_profile.get("temporal_pattern") if acoustic_profile else None
        acoustic_duration = acoustic_profile.get("duration_months") if acoustic_profile else None
        acoustic_annoyance = acoustic_profile.get("annoyance_score") if acoustic_profile else None
        acoustic_sleep = acoustic_profile.get("sleep_score") if acoustic_profile else None
        acoustic_concentration = acoustic_profile.get("concentration_score") if acoustic_profile else None
        acoustic_stress = acoustic_profile.get("stress_score") if acoustic_profile else None
        acoustic_profile_date = acoustic_profile.get("created_at") if acoustic_profile else None
        has_audiogram = bool(acoustic_context.get("has_audiogram"))
        trial_count = int(acoustic_context.get("trial_count", 0))

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
                        ai_result = backend_generate_ai_candidates(acoustic_patient_id, limit=5)
                        ranked_candidates = ai_result["candidates"]
                        st.session_state[ranking_key] = ranked_candidates
                        st.session_state[index_key] = 0
                        st.session_state[recommendation_key] = ranked_candidates[0]
                        st.session_state[created_key] = ai_result["created_at"]
                        st.session_state[f"audiologist_ai_provenance_{acoustic_patient_id}"] = ai_result.get("provenance")
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

                provenance = st.session_state.get(f"audiologist_ai_provenance_{acoustic_patient_id}")

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
                    assignment_id = backend_send_acoustic_candidate(
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

        pitch = backend_get_latest_tinnitus_pitch(
            patient_id
        )

        if pitch:

            st.metric(
                "Tinnitus Pitch",
                f"{int(pitch)} Hz"
            )

        # -------------------------------------------------
        # Latest saved audiogram
        # -------------------------------------------------

        latest_audiogram = backend_get_latest_audiogram(
            patient_id
        )

        st.subheader(
            "Latest Saved Audiogram"
        )

        if latest_audiogram.get("created_at"):

            st.caption(
                f"Saved: {latest_audiogram['created_at']}"
            )

            saved_right = [
                latest_audiogram["Right"].get(
                    int(freq),
                    20.0
                )
                for freq in AUDIOGRAM_FREQUENCIES
            ]

            saved_left = [
                latest_audiogram["Left"].get(
                    int(freq),
                    20.0
                )
                for freq in AUDIOGRAM_FREQUENCIES
            ]

            st.pyplot(
                plot_audiogram(
                    saved_right,
                    saved_left,
                    tinnitus_pitch=pitch
                )
            )

            saved_table = pd.DataFrame(
                {
                    "Frequency (Hz)": AUDIOGRAM_FREQUENCIES,
                    "Right ear (dB HL)": saved_right,
                    "Left ear (dB HL)": saved_left,
                }
            )

            with st.expander(
                "View saved thresholds"
            ):
                st.dataframe(
                    saved_table,
                    use_container_width=True,
                    hide_index=True
                )

            history = backend_get_audiogram_history(
                patient_id,
                limit=10
            )

            if len(history) > 1:
                with st.expander(
                    "Previous Audiograms"
                ):
                    for historical in history[1:]:
                        st.markdown(
                            f"**Saved: {historical['created_at']}**"
                        )
                        historical_right = [
                            historical["Right"].get(int(freq), 20.0)
                            for freq in AUDIOGRAM_FREQUENCIES
                        ]
                        historical_left = [
                            historical["Left"].get(int(freq), 20.0)
                            for freq in AUDIOGRAM_FREQUENCIES
                        ]
                        st.pyplot(
                            plot_audiogram(
                                historical_right,
                                historical_left,
                                tinnitus_pitch=pitch
                            )
                        )
        else:
            st.info(
                "No audiogram has been saved for this patient yet."
            )

        st.markdown("---")
        st.subheader(
            "Enter New Audiogram"
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

        action_col1, action_col2 = st.columns(2)

        with action_col1:
            if st.button(
                "Plot Audiogram",
                use_container_width=True
            ):

                st.pyplot(
                    plot_audiogram(
                        right,
                        left,
                        tinnitus_pitch=pitch
                    )
                )

        with action_col2:
            if st.button(
                "Save Audiogram",
                use_container_width=True
            ):

                backend_save_audiogram(
                    patient_id=patient_id,
                    frequencies=AUDIOGRAM_FREQUENCIES,
                    right_thresholds=right,
                    left_thresholds=left,
                )

                st.success(
                    "Audiogram saved."
                )
                st.rerun()


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

        try:
            report_summary = backend_get_report_summary(patient_id)
        except Exception as error:
            st.error(f"Could not load report data: {error}")
            report_summary = None

        if report_summary is not None:
            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Patient",
                report_summary["patient_code"]
            )

            c2.metric(
                "Acoustic Tests",
                report_summary["acoustic_tests"]
            )

            c3.metric(
                "Feedback",
                report_summary["feedback_count"]
            )

            c4.metric(
                "Audiogram",
                "Available" if report_summary["has_audiogram"] else "Not available"
            )

            if st.button(
                "Generate PDF Report"
            ):

                try:
                    filename = backend_generate_clinical_report(
                        patient_id,
                        generated_by_user_id=st.session_state.get("auth_user_id"),
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

                filename = st.session_state[
                    "generated_report"
                ]

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


            st.markdown("### Report History")

            try:
                report_history = backend_get_report_history(
                    patient_id,
                    limit=10,
                )
            except Exception as error:
                report_history = []
                st.warning(
                    f"Could not load report history: {error}"
                )

            if report_history:
                history_rows = []

                for item in report_history:
                    history_rows.append(
                        {
                            "Generated": item.get("generated_at"),
                            "Version": item.get("report_version"),
                            "File": item.get("filename"),
                            "Status": item.get("status"),
                        }
                    )

                st.dataframe(
                    history_rows,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption(
                    "No generated reports are recorded yet."
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