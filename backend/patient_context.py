import streamlit as st


# ============================================================
# ACTIVE PATIENT CONTEXT
# ============================================================
#
# Important:
# The Streamlit widget key "active_patient_selector" is owned
# by the selectbox/radio widget itself. We must NEVER write to
# that key after the widget has been instantiated.
#
# We therefore store the actual shared patient context in
# separate session-state keys:
#
#   active_patient_id
#   active_patient_code
#
# ============================================================


def initialize_patient_context():
    """Create the shared patient-context state once."""

    defaults = {
        "active_patient_id": None,
        "active_patient_code": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_active_patient(patient_id=None, patient_code=None):
    """
    Save the patient currently being reviewed by the audiologist.

    This function intentionally does NOT modify
    st.session_state["active_patient_selector"] because that key
    belongs to the Streamlit widget.
    """

    initialize_patient_context()

    st.session_state["active_patient_id"] = (
        int(patient_id) if patient_id is not None else None
    )

    st.session_state["active_patient_code"] = (
        str(patient_code) if patient_code is not None else None
    )


def clear_active_patient():
    """Clear the professional patient context."""

    initialize_patient_context()

    st.session_state["active_patient_id"] = None
    st.session_state["active_patient_code"] = None


def get_active_patient_id():
    """Return the current patient database ID."""

    initialize_patient_context()
    return st.session_state.get("active_patient_id")


def get_active_patient_code():
    """Return the current patient code."""

    initialize_patient_context()
    return st.session_state.get("active_patient_code")


def get_active_patient():
    """Return the current patient context as a dictionary."""

    initialize_patient_context()

    return {
        "patient_id": st.session_state.get("active_patient_id"),
        "patient_code": st.session_state.get("active_patient_code"),
    }


def sync_active_patient_from_selector(patient_lookup):
    """
    Optional helper.

    patient_lookup should be a mapping such as:

        {
            "001": 1,
            "002": 2,
        }

    The widget is allowed to control active_patient_selector.
    This function only READS that widget value and copies the
    corresponding information into the independent patient context.
    """

    initialize_patient_context()

    selected_code = st.session_state.get("active_patient_selector")

    if selected_code is None:
        return None

    patient_id = patient_lookup.get(selected_code)

    if patient_id is None:
        return None

    set_active_patient(
        patient_id=patient_id,
        patient_code=selected_code,
    )

    return patient_id
