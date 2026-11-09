import pandas as pd
import streamlit as st

from database import get_connection
from optimization import calculate_response_score

try:
    from backend.patient_context import (
        get_active_patient_id,
        get_active_patient_code,
    )
except Exception:
    get_active_patient_id = None
    get_active_patient_code = None

try:
    from backend.acoustic_assignment_service import (
        get_pending_assignment,
        list_assignment_feedback,
    )
except Exception:
    get_pending_assignment = None
    list_assignment_feedback = None


# ============================================================
# SMARTTINNITUS ASSISTANT — CLEAN FINAL FRONTEND VERSION
# ============================================================
#
# This module matches the final SmartTinnitus workflow:
# - one active patient selected from Patients on the audiologist side
# - no extra patient selector inside the Assistant
# - no Adaptive Session questions
# - no Confidence / Active Learning questions
# - no old Approve / Modify / Reject-after-feedback wording
# - explanations are grounded in the stored SmartTinnitus record
# ============================================================


def _table_exists(conn, table_name):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _safe_df(conn, query, params=()):
    try:
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()


def _load_patient_context(patient_id):
    conn = get_connection()
    try:
        patient_df = _safe_df(
            conn,
            """
            SELECT id, patient_code, age, sex, created_at
            FROM patients
            WHERE id = ?
            LIMIT 1
            """,
            (int(patient_id),),
        )

        profile_df = _safe_df(
            conn,
            """
            SELECT *
            FROM tinnitus_profiles
            WHERE patient_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (int(patient_id),),
        )

        trials_df = _safe_df(
            conn,
            """
            SELECT *
            FROM stimulus_trials
            WHERE patient_id = ?
            ORDER BY id ASC
            """,
            (int(patient_id),),
        )

        audiogram_df = (
            _safe_df(
                conn,
                """
                SELECT *
                FROM audiograms
                WHERE patient_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (int(patient_id),),
            )
            if _table_exists(conn, "audiograms")
            else pd.DataFrame()
        )

        care_df = (
            _safe_df(
                conn,
                """
                SELECT *
                FROM care_requests
                WHERE patient_id = ?
                ORDER BY id DESC
                LIMIT 20
                """,
                (int(patient_id),),
            )
            if _table_exists(conn, "care_requests")
            else pd.DataFrame()
        )

        patient = None if patient_df.empty else patient_df.iloc[0].to_dict()
        profile = None if profile_df.empty else profile_df.iloc[0].to_dict()

        if not trials_df.empty:
            scores = []
            for _, row in trials_df.iterrows():
                try:
                    score = calculate_response_score(
                        row.get("comfort_score"),
                        row.get("tinnitus_noticeability"),
                        row.get("listen_again"),
                    )
                except Exception:
                    score = None
                scores.append(score)
            trials_df["response_score"] = scores
        else:
            trials_df["response_score"] = []

        pending_assignment = None
        feedback = []

        if get_pending_assignment is not None:
            try:
                pending_assignment = get_pending_assignment(int(patient_id))
            except Exception:
                pending_assignment = None

        if list_assignment_feedback is not None:
            try:
                feedback = list_assignment_feedback(int(patient_id), limit=20)
            except Exception:
                feedback = []

        return {
            "patient": patient,
            "profile": profile,
            "trials": trials_df,
            "audiogram": audiogram_df,
            "care": care_df,
            "pending_assignment": pending_assignment,
            "assignment_feedback": feedback,
        }
    finally:
        conn.close()


def _best_trial(trials_df):
    if trials_df.empty or "response_score" not in trials_df.columns:
        return None
    valid = trials_df[trials_df["response_score"].notna()].copy()
    if valid.empty:
        return None
    return valid.sort_values("response_score", ascending=False).iloc[0].to_dict()


def _latest_trial(trials_df):
    if trials_df.empty:
        return None
    return trials_df.iloc[-1].to_dict()


def _fmt_frequency(value, stimulus_type=None):
    if stimulus_type == "White Noise":
        return "Broadband"
    try:
        return f"{float(value):.0f} Hz"
    except Exception:
        return "Not available"


def _summary(context):
    patient = context["patient"] or {}
    profile = context["profile"] or {}
    trials = context["trials"]
    feedback = context["assignment_feedback"]

    parts = [
        f"Patient **{patient.get('patient_code', 'Unknown')}**",
        f"age **{patient.get('age', 'N/A')}**",
        f"sex **{patient.get('sex', 'N/A')}**.",
    ]

    pitch = profile.get("estimated_pitch_hz")
    if pitch is not None:
        parts.append(f" Latest matched tinnitus pitch: **{_fmt_frequency(pitch)}**.")

    laterality = profile.get("laterality")
    if laterality:
        parts.append(f" Laterality: **{laterality}**.")

    if not trials.empty:
        parts.append(f" Recorded acoustic trials: **{len(trials)}**.")

    if feedback:
        latest = feedback[0]
        parts.append(
            " Latest returned acoustic feedback: "
            f"comfort **{latest.get('comfort_score', 'N/A')}/10**, "
            f"noticeability **{latest.get('tinnitus_noticeability', 'N/A')}/10**, "
            f"listen again **{latest.get('listen_again', 'N/A')}**."
        )

    return "".join(parts)


def _answer(question, context, role):
    q = question.lower().strip()
    profile = context["profile"] or {}
    trials = context["trials"]
    pending = context["pending_assignment"]
    feedback = context["assignment_feedback"]
    care = context["care"]
    audiogram = context["audiogram"]

    if any(x in q for x in ["summary", "summarize", "overview"]):
        return _summary(context)

    if "pitch" in q:
        pitch = profile.get("estimated_pitch_hz")
        if pitch is None:
            return "No matched tinnitus pitch is currently stored for this patient."
        return f"The latest stored tinnitus pitch is **{_fmt_frequency(pitch)}**."

    if any(x in q for x in ["best stimulus", "best sound", "best trial", "best response"]):
        best = _best_trial(trials)
        if best is None:
            return "There is not enough recorded acoustic feedback to identify a best observed trial yet."
        return (
            f"The strongest observed trial is **{best.get('stimulus_type', 'Unknown')}** at "
            f"**{_fmt_frequency(best.get('center_frequency_hz'), best.get('stimulus_type'))}** "
            f"with an observed response score of **{float(best['response_score']):.1f}/100**."
        )

    if any(x in q for x in ["latest trial", "last trial", "most recent trial"]):
        latest = _latest_trial(trials)
        if latest is None:
            return "No acoustic trials are currently stored for this patient."
        return (
            f"The latest recorded trial used **{latest.get('stimulus_type', 'Unknown')}** at "
            f"**{_fmt_frequency(latest.get('center_frequency_hz'), latest.get('stimulus_type'))}**."
        )

    if any(x in q for x in ["assigned acoustic", "assigned sound", "pending acoustic", "current sound"]):
        if pending is None:
            return "There is no acoustic test currently waiting for this patient."
        pathway = "AI-assisted" if pending.get("pathway") == "ai" else "Audiologist manual"
        return (
            f"The current assigned test is an **{pathway}** candidate: "
            f"**{pending.get('stimulus_type', 'Unknown')}**, "
            f"**{_fmt_frequency(pending.get('center_frequency_hz'), pending.get('stimulus_type'))}**, "
            f"relative level **{pending.get('relative_level', 'N/A')}**, "
            f"duration **{pending.get('duration_seconds', 'N/A')} s**."
        )

    if any(x in q for x in ["patient feedback", "latest feedback", "returned feedback", "acoustic feedback"]):
        if not feedback:
            return "No returned feedback from an assigned acoustic test is currently stored."
        latest = feedback[0]
        text = (
            f"Latest feedback: comfort **{latest.get('comfort_score', 'N/A')}/10**, "
            f"tinnitus noticeability **{latest.get('tinnitus_noticeability', 'N/A')}/10**, "
            f"listen again **{latest.get('listen_again', 'N/A')}**."
        )
        if latest.get("patient_comment"):
            text += f" Patient comment: **{latest['patient_comment']}**."
        return text

    if any(x in q for x in ["ai recommend", "ai candidate", "why did the ai", "ai suggestion"]):
        ai_rows = trials
        if not ai_rows.empty and "source_type" in ai_rows.columns:
            ai_rows = ai_rows[
                ai_rows["source_type"].astype(str).str.contains("ai", case=False, na=False)
            ]
        if ai_rows.empty:
            return "No stored AI-assisted acoustic trial is available for this patient yet."
        latest = ai_rows.iloc[-1].to_dict()
        predicted = latest.get("model_predicted_score")
        result = (
            f"The latest stored AI-assisted candidate is **{latest.get('stimulus_type', 'Unknown')}** at "
            f"**{_fmt_frequency(latest.get('center_frequency_hz'), latest.get('stimulus_type'))}**."
        )
        if predicted is not None:
            try:
                result += f" Its stored model-predicted response was **{float(predicted):.1f}/100**."
            except Exception:
                pass
        result += " The audiologist reviews an AI-generated candidate before sending it to the patient."
        return result

    if "audiogram" in q or "audiometry" in q:
        if audiogram.empty:
            return "No saved audiogram is currently available for this patient."
        return "A saved audiogram is available for this patient."

    if any(x in q for x in ["care", "appointment", "referral", "coordination"]):
        if care.empty:
            return "No care-coordination records are currently stored for this patient."
        latest = care.iloc[0].to_dict()
        status = latest.get("status", "N/A")
        title = latest.get("title") or latest.get("request_type") or "care request"
        return f"The latest care record is **{title}** with status **{status}**."

    if any(x in q for x in ["digital twin", "response map", "observed evidence", "twin evolution"]):
        return (
            "The Patient Acoustic Digital Twin uses the patient's stored profile, "
            "recorded acoustic evidence, and model predictions. The final interface keeps "
            "three views: **Acoustic Response Map**, **Observed Evidence**, and **Twin Evolution**."
        )

    if "response score" in q or "score mean" in q:
        return (
            "The SmartTinnitus response score is calculated from recorded acoustic-test feedback "
            "such as comfort, tinnitus noticeability, and willingness to listen again. "
            "It is used to compare acoustic responses inside the platform."
        )

    if role == "patient":
        return (
            "You can ask me about your tinnitus profile, tinnitus pitch, assigned acoustic test, "
            "latest acoustic feedback, best observed sound, care requests, audiogram availability, "
            "or Digital Twin."
        )

    return (
        "You can ask me to summarize the active patient, explain the tinnitus profile, "
        "check audiogram availability, review the current assigned acoustic test, inspect "
        "returned patient feedback, identify the best observed acoustic response, review AI-assisted "
        "candidate information, check care coordination, or explain the Digital Twin."
    )


def _suggested_questions(role):
    if role == "patient":
        return [
            "Summarize my SmartTinnitus record",
            "What is my tinnitus pitch?",
            "Do I have an assigned acoustic test?",
            "What was my latest acoustic feedback?",
            "What was my best observed sound?",
            "Is an audiogram available?",
            "What are my latest care requests?",
            "Explain my Digital Twin",
        ]

    return [
        "Summarize this patient",
        "What is the patient's tinnitus pitch?",
        "Is an audiogram available?",
        "What acoustic test is currently assigned?",
        "What was the latest patient feedback?",
        "What was the best observed acoustic response?",
        "What is the latest AI-assisted candidate?",
        "What is the latest care-coordination status?",
        "Explain this patient's Digital Twin",
    ]


def _resolve_patient(current_role, auth_patient_id):
    if current_role == "patient":
        return auth_patient_id

    if get_active_patient_id is not None:
        try:
            return get_active_patient_id()
        except Exception:
            pass

    return st.session_state.get("active_patient_id")


def show_smarttinnitus_chatbot(
    current_role,
    auth_patient_id=None
):
    st.title("SmartTinnitus Assistant")

    patient_id = _resolve_patient(
        current_role=current_role,
        auth_patient_id=auth_patient_id,
    )

    if patient_id is None:
        if current_role == "patient":
            st.error("This patient account is not linked to a patient record.")
        else:
            st.warning("Open a patient from Patients first.")
        return

    context = _load_patient_context(int(patient_id))

    if context["patient"] is None:
        st.error("The patient record could not be found.")
        return

    patient_code = context["patient"]["patient_code"]

    if current_role != "patient":
        st.markdown(f"**Active patient:** {patient_code}")

    profile = context["profile"] or {}
    trials = context["trials"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Patient", patient_code)
    c2.metric("Recorded Trials", len(trials))

    pitch = profile.get("estimated_pitch_hz")
    c3.metric(
        "Tinnitus Pitch",
        _fmt_frequency(pitch) if pitch is not None else "Not measured",
    )

    st.markdown("### Ask a question")

    with st.form(
        f"smarttinnitus_question_form_{current_role}_{patient_id}",
        clear_on_submit=True,
    ):
        typed_question = st.text_input(
            "Your question",
            placeholder=(
                "Example: What was the latest patient feedback?"
                if current_role != "patient"
                else "Example: Do I have an assigned acoustic test?"
            ),
        )
        submitted = st.form_submit_button(
            "Send Question",
            use_container_width=True,
        )

    st.markdown("### Suggested Questions")

    suggestions = _suggested_questions(current_role)
    cols = st.columns(2)
    selected = None

    for index, question in enumerate(suggestions):
        with cols[index % 2]:
            if st.button(
                question,
                use_container_width=True,
                key=f"assistant_suggestion_{current_role}_{patient_id}_{index}",
            ):
                selected = question

    state_key = f"smarttinnitus_chat_{current_role}_{patient_id}"

    if state_key not in st.session_state:
        st.session_state[state_key] = []

    messages = st.session_state[state_key]

    question = None
    if selected:
        question = selected
    elif submitted and typed_question.strip():
        question = typed_question.strip()

    if question:
        answer = _answer(question, context, current_role)
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
        st.session_state[state_key] = messages
        st.rerun()

    if messages:
        st.markdown("---")
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    chat_question = st.chat_input("Ask another SmartTinnitus question...")

    if chat_question:
        answer = _answer(chat_question, context, current_role)
        messages.append({"role": "user", "content": chat_question})
        messages.append({"role": "assistant", "content": answer})
        st.session_state[state_key] = messages
        st.rerun()

    if messages:
        if st.button(
            "Clear Conversation",
            key=f"clear_chat_{current_role}_{patient_id}",
        ):
            st.session_state[state_key] = []
            st.rerun()
