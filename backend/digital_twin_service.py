import pandas as pd

from backend.db import get_connection
from optimization import calculate_response_score


def resolve_patient_id(selected_value):
    """Resolve a patient id from either a numeric id or a patient code."""
    if selected_value is None:
        return None

    if isinstance(selected_value, int):
        return int(selected_value)

    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT id
            FROM patients
            WHERE patient_code = ?
            LIMIT 1
            """,
            (str(selected_value),),
        ).fetchone()
        return int(row[0]) if row is not None else None
    finally:
        conn.close()


def get_evidence_summary(patient_id):
    """Return patient-specific Digital Twin evidence counts."""
    if patient_id is None:
        return {
            "observed_trials": 0,
            "standard_results": 0,
            "ai_results": 0,
            "audiogram_available": False,
        }

    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT
                SUM(CASE
                    WHEN comfort_score IS NOT NULL
                     AND tinnitus_noticeability IS NOT NULL
                     AND source_type NOT IN ('patient_standard_result')
                    THEN 1 ELSE 0 END) AS observed_trials,
                SUM(CASE
                    WHEN source_type IN ('audiologist_manual_result', 'patient_standard_result')
                    THEN 1 ELSE 0 END) AS standard_results,
                SUM(CASE
                    WHEN source_type IN ('audiologist_ai_result', 'patient_ai_result')
                    THEN 1 ELSE 0 END) AS ai_results
            FROM stimulus_trials
            WHERE patient_id = ?
            """,
            (int(patient_id),),
        ).fetchone()

        audiogram_row = conn.execute(
            "SELECT COUNT(*) FROM audiograms WHERE patient_id = ?",
            (int(patient_id),),
        ).fetchone()

        return {
            "observed_trials": int((row[0] if row else 0) or 0),
            "standard_results": int((row[1] if row else 0) or 0),
            "ai_results": int((row[2] if row else 0) or 0),
            "audiogram_available": bool(audiogram_row and int(audiogram_row[0] or 0) > 0),
        }
    finally:
        conn.close()


def get_personalization_result_counts(patient_id):
    """Return completed manual and AI-assisted acoustic-result counts."""
    summary = get_evidence_summary(patient_id)
    return summary["standard_results"], summary["ai_results"]


def _load_evolution_rows(patient_id, patient_view=False):
    if patient_id is None:
        return pd.DataFrame()

    source_filter = """
        source_type IN (
            'audiologist_manual_result',
            'audiologist_ai_result',
            'patient_standard_result',
            'patient_ai_result'
        )
    """ if patient_view else """
        (
            comfort_score IS NOT NULL
            OR source_type IN (
                'audiologist_manual_result',
                'audiologist_ai_result',
                'patient_standard_result',
                'patient_ai_result'
            )
        )
    """

    conn = get_connection()
    try:
        df = pd.read_sql_query(
            f"""
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
              AND {source_filter}
            ORDER BY id ASC
            """,
            conn,
            params=(int(patient_id),),
        )
    finally:
        conn.close()

    if df.empty:
        return df

    scores = []
    for _, row in df.iterrows():
        if pd.notna(row.get("comfort_score")) and pd.notna(row.get("tinnitus_noticeability")):
            try:
                score = calculate_response_score(
                    row.get("comfort_score"),
                    row.get("tinnitus_noticeability"),
                    row.get("listen_again"),
                )
            except Exception:
                score = None
        else:
            score = None
        scores.append(score)

    df["observed_response"] = scores
    return df


def get_audiologist_twin_evolution(patient_id):
    """Return an audiologist-ready Twin Evolution table plus summary metrics."""
    df = _load_evolution_rows(patient_id, patient_view=False)
    summary = get_evidence_summary(patient_id)

    if df.empty:
        return {"summary": summary, "rows": pd.DataFrame()}

    source_labels = {
        "patient_personalization_initial": "Standard · Initial Test",
        "patient_personalization_refined": "Standard · Refined Test",
        "audiologist_manual_result": "Audiologist Manual · Supervised Result",
        "audiologist_ai_result": "AI-Assisted · Supervised Result",
        "patient_standard_result": "Legacy Standard · Final Result",
        "patient_ai_result": "Legacy AI · Patient-Tested Result",
        "audiologist_supervised_manual": "Audiologist · Supervised Test",
        "manual": "Manual Test",
    }

    display = df.copy()
    display["Evidence Source"] = display["source_type"].map(source_labels).fillna(display["source_type"])
    display = display[[
        "evidence_date", "Evidence Source", "stimulus_type",
        "center_frequency_hz", "relative_level", "model_predicted_score",
        "observed_response"
    ]]
    display.columns = [
        "Date", "Evidence Source", "Stimulus", "Frequency (Hz)",
        "Relative Level", "Predicted Response /100", "Observed Response /100"
    ]

    for col in ["Predicted Response /100", "Observed Response /100"]:
        display[col] = pd.to_numeric(display[col], errors="coerce").round(1)

    return {"summary": summary, "rows": display}


def get_patient_twin_evolution(patient_id):
    """Return the patient's completed supervised manual/AI acoustic history."""
    df = _load_evolution_rows(patient_id, patient_view=True)
    if df.empty:
        return {
            "manual_count": 0,
            "ai_count": 0,
            "total_count": 0,
            "rows": pd.DataFrame(),
        }

    pathway = {
        "audiologist_manual_result": "Audiologist Manual",
        "audiologist_ai_result": "AI-Assisted",
        "patient_standard_result": "Legacy Standard",
        "patient_ai_result": "Legacy AI-Assisted",
    }
    df["Pathway"] = df["source_type"].map(pathway)

    display = df[[
        "evidence_date", "Pathway", "stimulus_type", "center_frequency_hz",
        "relative_level", "comfort_score", "tinnitus_noticeability",
        "observed_response"
    ]].copy()
    display.columns = [
        "Date", "Pathway", "Stimulus", "Frequency (Hz)", "Relative Level",
        "Comfort /10", "Tinnitus Noticeability /10", "Observed Response /100"
    ]
    display["Observed Response /100"] = pd.to_numeric(
        display["Observed Response /100"], errors="coerce"
    ).round(1)

    manual_count = int(display["Pathway"].isin(["Audiologist Manual", "Legacy Standard"]).sum())
    ai_count = int(display["Pathway"].isin(["AI-Assisted", "Legacy AI-Assisted"]).sum())

    return {
        "manual_count": manual_count,
        "ai_count": ai_count,
        "total_count": len(display),
        "rows": display,
    }
