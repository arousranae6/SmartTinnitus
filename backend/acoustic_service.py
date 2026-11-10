import pandas as pd

from backend.db import get_connection
from optimization import calculate_response_score
from adaptive_personalization import generate_adaptive_candidate_stimuli
from ml_model import load_model, model_exists, predict_responses_batch, get_saved_model_metrics
from provenance import get_model_provenance, now_timestamp

from backend.acoustic_assignment_service import (
    create_assignment,
    get_pending_assignment,
    submit_assignment_feedback,
    list_assignment_feedback,
)

MODEL_FILE = "smarttinnitus_response_model.joblib"


def _patient_trial_history(connection, patient_id):
    history_df = pd.read_sql_query(
        """
        SELECT stimulus_type, center_frequency_hz, bandwidth_hz, relative_level,
               comfort_score, tinnitus_noticeability, listen_again
        FROM stimulus_trials
        WHERE patient_id = ?
        ORDER BY id ASC
        """,
        connection,
        params=(int(patient_id),),
    )
    if history_df.empty:
        history_df["response_score"] = []
        return history_df

    scores = []
    for _, row in history_df.iterrows():
        try:
            scores.append(calculate_response_score(
                row["comfort_score"],
                row["tinnitus_noticeability"],
                row["listen_again"],
            ))
        except Exception:
            scores.append(None)
    history_df["response_score"] = scores
    return history_df


def get_acoustic_context(patient_id):
    conn = get_connection()
    try:
        profile_row = conn.execute(
            """
            SELECT id, estimated_pitch_hz, loudness_relative, tinnitus_type,
                   laterality, temporal_pattern, duration_months, annoyance_score,
                   sleep_score, concentration_score, stress_score, created_at
            FROM tinnitus_profiles
            WHERE patient_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (int(patient_id),),
        ).fetchone()

        profile = None
        if profile_row is not None:
            keys = [
                "id", "estimated_pitch_hz", "loudness_relative", "tinnitus_type",
                "laterality", "temporal_pattern", "duration_months", "annoyance_score",
                "sleep_score", "concentration_score", "stress_score", "created_at",
            ]
            profile = dict(zip(keys, profile_row))

        audiogram_row = conn.execute(
            "SELECT COUNT(*), MAX(created_at) FROM audiograms WHERE patient_id = ?",
            (int(patient_id),),
        ).fetchone()
        audiogram_count = int((audiogram_row[0] if audiogram_row else 0) or 0)
        trial_count = conn.execute(
            "SELECT COUNT(*) FROM stimulus_trials WHERE patient_id = ?",
            (int(patient_id),),
        ).fetchone()[0]

        return {
            "patient_id": int(patient_id),
            "profile": profile,
            "has_audiogram": audiogram_count > 0,
            "audiogram_count": audiogram_count,
            "latest_audiogram_at": audiogram_row[1] if audiogram_row else None,
            "trial_count": int(trial_count or 0),
        }
    finally:
        conn.close()


def generate_ai_candidates(patient_id, limit=5):
    if not model_exists():
        raise RuntimeError("No trained AI model is available.")

    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT tinnitus_type, laterality, temporal_pattern, duration_months,
                   annoyance_score, sleep_score, concentration_score, stress_score,
                   estimated_pitch_hz, loudness_relative
            FROM tinnitus_profiles
            WHERE patient_id = ?
            ORDER BY id DESC LIMIT 1
            """,
            (int(patient_id),),
        ).fetchone()
        if row is None:
            raise ValueError("No tinnitus profile is available for this patient.")

        patient_features = {
            "tinnitus_type": row[0],
            "laterality": row[1],
            "temporal_pattern": row[2],
            "duration_months": row[3],
            "annoyance_score": row[4],
            "sleep_score": row[5],
            "concentration_score": row[6],
            "stress_score": row[7],
            "estimated_pitch_hz": row[8],
            "loudness_relative": row[9],
        }

        history_df = _patient_trial_history(conn, patient_id)
        candidate_stimuli, _ = generate_adaptive_candidate_stimuli(
            tinnitus_pitch=float(row[8]) if row[8] is not None else 4000.0,
            trials_df=history_df,
        )
        predictions = predict_responses_batch(
            patient_features,
            candidate_stimuli,
            model=load_model(),
        )

        ranked = []
        for candidate, prediction in zip(candidate_stimuli, predictions):
            ranked.append({
                "stimulus_type": candidate["stimulus_type"],
                "frequency": float(candidate["center_frequency_hz"]),
                "bandwidth": (
                    float(candidate["bandwidth_hz"])
                    if candidate.get("bandwidth_hz") is not None else None
                ),
                "level": float(candidate["relative_level"]),
                "predicted_score": float(prediction),
            })

        ranked.sort(key=lambda item: item["predicted_score"], reverse=True)
        if not ranked:
            raise ValueError("The AI model did not produce candidate configurations.")

        return {
            "candidates": ranked[:max(1, int(limit))],
            "created_at": now_timestamp(),
            "provenance": get_model_provenance(MODEL_FILE, get_saved_model_metrics()),
        }
    finally:
        conn.close()


def send_candidate(**kwargs):
    if not kwargs.get("recommendation_created_at"):
        kwargs["recommendation_created_at"] = now_timestamp()
    return create_assignment(**kwargs)


def get_patient_pending_assignment(patient_id):
    return get_pending_assignment(patient_id)


def submit_patient_feedback(
    assignment_id,
    patient_id,
    comfort_score,
    tinnitus_noticeability,
    listen_again,
    patient_comment="",
):
    return submit_assignment_feedback(
        assignment_id=assignment_id,
        patient_id=patient_id,
        comfort_score=comfort_score,
        tinnitus_noticeability=tinnitus_noticeability,
        listen_again=listen_again,
        patient_comment=patient_comment,
    )


def get_patient_feedback(patient_id, limit=20):
    return list_assignment_feedback(patient_id, limit=limit)
