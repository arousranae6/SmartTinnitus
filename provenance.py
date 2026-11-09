import os
from datetime import datetime


def now_timestamp():
    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def get_latest_profile_id(
    conn,
    patient_id
):
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id
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

    return (
        int(
            row[0]
        )
        if row is not None
        else None
    )


def get_model_provenance(
    model_file,
    saved_metrics=None
):
    saved_metrics = (
        saved_metrics
        if isinstance(
            saved_metrics,
            dict
        )
        else {}
    )

    model_version = (
        saved_metrics.get(
            "model_version"
        )
        or
        saved_metrics.get(
            "version"
        )
    )

    model_trained_at = (
        saved_metrics.get(
            "model_trained_at"
        )
        or
        saved_metrics.get(
            "trained_at"
        )
        or
        saved_metrics.get(
            "training_completed_at"
        )
    )

    if (
        model_file
        and
        os.path.exists(
            model_file
        )
    ):
        modified_time = os.path.getmtime(
            model_file
        )

        artifact_time = datetime.fromtimestamp(
            modified_time
        )

        if not model_version:
            model_version = (
                "model_"
                +
                artifact_time.strftime(
                    "%Y%m%d_%H%M%S"
                )
            )

        if not model_trained_at:
            model_trained_at = (
                artifact_time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

    return {
        "model_version":
            model_version,

        "model_trained_at":
            model_trained_at
    }
