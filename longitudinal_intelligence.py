import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from database import get_connection
from optimization import calculate_response_score


# ============================================================
# SMARTTINNITUS — LONGITUDINAL INTELLIGENCE
# ============================================================
#
# Purpose:
# Describe how recorded patient-reported outcomes and acoustic
# response patterns change over time.
#
# IMPORTANT:
# - descriptive research analytics only
# - not proof of treatment efficacy
# - not diagnosis
# - "improving / stable / worsening" are cautious descriptors
#   based only on recorded SmartTinnitus measurements
# ============================================================


# ============================================================
# DATA
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
        params=(patient_id,)
    )

    conn.close()

    if df.empty:
        return None

    return df.iloc[0].to_dict()


def _load_sessions(
    patient_id
):
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            id,
            session_date,
            annoyance_before,
            annoyance_after,
            tinnitus_noticeability_before,
            tinnitus_noticeability_after,
            notes,
            created_at
        FROM sessions
        WHERE patient_id = ?
        ORDER BY session_date ASC, id ASC
        """,
        conn,
        params=(patient_id,)
    )

    conn.close()

    if not df.empty:

        df["session_date"] = pd.to_datetime(
            df["session_date"],
            errors="coerce"
        )

        for column in [
            "annoyance_before",
            "annoyance_after",
            "tinnitus_noticeability_before",
            "tinnitus_noticeability_after"
        ]:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df["annoyance_session_change"] = (
            df["annoyance_after"]
            -
            df["annoyance_before"]
        )

        df["noticeability_session_change"] = (
            df["tinnitus_noticeability_after"]
            -
            df["tinnitus_noticeability_before"]
        )

    return df


def _load_trials(
    patient_id
):
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            id,
            source_type,
            adaptive_session_id,
            stimulus_type,
            center_frequency_hz,
            bandwidth_hz,
            relative_level,
            comfort_score,
            tinnitus_noticeability,
            listen_again,
            tested_at,
            created_at
        FROM stimulus_trials
        WHERE patient_id = ?
        ORDER BY id ASC
        """,
        conn,
        params=(patient_id,)
    )

    conn.close()

    if df.empty:
        return df

    for column in [
        "center_frequency_hz",
        "bandwidth_hz",
        "relative_level",
        "comfort_score",
        "tinnitus_noticeability"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

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

    df["trial_time"] = pd.to_datetime(
        df["tested_at"],
        errors="coerce"
    )

    missing_time = df["trial_time"].isna()

    if missing_time.any():

        df.loc[
            missing_time,
            "trial_time"
        ] = pd.to_datetime(
            df.loc[
                missing_time,
                "created_at"
            ],
            errors="coerce"
        )

    return df


# ============================================================
# TREND HELPERS
# ============================================================


def _safe_float(
    value
):
    try:
        value = float(value)

        if math.isnan(value):
            return None

        return value

    except Exception:
        return None


def _trend_descriptor(
    values,
    lower_is_better=True,
    stable_threshold=0.5
):
    """
    Descriptive first-to-last trend.

    stable_threshold is in scale points for 0-10 measures.
    """

    series = pd.Series(
        values
    ).dropna()

    if len(series) < 2:

        return {
            "label": "Insufficient data",
            "change": None,
            "first": None,
            "last": None
        }

    first = float(
        series.iloc[0]
    )

    last = float(
        series.iloc[-1]
    )

    raw_change = (
        last
        -
        first
    )

    if abs(raw_change) <= stable_threshold:

        label = "Stable"

    else:

        improved = (
            raw_change < 0
            if lower_is_better
            else raw_change > 0
        )

        label = (
            "Improving"
            if improved
            else "Worsening"
        )

    return {
        "label": label,
        "change": raw_change,
        "first": first,
        "last": last
    }


def _response_trend(
    trials_df
):
    valid = trials_df[
        trials_df["response_score"].notna()
    ].copy()

    if len(valid) < 2:

        return {
            "label": "Insufficient data",
            "change": None,
            "early_mean": None,
            "recent_mean": None
        }

    split = max(
        1,
        len(valid) // 2
    )

    early = valid.iloc[:split]
    recent = valid.iloc[split:]

    if recent.empty:
        recent = valid.iloc[-1:]

    early_mean = float(
        early["response_score"].mean()
    )

    recent_mean = float(
        recent["response_score"].mean()
    )

    difference = (
        recent_mean
        -
        early_mean
    )

    if abs(difference) < 5:

        label = "Stable"

    elif difference > 0:

        label = "Improving"

    else:

        label = "Worsening"

    return {
        "label": label,
        "change": difference,
        "early_mean": early_mean,
        "recent_mean": recent_mean
    }


def _comfort_trend(
    trials_df
):
    valid = trials_df[
        trials_df["comfort_score"].notna()
    ].copy()

    if len(valid) < 2:

        return {
            "label": "Insufficient data",
            "change": None,
            "early_mean": None,
            "recent_mean": None
        }

    split = max(
        1,
        len(valid) // 2
    )

    early = valid.iloc[:split]
    recent = valid.iloc[split:]

    if recent.empty:
        recent = valid.iloc[-1:]

    early_mean = float(
        early["comfort_score"].mean()
    )

    recent_mean = float(
        recent["comfort_score"].mean()
    )

    difference = (
        recent_mean
        -
        early_mean
    )

    if abs(difference) < 0.5:

        label = "Stable"

    elif difference > 0:

        label = "Improving"

    else:

        label = "Worsening"

    return {
        "label": label,
        "change": difference,
        "early_mean": early_mean,
        "recent_mean": recent_mean
    }


def _trial_noticeability_trend(
    trials_df
):
    valid = trials_df[
        trials_df["tinnitus_noticeability"].notna()
    ].copy()

    if len(valid) < 2:

        return {
            "label": "Insufficient data",
            "change": None,
            "early_mean": None,
            "recent_mean": None
        }

    split = max(
        1,
        len(valid) // 2
    )

    early = valid.iloc[:split]
    recent = valid.iloc[split:]

    if recent.empty:
        recent = valid.iloc[-1:]

    early_mean = float(
        early["tinnitus_noticeability"].mean()
    )

    recent_mean = float(
        recent["tinnitus_noticeability"].mean()
    )

    difference = (
        recent_mean
        -
        early_mean
    )

    if abs(difference) < 0.5:

        label = "Stable"

    elif difference < 0:

        label = "Improving"

    else:

        label = "Worsening"

    return {
        "label": label,
        "change": difference,
        "early_mean": early_mean,
        "recent_mean": recent_mean
    }


def _preferred_stimulus_period(
    df
):
    valid = df[
        df["response_score"].notna()
    ].copy()

    if valid.empty:
        return None

    summary = (
        valid
        .groupby("stimulus_type")
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
            [
                "mean_response",
                "trial_count"
            ],
            ascending=[
                False,
                False
            ]
        )
    )

    if summary.empty:
        return None

    return summary.iloc[0].to_dict()


def _stimulus_evolution(
    trials_df
):
    valid = trials_df[
        trials_df["response_score"].notna()
    ].copy()

    if len(valid) < 2:

        return {
            "early": None,
            "recent": None,
            "changed": False
        }

    split = max(
        1,
        len(valid) // 2
    )

    early = valid.iloc[:split]
    recent = valid.iloc[split:]

    if recent.empty:
        recent = valid.iloc[-1:]

    early_pref = _preferred_stimulus_period(
        early
    )

    recent_pref = _preferred_stimulus_period(
        recent
    )

    changed = (
        early_pref is not None
        and
        recent_pref is not None
        and
        early_pref["stimulus_type"]
        !=
        recent_pref["stimulus_type"]
    )

    return {
        "early": early_pref,
        "recent": recent_pref,
        "changed": changed
    }


def _frequency_evolution(
    trials_df
):
    valid = trials_df[
        trials_df["response_score"].notna()
        &
        trials_df["center_frequency_hz"].notna()
        &
        trials_df["stimulus_type"].isin(
            [
                "Pure Tone",
                "Narrow-Band Noise"
            ]
        )
    ].copy()

    if len(valid) < 3:

        return {
            "early_center": None,
            "recent_center": None,
            "shift_hz": None
        }

    split = max(
        1,
        len(valid) // 2
    )

    early = valid.iloc[:split]
    recent = valid.iloc[split:]

    if recent.empty:
        recent = valid.iloc[-1:]

    early_top = early.sort_values(
        "response_score",
        ascending=False
    ).head(
        min(
            3,
            len(early)
        )
    )

    recent_top = recent.sort_values(
        "response_score",
        ascending=False
    ).head(
        min(
            3,
            len(recent)
        )
    )

    early_center = float(
        early_top["center_frequency_hz"].median()
    )

    recent_center = float(
        recent_top["center_frequency_hz"].median()
    )

    return {
        "early_center": early_center,
        "recent_center": recent_center,
        "shift_hz": (
            recent_center
            -
            early_center
        )
    }


def _overall_pattern(
    annoyance,
    followup_notice,
    response,
    comfort,
    trial_notice
):
    labels = [
        annoyance["label"],
        followup_notice["label"],
        response["label"],
        comfort["label"],
        trial_notice["label"]
    ]

    valid = [
        label
        for label in labels
        if label
        not in [
            "Insufficient data"
        ]
    ]

    if len(valid) < 2:

        return (
            "Insufficient longitudinal evidence",
            "More repeated measurements are needed before describing an overall pattern."
        )

    improving = valid.count(
        "Improving"
    )

    worsening = valid.count(
        "Worsening"
    )

    stable = valid.count(
        "Stable"
    )

    if (
        improving >= 2
        and
        worsening == 0
    ):

        return (
            "Cautiously improving",
            "Multiple recorded measures move in a favorable descriptive direction."
        )

    if (
        worsening >= 2
        and
        improving == 0
    ):

        return (
            "Cautiously worsening",
            "Multiple recorded measures move in an unfavorable descriptive direction."
        )

    if stable >= max(
        improving,
        worsening
    ):

        return (
            "Predominantly stable",
            "Most available measures show little descriptive change."
        )

    return (
        "Mixed pattern",
        "Recorded measures do not all move in the same direction."
    )


# ============================================================
# INTERPRETATION
# ============================================================


def _build_interpretation(
    sessions_df,
    trials_df
):
    annoyance = _trend_descriptor(
        sessions_df["annoyance_after"]
        if not sessions_df.empty
        else [],
        lower_is_better=True
    )

    followup_notice = _trend_descriptor(
        sessions_df["tinnitus_noticeability_after"]
        if not sessions_df.empty
        else [],
        lower_is_better=True
    )

    response = _response_trend(
        trials_df
    )

    comfort = _comfort_trend(
        trials_df
    )

    trial_notice = _trial_noticeability_trend(
        trials_df
    )

    stimulus = _stimulus_evolution(
        trials_df
    )

    frequency = _frequency_evolution(
        trials_df
    )

    overall, overall_explanation = _overall_pattern(
        annoyance,
        followup_notice,
        response,
        comfort,
        trial_notice
    )

    return {
        "annoyance": annoyance,
        "followup_notice": followup_notice,
        "response": response,
        "comfort": comfort,
        "trial_notice": trial_notice,
        "stimulus": stimulus,
        "frequency": frequency,
        "overall": overall,
        "overall_explanation": overall_explanation
    }


# ============================================================
# CHARTS
# ============================================================


def _plot_followup_trends(
    sessions_df
):
    fig, ax = plt.subplots(
        figsize=(9.5, 4.5)
    )

    ax.plot(
        sessions_df["session_date"],
        sessions_df["annoyance_after"],
        marker="o",
        label="Annoyance after session"
    )

    ax.plot(
        sessions_df["session_date"],
        sessions_df["tinnitus_noticeability_after"],
        marker="o",
        label="Noticeability after session"
    )

    ax.set_ylim(
        -0.5,
        10.5
    )

    ax.set_xlabel(
        "Session date"
    )

    ax.set_ylabel(
        "Patient-reported score (0–10)"
    )

    ax.set_title(
        "Longitudinal Follow-up Measures"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.25
    )

    fig.autofmt_xdate()
    fig.tight_layout()

    return fig


def _plot_response_history(
    trials_df
):
    valid = trials_df[
        trials_df["response_score"].notna()
    ].copy()

    fig, ax = plt.subplots(
        figsize=(9.5, 4.5)
    )

    ax.plot(
        range(
            1,
            len(valid) + 1
        ),
        valid["response_score"],
        marker="o",
        label="Observed response score"
    )

    if len(valid) >= 3:

        rolling = valid[
            "response_score"
        ].rolling(
            window=3,
            min_periods=1
        ).mean()

        ax.plot(
            range(
                1,
                len(valid) + 1
            ),
            rolling,
            linewidth=2,
            label="3-trial moving average"
        )

    ax.set_ylim(
        0,
        100
    )

    ax.set_xlabel(
        "Chronological scored trial"
    )

    ax.set_ylabel(
        "Prototype response score /100"
    )

    ax.set_title(
        "Acoustic Response Evolution"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.25
    )

    fig.tight_layout()

    return fig


# ============================================================
# UI
# ============================================================


def show_longitudinal_intelligence_page(
    current_role,
    auth_patient_id=None
):
    st.title(
        "Longitudinal Intelligence"
    )

    st.caption(
        "A longitudinal overview of patient-reported measures, acoustic response, "
        "comfort, noticeability and stimulus preference over time."
    )

    st.warning(
        "Longitudinal labels such as improving, stable, or worsening describe recorded SmartTinnitus measurements only. "
        "They do not establish clinical improvement, deterioration, diagnosis, or treatment effectiveness."
    )

    # --------------------------------------------------------
    # Resolve patient
    # --------------------------------------------------------

    if current_role == "patient":

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

        patient_options = {
            f"{row['patient_code']} (ID {int(row['id'])})":
                int(row["id"])
            for _, row
            in patients_df.iterrows()
        }

        selected = st.selectbox(
            "Patient",
            list(
                patient_options.keys()
            ),
            key="longitudinal_patient"
        )

        patient_id = patient_options[
            selected
        ]

        patient = _load_patient(
            patient_id
        )

    sessions_df = _load_sessions(
        patient_id
    )

    trials_df = _load_trials(
        patient_id
    )

    interpretation = _build_interpretation(
        sessions_df,
        trials_df
    )

    # --------------------------------------------------------
    # Snapshot
    # --------------------------------------------------------

    st.markdown(
        "## Longitudinal Snapshot"
    )

    st.caption(
        "Current data volume and overall descriptive pattern."
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Patient",
        patient["patient_code"]
    )

    c2.metric(
        "Follow-up Sessions",
        len(
            sessions_df
        )
    )

    c3.metric(
        "Scored Acoustic Trials",
        int(
            trials_df["response_score"].notna().sum()
        )
        if not trials_df.empty
        else 0
    )

    c4.metric(
        "Overall Descriptive Pattern",
        interpretation[
            "overall"
        ]
    )

    st.info(
        interpretation[
            "overall_explanation"
        ]
    )

    # --------------------------------------------------------
    # Key trends
    # --------------------------------------------------------

    st.markdown(
        "## Key Trends"
    )

    st.caption(
        "Trend labels are calculated from repeated SmartTinnitus measurements only."
    )

    t1, t2, t3 = st.columns(
        3
    )

    annoyance = interpretation[
        "annoyance"
    ]

    followup_notice = interpretation[
        "followup_notice"
    ]

    response = interpretation[
        "response"
    ]

    t1.metric(
        "Follow-up Annoyance",
        annoyance[
            "label"
        ],
        (
            f"{annoyance['change']:+.1f} points"
            if annoyance[
                "change"
            ]
            is not None
            else None
        ),
        delta_color="inverse"
    )

    t2.metric(
        "Follow-up Noticeability",
        followup_notice[
            "label"
        ],
        (
            f"{followup_notice['change']:+.1f} points"
            if followup_notice[
                "change"
            ]
            is not None
            else None
        ),
        delta_color="inverse"
    )

    t3.metric(
        "Prototype Response",
        response[
            "label"
        ],
        (
            f"{response['change']:+.1f} /100"
            if response[
                "change"
            ]
            is not None
            else None
        )
    )

    t1, t2, t3 = st.columns(
        3
    )

    comfort = interpretation[
        "comfort"
    ]

    trial_notice = interpretation[
        "trial_notice"
    ]

    stimulus = interpretation[
        "stimulus"
    ]

    t1.metric(
        "Acoustic Trial Comfort",
        comfort[
            "label"
        ],
        (
            f"{comfort['change']:+.1f} points"
            if comfort[
                "change"
            ]
            is not None
            else None
        )
    )

    t2.metric(
        "Trial Tinnitus Noticeability",
        trial_notice[
            "label"
        ],
        (
            f"{trial_notice['change']:+.1f} points"
            if trial_notice[
                "change"
            ]
            is not None
            else None
        ),
        delta_color="inverse"
    )

    recent_stimulus = (
        stimulus[
            "recent"
        ][
            "stimulus_type"
        ]
        if stimulus[
            "recent"
        ]
        is not None
        else "Insufficient data"
    )

    t3.metric(
        "Recent Preferred Stimulus",
        recent_stimulus
    )

    # --------------------------------------------------------
    # Follow-up intelligence
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "## Follow-up Intelligence"
    )

    if sessions_df.empty:

        st.info(
            "No follow-up sessions are recorded yet."
        )

    else:

        if len(
            sessions_df
        ) >= 2:

            st.pyplot(
                _plot_followup_trends(
                    sessions_df
                ),
                clear_figure=True,
                use_container_width=True
            )

        display_sessions = sessions_df[
            [
                "session_date",
                "annoyance_before",
                "annoyance_after",
                "annoyance_session_change",
                "tinnitus_noticeability_before",
                "tinnitus_noticeability_after",
                "noticeability_session_change",
                "notes"
            ]
        ].copy()

        display_sessions.columns = [
            "Date",
            "Annoyance Before",
            "Annoyance After",
            "Within-Session Annoyance Δ",
            "Noticeability Before",
            "Noticeability After",
            "Within-Session Noticeability Δ",
            "Notes"
        ]

        st.dataframe(
            display_sessions,
            use_container_width=True,
            hide_index=True
        )

        if len(
            sessions_df
        ) == 1:

            st.caption(
                "One follow-up session allows within-session comparison, but repeated sessions are needed for longitudinal trend classification."
            )

    # --------------------------------------------------------
    # Acoustic-response intelligence
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "## Acoustic Response Evolution"
    )

    valid_trials = (
        trials_df[
            trials_df[
                "response_score"
            ].notna()
        ].copy()
        if not trials_df.empty
        else pd.DataFrame()
    )

    if valid_trials.empty:

        st.info(
            "No scored acoustic trials are available."
        )

    else:

        st.pyplot(
            _plot_response_history(
                trials_df
            ),
            clear_figure=True,
            use_container_width=True
        )

        r1, r2, r3 = st.columns(
            3
        )

        r1.metric(
            "Early Mean Response",
            (
                f"{response['early_mean']:.1f} /100"
                if response[
                    "early_mean"
                ]
                is not None
                else "N/A"
            )
        )

        r2.metric(
            "Recent Mean Response",
            (
                f"{response['recent_mean']:.1f} /100"
                if response[
                    "recent_mean"
                ]
                is not None
                else "N/A"
            )
        )

        r3.metric(
            "Response Pattern",
            response[
                "label"
            ]
        )

    # --------------------------------------------------------
    # Preference evolution
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "## Acoustic Preference Evolution"
    )

    early_pref = stimulus[
        "early"
    ]

    recent_pref = stimulus[
        "recent"
    ]

    p1, p2, p3 = st.columns(
        3
    )

    p1.metric(
        "Earlier Preferred Stimulus",
        (
            early_pref[
                "stimulus_type"
            ]
            if early_pref is not None
            else "Insufficient data"
        )
    )

    p2.metric(
        "Recent Preferred Stimulus",
        (
            recent_pref[
                "stimulus_type"
            ]
            if recent_pref is not None
            else "Insufficient data"
        )
    )

    p3.metric(
        "Preference Shift",
        (
            "Changed"
            if stimulus[
                "changed"
            ]
            else
            (
                "Stable"
                if early_pref is not None
                and
                recent_pref is not None
                else "Insufficient data"
            )
        )
    )

    frequency = interpretation[
        "frequency"
    ]

    if (
        frequency[
            "early_center"
        ]
        is not None
        and
        frequency[
            "recent_center"
        ]
        is not None
    ):

        f1, f2, f3 = st.columns(
            3
        )

        f1.metric(
            "Earlier High-Response Center",
            f"{frequency['early_center']:.0f} Hz"
        )

        f2.metric(
            "Recent High-Response Center",
            f"{frequency['recent_center']:.0f} Hz"
        )

        f3.metric(
            "Descriptive Frequency Shift",
            f"{frequency['shift_hz']:+.0f} Hz"
        )

        st.caption(
            "Frequency shift is estimated from the median frequency of the strongest frequency-specific trials in the earlier and recent halves of the record."
        )

    # --------------------------------------------------------
    # Automatic interpretation
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "## Automatic Longitudinal Interpretation"
    )

    st.caption(
        "Plain-language synthesis of the recorded longitudinal pattern."
    )

    observations = []

    if annoyance[
        "label"
    ] != "Insufficient data":

        observations.append(
            (
                f"Follow-up annoyance is **{annoyance['label'].lower()}** "
                f"from {annoyance['first']:.1f} to {annoyance['last']:.1f}."
            )
        )

    if followup_notice[
        "label"
    ] != "Insufficient data":

        observations.append(
            (
                f"Follow-up tinnitus noticeability is "
                f"**{followup_notice['label'].lower()}** "
                f"from {followup_notice['first']:.1f} to "
                f"{followup_notice['last']:.1f}."
            )
        )

    if response[
        "label"
    ] != "Insufficient data":

        observations.append(
            (
                f"The mean prototype response score changed from "
                f"**{response['early_mean']:.1f}/100** in the earlier trials "
                f"to **{response['recent_mean']:.1f}/100** in the recent trials."
            )
        )

    if comfort[
        "label"
    ] != "Insufficient data":

        observations.append(
            (
                f"Average acoustic comfort is descriptively "
                f"**{comfort['label'].lower()}**."
            )
        )

    if trial_notice[
        "label"
    ] != "Insufficient data":

        observations.append(
            (
                f"Tinnitus noticeability during acoustic trials is descriptively "
                f"**{trial_notice['label'].lower()}**."
            )
        )

    if stimulus[
        "changed"
    ]:

        observations.append(
            (
                f"The preferred stimulus type shifted from "
                f"**{early_pref['stimulus_type']}** to "
                f"**{recent_pref['stimulus_type']}**."
            )
        )

    elif (
        early_pref is not None
        and
        recent_pref is not None
    ):

        observations.append(
            (
                f"The preferred stimulus type remained "
                f"**{recent_pref['stimulus_type']}** across the earlier and recent record."
            )
        )

    if frequency[
        "shift_hz"
    ] is not None:

        observations.append(
            (
                f"The center of the stronger frequency-specific responses shifted "
                f"by approximately **{frequency['shift_hz']:+.0f} Hz**."
            )
        )

    if observations:

        for observation in observations:

            st.markdown(
                f"- {observation}"
            )

    else:

        st.info(
            "More repeated measurements are required for an automatic longitudinal interpretation."
        )

    st.markdown(
        "### Overall Pattern"
    )

    st.success(
        interpretation[
            "overall"
        ]
    )

    st.caption(
        interpretation[
            "overall_explanation"
        ]
        +
        " This is a descriptive research output and must not be interpreted as demonstrated treatment efficacy."
    )
