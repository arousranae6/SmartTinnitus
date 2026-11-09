import os
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    KeepTogether
)

# ============================================================
# SMARTTINNITUS PROFESSIONAL REPORT
# ============================================================
#
# IMPORTANT:
# - Research / engineering prototype.
# - No diagnosis or validated treatment recommendation.
# - Relative stimulus level is digital amplitude, not dB SPL/dB HL.
# - Audiogram thresholds are clinician-entered dB HL.
# - Response score is a 0-100 prototype engineering score:
#     40% comfort
#     50% low tinnitus noticeability
#     10% willingness to listen again
# ============================================================


PAGE_WIDTH, PAGE_HEIGHT = A4

NAVY = colors.HexColor("#12364A")
TEAL = colors.HexColor("#1D7A88")
LIGHT_TEAL = colors.HexColor("#EAF5F6")
LIGHT_BLUE = colors.HexColor("#EEF5FB")
LIGHT_GREY = colors.HexColor("#F5F7F8")
MID_GREY = colors.HexColor("#71818A")
BORDER = colors.HexColor("#D8E3E7")
GREEN = colors.HexColor("#E8F5ED")
AMBER = colors.HexColor("#FFF4D8")


def _safe(value, default="Not available"):
    if value is None:
        return default

    try:
        if value != value:
            return default
    except Exception:
        pass

    text = str(value).strip()

    if text.lower() in {
        "",
        "none",
        "nan",
        "nat"
    }:
        return default

    return text


def _fmt_num(
    value,
    decimals=1,
    suffix=""
):
    if value is None:
        return "Not available"

    try:
        if value != value:
            return "Not available"
    except Exception:
        pass

    try:
        number = float(value)

        if decimals == 0:
            text = f"{number:.0f}"
        else:
            text = f"{number:.{decimals}f}"

        return f"{text}{suffix}"

    except Exception:
        return _safe(value)


def _response_score(trial):
    """
    SmartTinnitus 0-100 prototype engineering score:
    40% comfort + 50% low tinnitus noticeability
    + 10% willingness to listen again.
    """
    try:
        comfort = max(
            0.0,
            min(
                10.0,
                float(
                    trial.get(
                        "comfort"
                    )
                )
            )
        )

        noticeability = max(
            0.0,
            min(
                10.0,
                float(
                    trial.get(
                        "noticeability"
                    )
                )
            )
        )

        listen_again = (
            str(
                trial.get(
                    "listen_again"
                )
            )
            .strip()
            .lower()
        )

        comfort_component = (
            comfort
            /
            10.0
            *
            40.0
        )

        noticeability_component = (
            (
                10.0
                -
                noticeability
            )
            /
            10.0
            *
            50.0
        )

        listen_again_component = (
            10.0
            if listen_again
            ==
            "yes"
            else 0.0
        )

        return round(
            float(
                comfort_component
                +
                noticeability_component
                +
                listen_again_component
            ),
            2
        )

    except Exception:
        return None


def _section_title(text, styles):
    return Paragraph(
        text,
        styles["SectionTitle"]
    )


def _kv_table(
    rows,
    styles,
    widths=None
):
    data = []

    for label, value in rows:
        data.append(
            [
                Paragraph(
                    f"<b>{label}</b>",
                    styles["SmallBody"]
                ),
                Paragraph(
                    _safe(value),
                    styles["SmallBody"]
                )
            ]
        )

    table = Table(
        data,
        colWidths=(
            widths
            if widths is not None
            else [
                55 * mm,
                115 * mm
            ]
        ),
        hAlign="LEFT"
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    LIGHT_GREY
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    BORDER
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ]
        )
    )

    return table


def _data_table(
    headers,
    rows,
    styles,
    col_widths=None,
    font_size=7
):
    data = [
        [
            Paragraph(
                f"<b>{header}</b>",
                styles["TableHeader"]
            )
            for header
            in headers
        ]
    ]

    for row in rows:
        data.append(
            [
                Paragraph(
                    _safe(value, "—"),
                    styles["TableCell"]
                )
                for value
                in row
            ]
        )

    table = Table(
        data,
        colWidths=col_widths,
        repeatRows=1,
        hAlign="LEFT"
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    NAVY
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    BORDER
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        LIGHT_GREY
                    ]
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    font_size
                )
            ]
        )
    )

    return table


def _create_audiogram_image(
    audiogram_data,
    output_path
):
    if not audiogram_data:
        return None

    frequencies = audiogram_data.get(
        "frequencies"
    )
    right = audiogram_data.get(
        "right_ear"
    )
    left = audiogram_data.get(
        "left_ear"
    )

    if not frequencies or not right or not left:
        return None

    x = list(
        range(
            len(
                frequencies
            )
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            7.4,
            3.8
        )
    )

    ax.plot(
        x,
        right,
        marker="o",
        label="Right ear"
    )

    ax.plot(
        x,
        left,
        marker="x",
        label="Left ear"
    )

    tinnitus_pitch = audiogram_data.get(
        "tinnitus_pitch"
    )

    if tinnitus_pitch is not None:
        try:
            import numpy as np

            log_freqs = np.log2(
                frequencies
            )

            pitch_pos = np.interp(
                np.log2(
                    float(
                        tinnitus_pitch
                    )
                ),
                log_freqs,
                x
            )

            ax.axvline(
                pitch_pos,
                linestyle="--",
                linewidth=1,
                label="Estimated tinnitus pitch"
            )
        except Exception:
            pass

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "250",
            "500",
            "1k",
            "2k",
            "4k",
            "8k"
        ][:len(x)]
    )

    ax.set_ylim(
        120,
        -10
    )

    ax.set_xlabel(
        "Frequency (Hz)"
    )

    ax.set_ylabel(
        "Threshold (dB HL)"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return output_path


def _create_longitudinal_image(
    sessions,
    output_path
):
    if not sessions or len(
        sessions
    ) < 2:
        return None

    session_numbers = list(
        range(
            1,
            len(
                sessions
            )
            +
            1
        )
    )

    annoyance_before = [
        s.get(
            "annoyance_before"
        )
        for s
        in sessions
    ]

    annoyance_after = [
        s.get(
            "annoyance_after"
        )
        for s
        in sessions
    ]

    notice_before = [
        s.get(
            "noticeability_before"
        )
        for s
        in sessions
    ]

    notice_after = [
        s.get(
            "noticeability_after"
        )
        for s
        in sessions
    ]

    fig, ax = plt.subplots(
        figsize=(
            7.4,
            3.8
        )
    )

    ax.plot(
        session_numbers,
        annoyance_before,
        marker="o",
        label="Annoyance before"
    )

    ax.plot(
        session_numbers,
        annoyance_after,
        marker="o",
        label="Annoyance after"
    )

    ax.plot(
        session_numbers,
        notice_before,
        marker="x",
        label="Noticeability before"
    )

    ax.plot(
        session_numbers,
        notice_after,
        marker="x",
        label="Noticeability after"
    )

    ax.set_xticks(
        session_numbers
    )

    ax.set_xlabel(
        "Session"
    )

    ax.set_ylabel(
        "Patient-reported score (0-10)"
    )

    ax.set_ylim(
        0,
        10
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        fontsize=8,
        ncol=2
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return output_path


def _build_fingerprint(
    profile,
    trials,
    decisions
):
    fingerprint = {
        "tinnitus_pitch":
            profile.get(
                "estimated_pitch_hz"
            )
            if profile
            else None,

        "relative_loudness":
            profile.get(
                "loudness_relative"
            )
            if profile
            else None,

        "tinnitus_type":
            profile.get(
                "tinnitus_type"
            )
            if profile
            else None,

        "laterality":
            profile.get(
                "laterality"
            )
            if profile
            else None,

        "best_stimulus":
            None,

        "best_frequency":
            None,

        "best_bandwidth":
            None,

        "best_level":
            None,

        "average_comfort":
            None,

        "average_noticeability":
            None,

        "best_response_score":
            None,

        "latest_clinician_decision":
            None,

        "latest_clinician_stimulus":
            None,

        "latest_clinician_frequency":
            None,

        "latest_clinician_bandwidth":
            None,

        "latest_clinician_level":
            None,

        "latest_final_prediction":
            None
    }

    scored_trials = []

    for trial in (
        trials
        or []
    ):
        score = _response_score(
            trial
        )

        if score is not None:
            scored_trials.append(
                (
                    score,
                    trial
                )
            )

    if scored_trials:
        scored_trials.sort(
            key=lambda item: item[0],
            reverse=True
        )

        best_score, best_trial = (
            scored_trials[
                0
            ]
        )

        fingerprint[
            "best_stimulus"
        ] = best_trial.get(
            "stimulus_type"
        )

        if (
            best_trial.get(
                "stimulus_type"
            )
            !=
            "White Noise"
        ):
            fingerprint[
                "best_frequency"
            ] = best_trial.get(
                "frequency"
            )

        if (
            best_trial.get(
                "stimulus_type"
            )
            ==
            "Narrow-Band Noise"
        ):
            fingerprint[
                "best_bandwidth"
            ] = best_trial.get(
                "bandwidth"
            )

        fingerprint[
            "best_level"
        ] = best_trial.get(
            "level"
        )

        fingerprint[
            "best_response_score"
        ] = best_score

        comforts = [
            float(
                t.get(
                    "comfort"
                )
            )
            for t
            in trials
            if t.get(
                "comfort"
            )
            is not None
        ]

        notices = [
            float(
                t.get(
                    "noticeability"
                )
            )
            for t
            in trials
            if t.get(
                "noticeability"
            )
            is not None
        ]

        if comforts:
            fingerprint[
                "average_comfort"
            ] = sum(
                comforts
            ) / len(
                comforts
            )

        if notices:
            fingerprint[
                "average_noticeability"
            ] = sum(
                notices
            ) / len(
                notices
            )

    if decisions:
        latest = sorted(
            decisions,
            key=lambda d: (
                d.get(
                    "decision_id"
                )
                or
                0
            ),
            reverse=True
        )[0]

        fingerprint[
            "latest_clinician_decision"
        ] = latest.get(
            "decision"
        )

        fingerprint[
            "latest_clinician_stimulus"
        ] = latest.get(
            "final_stimulus_type"
        )

        fingerprint[
            "latest_clinician_frequency"
        ] = latest.get(
            "final_frequency_hz"
        )

        fingerprint[
            "latest_clinician_bandwidth"
        ] = latest.get(
            "final_bandwidth_hz"
        )

        fingerprint[
            "latest_clinician_level"
        ] = latest.get(
            "final_relative_level"
        )

        fingerprint[
            "latest_final_prediction"
        ] = latest.get(
            "final_predicted_score"
        )

    return fingerprint


def _trial_by_decision(
    trials
):
    result = {}

    for trial in (
        trials
        or []
    ):
        decision_id = trial.get(
            "decision_id"
        )

        if decision_id is not None:
            result[
                decision_id
            ] = trial

    return result


def generate_patient_report(
    file_path,
    patient_info,
    tinnitus_profile,
    stimulus_trials,
    sessions,
    audiogram_data=None,
    acoustic_fingerprint=None,
    audiologist_decisions=None
):
    """
    Generate the SmartTinnitus professional PDF.

    The last two arguments are optional to preserve compatibility
    with older app calls.
    """

    stimulus_trials = (
        stimulus_trials
        or []
    )

    sessions = (
        sessions
        or []
    )

    audiologist_decisions = (
        audiologist_decisions
        or []
    )

    if acoustic_fingerprint is None:
        acoustic_fingerprint = (
            _build_fingerprint(
                tinnitus_profile,
                stimulus_trials,
                audiologist_decisions
            )
        )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=4
        )
    )

    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=MID_GREY,
            alignment=TA_CENTER,
            spaceAfter=14
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=TEAL,
            spaceBefore=8,
            spaceAfter=8
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor(
                "#24343D"
            )
        )
    )

    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=6.5,
            leading=8,
            textColor=colors.white
        )
    )

    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=6.4,
            leading=8,
            textColor=colors.HexColor(
                "#24343D"
            )
        )
    )

    styles.add(
        ParagraphStyle(
            name="Disclaimer",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor(
                "#44545D"
            ),
            backColor=LIGHT_BLUE,
            borderColor=BORDER,
            borderWidth=0.5,
            borderPadding=7,
            spaceBefore=8,
            spaceAfter=8
        )
    )

    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="SmartTinnitus Professional Report"
    )

    story = []

    patient_code = _safe(
        patient_info.get(
            "patient_code"
        )
        if patient_info
        else None
    )

    story.append(
        Paragraph(
            "SmartTinnitus",
            styles["ReportTitle"]
        )
    )

    story.append(
        Paragraph(
            "Professional Tinnitus Profiling, Acoustic Personalization and Longitudinal Report",
            styles["Subtitle"]
        )
    )

    story.append(
        _kv_table(
            [
                (
                    "Patient",
                    patient_code
                ),
                (
                    "Report generated",
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                ),
                (
                    "Report status",
                    "Research / engineering prototype"
                )
            ],
            styles
        )
    )

    story.append(
        Spacer(
            1,
            10
        )
    )

    # ========================================================
    # 1. PATIENT / TINNITUS PROFILE
    # ========================================================

    story.append(
        _section_title(
            "1. Patient and Tinnitus Profile",
            styles
        )
    )

    story.append(
        _kv_table(
            [
                (
                    "Patient code",
                    patient_code
                ),
                (
                    "Age",
                    _safe(
                        patient_info.get(
                            "age"
                        )
                        if patient_info
                        else None
                    )
                ),
                (
                    "Sex",
                    _safe(
                        patient_info.get(
                            "sex"
                        )
                        if patient_info
                        else None
                    )
                ),
                (
                    "Laterality",
                    _safe(
                        tinnitus_profile.get(
                            "laterality"
                        )
                        if tinnitus_profile
                        else None
                    )
                ),
                (
                    "Tinnitus type",
                    _safe(
                        tinnitus_profile.get(
                            "tinnitus_type"
                        )
                        if tinnitus_profile
                        else None
                    )
                ),
                (
                    "Temporal pattern",
                    _safe(
                        tinnitus_profile.get(
                            "temporal_pattern"
                        )
                        if tinnitus_profile
                        else None
                    )
                ),
                (
                    "Duration",
                    (
                        _fmt_num(
                            tinnitus_profile.get(
                                "duration_months"
                            ),
                            0,
                            " months"
                        )
                        if tinnitus_profile
                        else "Not available"
                    )
                ),
                (
                    "Annoyance",
                    (
                        _fmt_num(
                            tinnitus_profile.get(
                                "annoyance_score"
                            ),
                            0,
                            " / 10"
                        )
                        if tinnitus_profile
                        else "Not available"
                    )
                ),
                (
                    "Sleep impact",
                    (
                        _fmt_num(
                            tinnitus_profile.get(
                                "sleep_score"
                            ),
                            0,
                            " / 10"
                        )
                        if tinnitus_profile
                        else "Not available"
                    )
                ),
                (
                    "Concentration impact",
                    (
                        _fmt_num(
                            tinnitus_profile.get(
                                "concentration_score"
                            ),
                            0,
                            " / 10"
                        )
                        if tinnitus_profile
                        else "Not available"
                    )
                ),
                (
                    "Stress",
                    (
                        _fmt_num(
                            tinnitus_profile.get(
                                "stress_score"
                            ),
                            0,
                            " / 10"
                        )
                        if tinnitus_profile
                        else "Not available"
                    )
                ),
                (
                    "Estimated tinnitus pitch",
                    (
                        _fmt_num(
                            tinnitus_profile.get(
                                "estimated_pitch_hz"
                            ),
                            0,
                            " Hz"
                        )
                        if tinnitus_profile
                        else "Not available"
                    )
                ),
                (
                    "Relative tinnitus loudness",
                    (
                        _fmt_num(
                            tinnitus_profile.get(
                                "loudness_relative"
                            ),
                            2
                        )
                        if tinnitus_profile
                        else "Not available"
                    )
                )
            ],
            styles
        )
    )

    # ========================================================
    # 2. ACOUSTIC FINGERPRINT
    # ========================================================

    story.append(
        _section_title(
            "2. Acoustic Fingerprint",
            styles
        )
    )

    story.append(
        Paragraph(
            "The Acoustic Fingerprint is an evolving descriptive summary of recorded tinnitus characteristics and acoustic-response history. It is not a diagnostic biomarker.",
            styles["SmallBody"]
        )
    )

    story.append(
        Spacer(
            1,
            5
        )
    )

    story.append(
        _kv_table(
            [
                (
                    "Best recorded stimulus",
                    acoustic_fingerprint.get(
                        "best_stimulus"
                    )
                ),
                (
                    "Best recorded frequency",
                    (
                        _fmt_num(
                            acoustic_fingerprint.get(
                                "best_frequency"
                            ),
                            0,
                            " Hz"
                        )
                        if acoustic_fingerprint.get(
                            "best_frequency"
                        )
                        is not None
                        else "N/A"
                    )
                ),
                (
                    "Best bandwidth",
                    (
                        _fmt_num(
                            acoustic_fingerprint.get(
                                "best_bandwidth"
                            ),
                            0,
                            " Hz"
                        )
                        if acoustic_fingerprint.get(
                            "best_bandwidth"
                        )
                        is not None
                        else "N/A"
                    )
                ),
                (
                    "Best relative level",
                    _fmt_num(
                        acoustic_fingerprint.get(
                            "best_level"
                        ),
                        2
                    )
                ),
                (
                    "Average comfort",
                    _fmt_num(
                        acoustic_fingerprint.get(
                            "average_comfort"
                        ),
                        1,
                        " / 10"
                    )
                ),
                (
                    "Average tinnitus noticeability",
                    _fmt_num(
                        acoustic_fingerprint.get(
                            "average_noticeability"
                        ),
                        1,
                        " / 10"
                    )
                ),
                (
                    "Best observed response score",
                    _fmt_num(
                        acoustic_fingerprint.get(
                            "best_response_score"
                        ),
                        1,
                        " / 100"
                    )
                ),
                (
                    "Latest clinician decision",
                    acoustic_fingerprint.get(
                        "latest_clinician_decision"
                    )
                ),
                (
                    "Latest final stimulus",
                    acoustic_fingerprint.get(
                        "latest_clinician_stimulus"
                    )
                ),
                (
                    "Latest final prediction",
                    _fmt_num(
                        acoustic_fingerprint.get(
                            "latest_final_prediction"
                        ),
                        1,
                        " / 100"
                    )
                )
            ],
            styles
        )
    )

    # ========================================================
    # 3. AUDIOGRAM
    # ========================================================

    story.append(
        _section_title(
            "3. Audiological Assessment",
            styles
        )
    )

    if audiogram_data:
        temp_dir = tempfile.mkdtemp(
            prefix="smarttinnitus_report_"
        )

        audiogram_path = os.path.join(
            temp_dir,
            "audiogram.png"
        )

        generated = _create_audiogram_image(
            audiogram_data,
            audiogram_path
        )

        if generated:
            story.append(
                Image(
                    generated,
                    width=170 * mm,
                    height=87 * mm
                )
            )

        audiogram_rows = []

        frequencies = audiogram_data.get(
            "frequencies",
            []
        )

        right = audiogram_data.get(
            "right_ear",
            []
        )

        left = audiogram_data.get(
            "left_ear",
            []
        )

        for index, frequency in enumerate(
            frequencies
        ):
            audiogram_rows.append(
                [
                    f"{frequency} Hz",
                    (
                        f"{right[index]} dB HL"
                        if index < len(
                            right
                        )
                        else "—"
                    ),
                    (
                        f"{left[index]} dB HL"
                        if index < len(
                            left
                        )
                        else "—"
                    )
                ]
            )

        story.append(
            _data_table(
                [
                    "Frequency",
                    "Right Ear",
                    "Left Ear"
                ],
                audiogram_rows,
                styles,
                col_widths=[
                    55 * mm,
                    55 * mm,
                    55 * mm
                ],
                font_size=8
            )
        )

    else:
        story.append(
            Paragraph(
                "No complete audiogram was available for this report.",
                styles["SmallBody"]
            )
        )

    # ========================================================
    # 4. AI + AUDIOLOGIST DECISION CHAIN
    # ========================================================

    story.append(
        PageBreak()
    )

    story.append(
        _section_title(
            "4. AI Recommendations and Audiologist Decisions",
            styles
        )
    )

    if audiologist_decisions:
        trial_lookup = _trial_by_decision(
            stimulus_trials
        )

        decision_rows = []

        for decision in sorted(
            audiologist_decisions,
            key=lambda d: (
                d.get(
                    "decision_id"
                )
                or
                0
            ),
            reverse=True
        ):
            linked_trial = trial_lookup.get(
                decision.get(
                    "decision_id"
                )
            )

            observed_score = (
                _response_score(
                    linked_trial
                )
                if linked_trial
                else None
            )

            final_prediction = (
                decision.get(
                    "final_predicted_score"
                )
            )

            difference = None

            if (
                observed_score is not None
                and
                final_prediction is not None
            ):
                try:
                    difference = (
                        float(
                            observed_score
                        )
                        -
                        float(
                            final_prediction
                        )
                    )
                except Exception:
                    difference = None

            ai_frequency = (
                "Broadband"
                if decision.get(
                    "ai_stimulus_type"
                )
                ==
                "White Noise"
                else _fmt_num(
                    decision.get(
                        "ai_frequency_hz"
                    ),
                    0,
                    " Hz"
                )
            )

            final_frequency = (
                "Broadband"
                if decision.get(
                    "final_stimulus_type"
                )
                ==
                "White Noise"
                else _fmt_num(
                    decision.get(
                        "final_frequency_hz"
                    ),
                    0,
                    " Hz"
                )
            )

            decision_rows.append(
                [
                    decision.get(
                        "decision_id"
                    ),
                    decision.get(
                        "session_date"
                    ),
                    decision.get(
                        "decision"
                    ),
                    (
                        f"{_safe(decision.get('ai_stimulus_type'))}<br/>"
                        f"{ai_frequency}"
                    ),
                    _fmt_num(
                        decision.get(
                            "ai_predicted_score"
                        ),
                        1,
                        " / 100"
                    ),
                    (
                        f"{_safe(decision.get('final_stimulus_type'))}<br/>"
                        f"{final_frequency}"
                    ),
                    _fmt_num(
                        final_prediction,
                        1,
                        " / 100"
                    ),
                    _fmt_num(
                        observed_score,
                        1,
                        " / 100"
                    ),
                    (
                        _fmt_num(
                            difference,
                            1
                        )
                        if difference
                        is not None
                        else "—"
                    )
                ]
            )

        story.append(
            _data_table(
                [
                    "ID",
                    "Session",
                    "Decision",
                    "Original AI Proposal",
                    "AI Prediction",
                    "Final Configuration",
                    "Final Prediction",
                    "Observed",
                    "Observed - Pred."
                ],
                decision_rows,
                styles,
                col_widths=[
                    9 * mm,
                    18 * mm,
                    17 * mm,
                    28 * mm,
                    19 * mm,
                    28 * mm,
                    19 * mm,
                    18 * mm,
                    18 * mm
                ],
                font_size=5.7
            )
        )

        story.append(
            Spacer(
                1,
                8
            )
        )

        latest = sorted(
            audiologist_decisions,
            key=lambda d: (
                d.get(
                    "decision_id"
                )
                or
                0
            ),
            reverse=True
        )[0]

        story.append(
            Paragraph(
                "<b>Latest traceability</b>",
                styles["SmallBody"]
            )
        )

        story.append(
            Spacer(
                1,
                3
            )
        )

        story.append(
            _kv_table(
                [
                    (
                        "Profile ID",
                        latest.get(
                            "profile_id"
                        )
                    ),
                    (
                        "Model version",
                        latest.get(
                            "model_version"
                        )
                    ),
                    (
                        "Model trained at",
                        latest.get(
                            "model_trained_at"
                        )
                    ),
                    (
                        "Recommendation created",
                        latest.get(
                            "recommendation_created_at"
                        )
                    ),
                    (
                        "Audiologist notes",
                        latest.get(
                            "clinician_notes"
                        )
                    )
                ],
                styles
            )
        )

    else:
        story.append(
            Paragraph(
                "No audiologist decision records were available.",
                styles["SmallBody"]
            )
        )

    # ========================================================
    # 5. STIMULUS RESPONSE HISTORY
    # ========================================================

    story.append(
        _section_title(
            "5. Acoustic Response History",
            styles
        )
    )

    if stimulus_trials:
        trial_rows = []

        for trial in stimulus_trials:
            stimulus = trial.get(
                "stimulus_type"
            )

            frequency = (
                "Broadband"
                if stimulus
                ==
                "White Noise"
                else _fmt_num(
                    trial.get(
                        "frequency"
                    ),
                    0,
                    " Hz"
                )
            )

            score = _response_score(
                trial
            )

            trial_rows.append(
                [
                    trial.get(
                        "session_id"
                    ),
                    trial.get(
                        "source_type"
                    ),
                    stimulus,
                    frequency,
                    (
                        _fmt_num(
                            trial.get(
                                "bandwidth"
                            ),
                            0,
                            " Hz"
                        )
                        if stimulus
                        ==
                        "Narrow-Band Noise"
                        else "N/A"
                    ),
                    _fmt_num(
                        trial.get(
                            "level"
                        ),
                        2
                    ),
                    _fmt_num(
                        trial.get(
                            "comfort"
                        ),
                        0,
                        " / 10"
                    ),
                    _fmt_num(
                        trial.get(
                            "noticeability"
                        ),
                        0,
                        " / 10"
                    ),
                    _fmt_num(
                        score,
                        1,
                        " / 100"
                    ),
                    trial.get(
                        "tested_at"
                    )
                    or
                    trial.get(
                        "created_at"
                    )
                ]
            )

        story.append(
            _data_table(
                [
                    "Session",
                    "Source",
                    "Stimulus",
                    "Frequency",
                    "Bandwidth",
                    "Level",
                    "Comfort",
                    "Noticeability",
                    "Response",
                    "Tested"
                ],
                trial_rows,
                styles,
                col_widths=[
                    11 * mm,
                    19 * mm,
                    24 * mm,
                    18 * mm,
                    18 * mm,
                    13 * mm,
                    15 * mm,
                    18 * mm,
                    17 * mm,
                    25 * mm
                ],
                font_size=5.5
            )
        )

    else:
        story.append(
            Paragraph(
                "No stimulus-response trials were available.",
                styles["SmallBody"]
            )
        )

    # ========================================================
    # 6. LONGITUDINAL FOLLOW-UP
    # ========================================================

    story.append(
        _section_title(
            "6. Longitudinal Follow-up",
            styles
        )
    )

    if sessions:
        session_rows = []

        for session in sessions:
            session_rows.append(
                [
                    session.get(
                        "session_id"
                    ),
                    session.get(
                        "date"
                    ),
                    session.get(
                        "annoyance_before"
                    ),
                    session.get(
                        "annoyance_after"
                    ),
                    session.get(
                        "noticeability_before"
                    ),
                    session.get(
                        "noticeability_after"
                    ),
                    session.get(
                        "notes"
                    )
                ]
            )

        story.append(
            _data_table(
                [
                    "Session",
                    "Date",
                    "Annoyance Before",
                    "Annoyance After",
                    "Noticeability Before",
                    "Noticeability After",
                    "Notes"
                ],
                session_rows,
                styles,
                col_widths=[
                    12 * mm,
                    23 * mm,
                    23 * mm,
                    23 * mm,
                    25 * mm,
                    25 * mm,
                    40 * mm
                ],
                font_size=6
            )
        )

        temp_dir_2 = tempfile.mkdtemp(
            prefix="smarttinnitus_followup_"
        )

        followup_path = os.path.join(
            temp_dir_2,
            "followup.png"
        )

        generated_followup = (
            _create_longitudinal_image(
                sessions,
                followup_path
            )
        )

        if generated_followup:
            story.append(
                Spacer(
                    1,
                    8
                )
            )

            story.append(
                Image(
                    generated_followup,
                    width=170 * mm,
                    height=87 * mm
                )
            )

    else:
        story.append(
            Paragraph(
                "No longitudinal follow-up sessions were available.",
                styles["SmallBody"]
            )
        )

    # ========================================================
    # 7. INTERPRETATION / LIMITATIONS
    # ========================================================

    story.append(
        _section_title(
            "7. Interpretation and Prototype Limitations",
            styles
        )
    )

    story.append(
        Paragraph(
            "<b>Response score:</b> SmartTinnitus uses a 0-100 engineering score combining 40% comfort, 50% low tinnitus noticeability, and 10% willingness to listen again. It is not a validated clinical outcome scale.",
            styles["SmallBody"]
        )
    )

    story.append(
        Spacer(
            1,
            5
        )
    )

    story.append(
        Paragraph(
            "<b>AI predictions:</b> Predicted scores rank candidate acoustic configurations according to the current prototype model. They do not demonstrate clinical effectiveness, treatment benefit, or diagnosis.",
            styles["SmallBody"]
        )
    )

    story.append(
        Spacer(
            1,
            5
        )
    )

    story.append(
        Paragraph(
            "<b>Acoustic levels:</b> Relative stimulus level refers to digital signal amplitude. It is not calibrated sound pressure level (dB SPL) or audiometric hearing level (dB HL). Audiogram thresholds, when entered by the clinician, are expressed in dB HL.",
            styles["SmallBody"]
        )
    )

    story.append(
        Spacer(
            1,
            10
        )
    )

    story.append(
        Paragraph(
            "SmartTinnitus is a research and engineering prototype. This report does not provide a medical diagnosis, does not replace an audiologist or physician, and does not constitute a validated tinnitus treatment recommendation.",
            styles["Disclaimer"]
        )
    )

    def _footer(
        canvas,
        document
    ):
        canvas.saveState()

        canvas.setFont(
            "Helvetica",
            7
        )

        canvas.setFillColor(
            MID_GREY
        )

        canvas.drawString(
            16 * mm,
            8 * mm,
            f"SmartTinnitus - Patient {patient_code}"
        )

        canvas.drawRightString(
            PAGE_WIDTH
            -
            16 * mm,
            8 * mm,
            f"Page {document.page}"
        )

        canvas.restoreState()

    doc.build(
        story,
        onFirstPage=_footer,
        onLaterPages=_footer
    )

    return file_path
