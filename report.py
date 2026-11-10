"""PDF Clinical Report generator for the final SmartTinnitus workflow."""

from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


def _value(value, default="Not available"):
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _number(value, digits=1, suffix=""):
    if value is None:
        return "Not available"
    try:
        number = float(value)
        if digits == 0:
            return f"{number:.0f}{suffix}"
        return f"{number:.{digits}f}{suffix}"
    except Exception:
        return _value(value)


def _section(story, title, styles):
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(title, styles["section"]))
    story.append(Spacer(1, 2 * mm))


def _key_value_table(rows):
    table = Table(rows, colWidths=[55 * mm, 115 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F6F7")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#17384A")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D8E5E8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _data_table(rows, widths=None):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174F63")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D8E5E8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    colors.HexColor("#F8FBFC"),
                ]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8E5E8"))
    canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6F8189"))
    canvas.drawString(
        18 * mm,
        10 * mm,
        "SmartTinnitus clinical report",
    )
    canvas.drawRightString(
        192 * mm,
        10 * mm,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def generate_patient_report(
    filename,
    patient,
    profile,
    audiogram,
    assignments,
    feedback,
    digital_twin,
    care_requests,
):
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="report_title",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=25,
            textColor=colors.HexColor("#17384A"),
            alignment=TA_LEFT,
            spaceAfter=4,
        )
    )

    styles.add(
        ParagraphStyle(
            name="subtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#6F8189"),
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#174F63"),
            spaceBefore=4,
            spaceAfter=4,
        )
    )

    styles.add(
        ParagraphStyle(
            name="body_small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#2C4651"),
        )
    )

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title="SmartTinnitus Clinical Report",
        author="SmartTinnitus",
    )

    story = []

    story.append(Paragraph("SmartTinnitus Clinical Report", styles["report_title"]))
    story.append(
        Paragraph(
            f"Patient {_value(patient.get('patient_code'))} · "
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["subtitle"],
        )
    )

    # Patient information
    _section(story, "1. Patient Information", styles)
    story.append(
        _key_value_table(
            [
                ["Patient code", _value(patient.get("patient_code"))],
                ["Age", _value(patient.get("age"))],
                ["Sex", _value(patient.get("sex"))],
            ]
        )
    )

    # Tinnitus profile
    _section(story, "2. Tinnitus Profile", styles)

    if profile:
        story.append(
            _key_value_table(
                [
                    ["Laterality", _value(profile.get("laterality"))],
                    ["Tinnitus type", _value(profile.get("tinnitus_type"))],
                    ["Temporal pattern", _value(profile.get("temporal_pattern"))],
                    ["Duration", _number(profile.get("duration_months"), 0, " months")],
                    ["Annoyance", _number(profile.get("annoyance_score"), 1, "/10")],
                    ["Sleep impact", _number(profile.get("sleep_score"), 1, "/10")],
                    ["Concentration impact", _number(profile.get("concentration_score"), 1, "/10")],
                    ["Stress", _number(profile.get("stress_score"), 1, "/10")],
                    ["Matched tinnitus pitch", _number(profile.get("estimated_pitch_hz"), 0, " Hz")],
                    ["Relative loudness", _number(profile.get("loudness_relative"), 1)],
                ]
            )
        )
    else:
        story.append(Paragraph("No tinnitus profile is stored.", styles["body_small"]))

    # Audiometry
    _section(story, "3. Audiometry", styles)

    if audiogram:
        story.append(
            Paragraph(
                f"Latest saved audiogram: {_value(audiogram.get('created_at'))}",
                styles["body_small"],
            )
        )
        story.append(Spacer(1, 2 * mm))

        rows = [["Frequency", "Right ear", "Left ear"]]
        for freq, right, left in zip(
            audiogram.get("frequencies", []),
            audiogram.get("right_ear", []),
            audiogram.get("left_ear", []),
        ):
            rows.append(
                [
                    f"{freq} Hz",
                    "—" if right is None else f"{right} dB HL",
                    "—" if left is None else f"{left} dB HL",
                ]
            )

        story.append(
            _data_table(
                rows,
                widths=[45 * mm, 55 * mm, 55 * mm],
            )
        )
    else:
        story.append(Paragraph("No saved audiogram is available.", styles["body_small"]))

    # Acoustic Personalization
    _section(story, "4. Acoustic Personalization", styles)

    if assignments:
        rows = [
            [
                "Sent",
                "Pathway",
                "Stimulus",
                "Frequency",
                "Bandwidth",
                "Level",
                "Duration",
                "Status",
            ]
        ]

        for item in assignments[:20]:
            pathway = (
                "AI-assisted"
                if str(item.get("pathway", "")).lower() == "ai"
                else "Manual"
            )

            rows.append(
                [
                    _value(item.get("sent_at"), "—"),
                    pathway,
                    _value(item.get("stimulus_type")),
                    _number(item.get("center_frequency_hz"), 0, " Hz"),
                    _number(item.get("bandwidth_hz"), 0, " Hz"),
                    _number(item.get("relative_level"), 1),
                    _number(item.get("duration_seconds"), 0, " s"),
                    _value(item.get("status")),
                ]
            )

        story.append(
            _data_table(
                rows,
                widths=[
                    24 * mm,
                    20 * mm,
                    25 * mm,
                    23 * mm,
                    22 * mm,
                    18 * mm,
                    18 * mm,
                    25 * mm,
                ],
            )
        )
    else:
        story.append(
            Paragraph(
                "No acoustic test has been assigned in the current workflow.",
                styles["body_small"],
            )
        )

    # Feedback
    _section(story, "5. Patient Acoustic Feedback", styles)

    if feedback:
        rows = [
            [
                "Responded",
                "Stimulus",
                "Comfort",
                "Noticeability",
                "Listen again",
                "Comment",
            ]
        ]

        for item in feedback[:20]:
            rows.append(
                [
                    _value(item.get("responded_at"), "—"),
                    _value(item.get("stimulus_type")),
                    _number(item.get("comfort_score"), 0, "/10"),
                    _number(item.get("tinnitus_noticeability"), 0, "/10"),
                    _value(item.get("listen_again")),
                    _value(item.get("patient_comment"), "—"),
                ]
            )

        story.append(
            _data_table(
                rows,
                widths=[
                    27 * mm,
                    30 * mm,
                    21 * mm,
                    25 * mm,
                    23 * mm,
                    50 * mm,
                ],
            )
        )
    else:
        story.append(
            Paragraph(
                "No returned patient acoustic feedback is currently stored.",
                styles["body_small"],
            )
        )

    # Digital Twin
    _section(story, "6. Patient Acoustic Digital Twin", styles)

    best = digital_twin.get("best_observed_response")

    twin_rows = [
        [
            "Observed acoustic evidence",
            str(digital_twin.get("observed_evidence_count", 0)),
        ],
        [
            "Returned assignment feedback",
            str(digital_twin.get("assignment_feedback_count", 0)),
        ],
    ]

    if best:
        twin_rows.extend(
            [
                [
                    "Best observed stimulus",
                    _value(best.get("stimulus_type")),
                ],
                [
                    "Best observed frequency",
                    _number(best.get("center_frequency_hz"), 0, " Hz"),
                ],
                [
                    "Best observed response score",
                    _number(best.get("response_score"), 1, "/100"),
                ],
            ]
        )

    story.append(_key_value_table(twin_rows))

    # Care coordination
    _section(story, "7. Care Coordination", styles)

    if care_requests:
        rows = [
            [
                "Date",
                "Direction",
                "Request",
                "Mode",
                "Status",
                "Scheduled",
            ]
        ]

        for item in care_requests[:15]:
            direction = (
                "Audiologist → Patient"
                if item.get("direction") == "audiologist_to_patient"
                else "Patient → Audiologist"
            )

            rows.append(
                [
                    _value(item.get("created_at"), "—"),
                    direction,
                    _value(item.get("title") or item.get("request_type")),
                    _value(item.get("communication_mode"), "—"),
                    _value(item.get("status")),
                    _value(item.get("scheduled_datetime"), "—"),
                ]
            )

        story.append(
            _data_table(
                rows,
                widths=[
                    27 * mm,
                    34 * mm,
                    45 * mm,
                    25 * mm,
                    23 * mm,
                    28 * mm,
                ],
            )
        )
    else:
        story.append(
            Paragraph(
                "No care-coordination records are currently stored.",
                styles["body_small"],
            )
        )

    # Clinical summary
    _section(story, "8. Clinical Summary", styles)

    latest_feedback = digital_twin.get("latest_feedback")
    summary_rows = [
        [
            "Latest tinnitus pitch",
            _number(
                profile.get("estimated_pitch_hz") if profile else None,
                0,
                " Hz",
            ),
        ],
        [
            "Audiogram",
            "Available" if audiogram else "Not available",
        ],
        [
            "Acoustic tests assigned",
            str(len(assignments)),
        ],
        [
            "Patient feedback records",
            str(len(feedback)),
        ],
    ]

    if best:
        summary_rows.append(
            [
                "Best observed acoustic response",
                (
                    f"{_value(best.get('stimulus_type'))} · "
                    f"{_number(best.get('center_frequency_hz'), 0, ' Hz')} · "
                    f"{_number(best.get('response_score'), 1, '/100')}"
                ),
            ]
        )

    if latest_feedback:
        summary_rows.append(
            [
                "Latest returned feedback",
                (
                    f"Comfort {_number(latest_feedback.get('comfort_score'), 0, '/10')}, "
                    f"noticeability "
                    f"{_number(latest_feedback.get('tinnitus_noticeability'), 0, '/10')}, "
                    f"listen again {_value(latest_feedback.get('listen_again'))}"
                ),
            ]
        )

    story.append(_key_value_table(summary_rows))

    doc.build(
        story,
        onFirstPage=_footer,
        onLaterPages=_footer,
    )
