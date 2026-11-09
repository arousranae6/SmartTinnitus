import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# STANDARD AUDIOGRAM FREQUENCIES
# =====================================================

AUDIOGRAM_FREQUENCIES = [
    250,
    500,
    1000,
    2000,
    4000,
    8000
]


# =====================================================
# CONVERT FREQUENCY TO AUDIOGRAM POSITION
# =====================================================
#
# Instead of using Matplotlib's logarithmic x-axis,
# we manually position octave frequencies.
#
# This avoids the MathText parsing error:
#
# $\\mathdefault{3\\times10^{2}}$
#
# while keeping the frequencies visually spaced
# in the expected audiogram order.
# =====================================================

def frequency_to_position(
    frequency
):

    if frequency is None:
        return None

    try:

        frequency = float(
            frequency
        )

    except Exception:

        return None

    if frequency <= 0:
        return None

    standard_frequencies = np.array(
        AUDIOGRAM_FREQUENCIES,
        dtype=float
    )

    standard_positions = np.arange(
        len(
            AUDIOGRAM_FREQUENCIES
        ),
        dtype=float
    )

    # -------------------------------------------------
    # Use logarithmic interpolation manually.
    #
    # Example:
    #
    # 1000 Hz → position 2
    # 2000 Hz → position 3
    #
    # A tinnitus pitch between those values will
    # appear between those two x-axis positions.
    # -------------------------------------------------

    log_standard = np.log2(
        standard_frequencies
    )

    log_frequency = np.log2(
        frequency
    )

    position = np.interp(
        log_frequency,
        log_standard,
        standard_positions
    )

    return float(
        position
    )


# =====================================================
# PLOT AUDIOGRAM
# =====================================================

def plot_audiogram(
    right_ear,
    left_ear,
    tinnitus_pitch=None
):

    # -------------------------------------------------
    # X positions
    # -------------------------------------------------

    x_positions = np.arange(
        len(
            AUDIOGRAM_FREQUENCIES
        )
    )

    # -------------------------------------------------
    # CREATE FIGURE
    # -------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            8,
            5
        )
    )

    # -------------------------------------------------
    # RIGHT EAR
    # -------------------------------------------------

    ax.plot(
        x_positions,
        right_ear,
        marker="o",
        linewidth=2,
        label="Right Ear"
    )

    # -------------------------------------------------
    # LEFT EAR
    # -------------------------------------------------

    ax.plot(
        x_positions,
        left_ear,
        marker="o",
        linewidth=2,
        label="Left Ear"
    )

    # -------------------------------------------------
    # TINNITUS PITCH MARKER
    # -------------------------------------------------

    if tinnitus_pitch is not None:

        tinnitus_position = (
            frequency_to_position(
                tinnitus_pitch
            )
        )

        if tinnitus_position is not None:

            ax.axvline(
                x=tinnitus_position,
                linestyle="--",
                linewidth=2,
                label=(
                    f"Tinnitus Pitch: "
                    f"{int(float(tinnitus_pitch))} Hz"
                )
            )

    # -------------------------------------------------
    # X AXIS
    # -------------------------------------------------

    ax.set_xticks(
        x_positions
    )

    ax.set_xticklabels(
        [
            "250",
            "500",
            "1k",
            "2k",
            "4k",
            "8k"
        ]
    )

    ax.set_xlim(
        -0.3,
        len(
            AUDIOGRAM_FREQUENCIES
        )
        -
        0.7
    )

    ax.set_xlabel(
        "Frequency (Hz)"
    )

    # -------------------------------------------------
    # Y AXIS
    # -------------------------------------------------
    #
    # Audiograms conventionally show better hearing
    # thresholds at the top and poorer thresholds
    # toward the bottom.
    # -------------------------------------------------

    ax.set_ylim(
        120,
        -10
    )

    ax.set_yticks(
        list(
            range(
                -10,
                121,
                10
            )
        )
    )

    ax.set_ylabel(
        "Hearing Level (dB HL)"
    )

    # -------------------------------------------------
    # TITLE
    # -------------------------------------------------

    ax.set_title(
        "Audiogram and Tinnitus Pitch"
    )

    # -------------------------------------------------
    # GRID
    # -------------------------------------------------

    ax.grid(
        True,
        alpha=0.3
    )

    # -------------------------------------------------
    # LEGEND
    # -------------------------------------------------

    ax.legend()

    # -------------------------------------------------
    # LAYOUT
    # -------------------------------------------------

    fig.tight_layout()

    return fig