# ============================================================
# SMARTTINNITUS
# ADAPTIVE PERSONALIZATION
# ============================================================
#
# Reusable candidate-generation logic shared by:
#
# - Patient-side AI Personalization
# - Audiologist Dashboard
#
# The goal is to avoid always generating exactly the same
# fixed candidates around the tinnitus pitch.
#
# STRATEGY
# --------
#
# 1. If there is not enough patient-specific response history:
#       -> use a broad search around the tinnitus pitch
#
# 2. If enough patient-specific response history exists:
#       -> identify the strongest recorded response region
#       -> generate a tighter frequency search around that region
#       -> adapt relative levels from the better-performing trials
#       -> adapt narrow-band-noise bandwidth when enough NBN data exist
#
# IMPORTANT
# ---------
#
# This is a prototype engineering personalization strategy.
# It is NOT a validated clinical treatment-selection algorithm.
#
# Relative level is digital amplitude and is NOT dB SPL or dB HL.
# ============================================================


import math

import numpy as np
import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

MIN_FREQUENCY_HZ = 250
MAX_FREQUENCY_HZ = 10000

MIN_RELATIVE_LEVEL = 0.01
MAX_RELATIVE_LEVEL = 0.05

DEFAULT_LEVELS = [
    0.02,
    0.03,
    0.04,
    0.05
]

DEFAULT_BANDWIDTHS = [
    250,
    500,
    750,
    1000
]

STIMULUS_TYPES = [
    "Pure Tone",
    "White Noise",
    "Narrow-Band Noise"
]

MIN_TRIALS_FOR_ADAPTATION = 3


# ============================================================
# HELPERS
# ============================================================


def _is_valid_number(value):

    try:

        value = float(value)

        return math.isfinite(value)

    except Exception:

        return False


def _clip_frequency(
    frequency
):

    frequency = float(
        frequency
    )

    return max(
        MIN_FREQUENCY_HZ,
        min(
            MAX_FREQUENCY_HZ,
            frequency
        )
    )


def _clip_level(
    level
):

    level = float(
        level
    )

    return max(
        MIN_RELATIVE_LEVEL,
        min(
            MAX_RELATIVE_LEVEL,
            level
        )
    )


def _round_frequency(
    frequency,
    step=50
):

    frequency = float(
        frequency
    )

    return round(
        frequency / step
    ) * step


def _round_level(
    level
):

    return round(
        float(level),
        2
    )


def _unique_sorted(
    values
):

    cleaned = []

    for value in values:

        if not _is_valid_number(
            value
        ):

            continue

        value = float(
            value
        )

        if value not in cleaned:

            cleaned.append(
                value
            )

    return sorted(
        cleaned
    )


# ============================================================
# BROAD SEARCH
# ============================================================


def get_broad_frequency_candidates(
    tinnitus_pitch
):

    pitch = float(
        tinnitus_pitch
    )

    frequencies = [
        pitch - 1000,
        pitch - 500,
        pitch,
        pitch + 500,
        pitch + 1000
    ]

    frequencies = [
        _round_frequency(
            _clip_frequency(
                frequency
            )
        )
        for frequency
        in frequencies
    ]

    return _unique_sorted(
        frequencies
    )


# ============================================================
# PREPARE TRIAL HISTORY
# ============================================================


def prepare_trial_history(
    trials_df
):

    required_columns = [
        "stimulus_type",
        "center_frequency_hz",
        "bandwidth_hz",
        "relative_level",
        "response_score"
    ]

    if (
        trials_df is None
        or
        len(trials_df) == 0
    ):

        # Return an empty dataframe with the expected schema.
        # This prevents KeyError when later code accesses
        # columns such as history["stimulus_type"].
        return pd.DataFrame(
            columns=required_columns
        )

    df = (
        trials_df
        .copy()
    )

    for column in required_columns:

        if column not in df.columns:

            df[
                column
            ] = np.nan

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in [
        "center_frequency_hz",
        "bandwidth_hz",
        "relative_level",
        "response_score"
    ]:

        df[
            column
        ] = pd.to_numeric(
            df[
                column
            ],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove rows with no usable response score
    # --------------------------------------------------------

    df = (
        df[
            df[
                "response_score"
            ]
            .notna()
        ]
        .copy()
    )

    return df


# ============================================================
# FIND TOP-PERFORMING TRIALS
# ============================================================


def get_top_trials(
    trials_df,
    maximum_trials=5
):

    df = prepare_trial_history(
        trials_df
    )

    if df.empty:

        return df

    # Keep the strongest-performing subset.
    #
    # With a small history we keep up to the best 3.
    # With a larger history we keep roughly the best 40%,
    # capped at maximum_trials.

    number_of_trials = len(
        df
    )

    top_count = max(
        3,
        int(
            math.ceil(
                number_of_trials * 0.40
            )
        )
    )

    top_count = min(
        maximum_trials,
        top_count,
        number_of_trials
    )

    return (
        df
        .sort_values(
            "response_score",
            ascending=False
        )
        .head(
            top_count
        )
        .copy()
    )


# ============================================================
# ADAPTIVE FREQUENCY REGION
# ============================================================


def get_adaptive_frequency_candidates(
    tinnitus_pitch,
    trials_df
):

    history = prepare_trial_history(
        trials_df
    )

    # --------------------------------------------------------
    # Frequency-specific trials only
    #
    # White noise is broadband, so its stored center frequency
    # should not define a "preferred frequency region".
    # --------------------------------------------------------

    frequency_history = (
        history[
            history[
                "stimulus_type"
            ]
            .isin(
                [
                    "Pure Tone",
                    "Narrow-Band Noise"
                ]
            )
            &
            history[
                "center_frequency_hz"
            ]
            .notna()
        ]
        .copy()
    )

    if len(
        frequency_history
    ) < MIN_TRIALS_FOR_ADAPTATION:

        return (
            get_broad_frequency_candidates(
                tinnitus_pitch
            ),
            {
                "mode":
                    "Broad search",

                "reason":
                    (
                        "Not enough frequency-specific "
                        "patient response history."
                    ),

                "center_frequency_hz":
                    float(
                        tinnitus_pitch
                    ),

                "frequency_min_hz":
                    min(
                        get_broad_frequency_candidates(
                            tinnitus_pitch
                        )
                    ),

                "frequency_max_hz":
                    max(
                        get_broad_frequency_candidates(
                            tinnitus_pitch
                        )
                    )
            }
        )

    # --------------------------------------------------------
    # Use the best frequency-specific trials.
    # --------------------------------------------------------

    top_frequency_trials = (
        frequency_history
        .sort_values(
            "response_score",
            ascending=False
        )
        .head(
            min(
                5,
                max(
                    3,
                    int(
                        math.ceil(
                            len(
                                frequency_history
                            )
                            *
                            0.40
                        )
                    )
                )
            )
        )
        .copy()
    )

    frequencies = (
        top_frequency_trials[
            "center_frequency_hz"
        ]
        .astype(float)
        .to_numpy()
    )

    scores = (
        top_frequency_trials[
            "response_score"
        ]
        .astype(float)
        .to_numpy()
    )

    # Shift scores to positive weights if necessary.
    minimum_score = np.min(
        scores
    )

    weights = (
        scores
        -
        minimum_score
        +
        1.0
    )

    weighted_center = float(
        np.average(
            frequencies,
            weights=weights
        )
    )

    weighted_center = (
        _round_frequency(
            _clip_frequency(
                weighted_center
            ),
            step=50
        )
    )

    # --------------------------------------------------------
    # Narrower adaptive search around the promising region.
    # --------------------------------------------------------

    adaptive_frequencies = [
        weighted_center - 500,
        weighted_center - 250,
        weighted_center,
        weighted_center + 250,
        weighted_center + 500
    ]

    adaptive_frequencies = [
        _round_frequency(
            _clip_frequency(
                frequency
            ),
            step=50
        )
        for frequency
        in adaptive_frequencies
    ]

    adaptive_frequencies = _unique_sorted(
        adaptive_frequencies
    )

    return (
        adaptive_frequencies,
        {
            "mode":
                "Adaptive search",

            "reason":
                (
                    "Frequency region derived from the "
                    "patient's better-performing trials."
                ),

            "center_frequency_hz":
                weighted_center,

            "frequency_min_hz":
                min(
                    adaptive_frequencies
                ),

            "frequency_max_hz":
                max(
                    adaptive_frequencies
                ),

            "frequency_trials_used":
                int(
                    len(
                        top_frequency_trials
                    )
                )
        }
    )


# ============================================================
# ADAPTIVE RELATIVE LEVELS
# ============================================================


def get_adaptive_levels(
    trials_df
):

    history = prepare_trial_history(
        trials_df
    )

    if len(
        history
    ) < MIN_TRIALS_FOR_ADAPTATION:

        return (
            DEFAULT_LEVELS.copy(),
            {
                "level_mode":
                    "Broad levels",

                "preferred_level":
                    None
            }
        )

    top_trials = get_top_trials(
        history
    )

    valid_levels = (
        top_trials[
            "relative_level"
        ]
        .dropna()
        .astype(float)
    )

    if valid_levels.empty:

        return (
            DEFAULT_LEVELS.copy(),
            {
                "level_mode":
                    "Broad levels",

                "preferred_level":
                    None
            }
        )

    preferred_level = float(
        valid_levels.median()
    )

    preferred_level = (
        _round_level(
            _clip_level(
                preferred_level
            )
        )
    )

    candidate_levels = [
        preferred_level - 0.01,
        preferred_level,
        preferred_level + 0.01
    ]

    candidate_levels = [
        _round_level(
            _clip_level(
                level
            )
        )
        for level
        in candidate_levels
    ]

    candidate_levels = _unique_sorted(
        candidate_levels
    )

    return (
        candidate_levels,
        {
            "level_mode":
                "Adaptive levels",

            "preferred_level":
                preferred_level
        }
    )


# ============================================================
# ADAPTIVE NARROW-BAND-NOISE BANDWIDTH
# ============================================================


def get_adaptive_bandwidths(
    trials_df
):

    history = prepare_trial_history(
        trials_df
    )

    nbn_history = (
        history[
            (
                history[
                    "stimulus_type"
                ]
                ==
                "Narrow-Band Noise"
            )
            &
            history[
                "bandwidth_hz"
            ]
            .notna()
        ]
        .copy()
    )

    if len(
        nbn_history
    ) < MIN_TRIALS_FOR_ADAPTATION:

        return (
            DEFAULT_BANDWIDTHS.copy(),
            {
                "bandwidth_mode":
                    "Broad bandwidths",

                "preferred_bandwidth_hz":
                    None
            }
        )

    top_nbn = (
        nbn_history
        .sort_values(
            "response_score",
            ascending=False
        )
        .head(
            min(
                3,
                len(
                    nbn_history
                )
            )
        )
    )

    preferred_bandwidth = float(
        top_nbn[
            "bandwidth_hz"
        ]
        .median()
    )

    preferred_bandwidth = int(
        round(
            preferred_bandwidth / 50
        )
        *
        50
    )

    candidate_bandwidths = [
        max(
            100,
            preferred_bandwidth - 250
        ),
        max(
            100,
            preferred_bandwidth
        ),
        min(
            2000,
            preferred_bandwidth + 250
        )
    ]

    candidate_bandwidths = [
        int(
            value
        )
        for value
        in _unique_sorted(
            candidate_bandwidths
        )
    ]

    return (
        candidate_bandwidths,
        {
            "bandwidth_mode":
                "Adaptive bandwidths",

            "preferred_bandwidth_hz":
                preferred_bandwidth
        }
    )


# ============================================================
# STIMULUS-TYPE RESPONSE SUMMARY
# ============================================================


def get_stimulus_type_summary(
    trials_df
):

    history = prepare_trial_history(
        trials_df
    )

    if history.empty:

        return pd.DataFrame(
            columns=[
                "stimulus_type",
                "trial_count",
                "mean_response_score",
                "best_response_score"
            ]
        )

    summary = (
        history
        .groupby(
            "stimulus_type",
            dropna=False
        )
        .agg(
            trial_count=(
                "response_score",
                "count"
            ),

            mean_response_score=(
                "response_score",
                "mean"
            ),

            best_response_score=(
                "response_score",
                "max"
            )
        )
        .reset_index()
    )

    summary[
        "mean_response_score"
    ] = (
        summary[
            "mean_response_score"
        ]
        .round(
            1
        )
    )

    summary[
        "best_response_score"
    ] = (
        summary[
            "best_response_score"
        ]
        .round(
            1
        )
    )

    return (
        summary
        .sort_values(
            "mean_response_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# MAIN ADAPTIVE CANDIDATE GENERATOR
# ============================================================


def generate_adaptive_candidate_stimuli(
    tinnitus_pitch,
    trials_df=None
):

    (
        frequencies,
        frequency_metadata
    ) = get_adaptive_frequency_candidates(
        tinnitus_pitch,
        trials_df
    )

    (
        levels,
        level_metadata
    ) = get_adaptive_levels(
        trials_df
    )

    (
        bandwidths,
        bandwidth_metadata
    ) = get_adaptive_bandwidths(
        trials_df
    )

    candidates = []

    for stimulus_type in STIMULUS_TYPES:

        # White noise is broadband.
        #
        # The ML pipeline still expects a center_frequency_hz
        # value, so we use tinnitus pitch internally. It should
        # not be interpreted as the white-noise center frequency.

        if (
            stimulus_type
            ==
            "White Noise"
        ):

            stimulus_frequencies = [
                float(
                    tinnitus_pitch
                )
            ]

        else:

            stimulus_frequencies = (
                frequencies
            )

        for frequency in stimulus_frequencies:

            for level in levels:

                if (
                    stimulus_type
                    ==
                    "Narrow-Band Noise"
                ):

                    stimulus_bandwidths = (
                        bandwidths
                    )

                else:

                    stimulus_bandwidths = [
                        0
                    ]

                for bandwidth in stimulus_bandwidths:

                    candidates.append(
                        {
                            "stimulus_type":
                                stimulus_type,

                            "center_frequency_hz":
                                float(
                                    frequency
                                ),

                            "bandwidth_hz":
                                float(
                                    bandwidth
                                ),

                            "relative_level":
                                float(
                                    level
                                )
                        }
                    )

    stimulus_summary = get_stimulus_type_summary(
        trials_df
    )

    preferred_stimulus_type = None

    if not stimulus_summary.empty:

        preferred_stimulus_type = (
            stimulus_summary.iloc[
                0
            ][
                "stimulus_type"
            ]
        )

    metadata = {
        **frequency_metadata,
        **level_metadata,
        **bandwidth_metadata,

        "candidate_count":
            int(
                len(
                    candidates
                )
            ),

        "patient_trial_count":
            int(
                0
                if trials_df is None
                else len(
                    trials_df
                )
            ),

        "preferred_stimulus_type":
            preferred_stimulus_type,

        "stimulus_type_summary":
            stimulus_summary
    }

    return (
        candidates,
        metadata
    )


# ============================================================
# HUMAN-READABLE SUMMARY
# ============================================================


def get_adaptive_personalization_summary(
    tinnitus_pitch,
    trials_df=None
):

    (
        candidates,
        metadata
    ) = generate_adaptive_candidate_stimuli(
        tinnitus_pitch,
        trials_df
    )

    return {
        "mode":
            metadata[
                "mode"
            ],

        "reason":
            metadata[
                "reason"
            ],

        "candidate_count":
            metadata[
                "candidate_count"
            ],

        "patient_trial_count":
            metadata[
                "patient_trial_count"
            ],

        "search_center_hz":
            metadata[
                "center_frequency_hz"
            ],

        "frequency_min_hz":
            metadata[
                "frequency_min_hz"
            ],

        "frequency_max_hz":
            metadata[
                "frequency_max_hz"
            ],

        "preferred_level":
            metadata[
                "preferred_level"
            ],

        "preferred_bandwidth_hz":
            metadata[
                "preferred_bandwidth_hz"
            ],

        "preferred_stimulus_type":
            metadata[
                "preferred_stimulus_type"
            ]
    }
