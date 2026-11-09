# =====================================================
# SMARTTINNITUS RESPONSE SCORING
# =====================================================
#
# Prototype weighted response score:
#
# 40% Comfort
# 50% Low tinnitus noticeability
# 10% Willingness to listen again
#
# Final score range:
# 0 - 100
#
# IMPORTANT:
# This is an engineering score created for the
# SmartTinnitus prototype.
#
# It is NOT a validated clinical outcome measure.
# =====================================================


def calculate_response_score(
    comfort_score,
    tinnitus_noticeability,
    listen_again
):

    # =================================================
    # VALIDATE / NORMALIZE INPUTS
    # =================================================

    comfort_score = float(
        comfort_score
    )

    tinnitus_noticeability = float(
        tinnitus_noticeability
    )

    # Keep values inside expected 0-10 range
    comfort_score = max(
        0,
        min(
            10,
            comfort_score
        )
    )

    tinnitus_noticeability = max(
        0,
        min(
            10,
            tinnitus_noticeability
        )
    )

    # =================================================
    # 40% COMFORT
    # =================================================
    #
    # Example:
    # Comfort = 8/10
    #
    # 8 / 10 × 40
    # = 32 points
    # =================================================

    comfort_component = (
        comfort_score
        /
        10
    ) * 40

    # =================================================
    # 50% LOW TINNITUS NOTICEABILITY
    # =================================================
    #
    # Lower noticeability is better.
    #
    # Example:
    # Noticeability = 3/10
    #
    # 10 - 3 = 7
    #
    # 7 / 10 × 50
    # = 35 points
    # =================================================

    noticeability_component = (
        (
            10
            -
            tinnitus_noticeability
        )
        /
        10
    ) * 50

    # =================================================
    # 10% LISTEN AGAIN
    # =================================================

    if str(
        listen_again
    ).strip().lower() == "yes":

        listen_again_component = 10

    else:

        listen_again_component = 0

    # =================================================
    # FINAL SCORE
    # =================================================

    final_score = (
        comfort_component
        +
        noticeability_component
        +
        listen_again_component
    )

    return round(
        float(
            final_score
        ),
        2
    )


# =====================================================
# FIND BEST PREVIOUS TRIAL
# =====================================================

def find_best_trial(
    trials
):

    scored_trials = []

    for trial in trials:

        (
            stimulus_type,
            frequency,
            bandwidth,
            level,
            comfort,
            noticeability,
            listen_again
        ) = trial

        score = calculate_response_score(
            comfort,
            noticeability,
            listen_again
        )

        scored_trials.append(
            {
                "stimulus_type":
                    stimulus_type,

                "frequency":
                    frequency,

                "bandwidth":
                    bandwidth,

                "level":
                    level,

                "comfort":
                    comfort,

                "noticeability":
                    noticeability,

                "listen_again":
                    listen_again,

                "score":
                    score
            }
        )

    if len(
        scored_trials
    ) == 0:

        return None

    return max(
        scored_trials,
        key=lambda x:
            x[
                "score"
            ]
    )


# =====================================================
# RULE-BASED NEXT TRIAL
# =====================================================

def recommend_next_trial(
    best_trial
):

    if best_trial is None:

        return None

    recommended_type = (
        best_trial[
            "stimulus_type"
        ]
    )

    recommended_frequency = (
        best_trial[
            "frequency"
        ]
    )

    recommended_level = (
        best_trial[
            "level"
        ]
    )

    if (
        best_trial[
            "bandwidth"
        ]
        is not None
    ):

        recommended_bandwidth = max(
            100,
            best_trial[
                "bandwidth"
            ]
            -
            100
        )

    else:

        recommended_bandwidth = 500

    return {
        "stimulus_type":
            recommended_type,

        "frequency":
            recommended_frequency,

        "bandwidth":
            recommended_bandwidth,

        "level":
            recommended_level
    }