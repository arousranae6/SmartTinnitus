def initialize_pitch_state(session_state):
    if "pitch_center" not in session_state:
        session_state.pitch_center = 6000

    if "pitch_step" not in session_state:
        session_state.pitch_step = 2000

    if "pitch_round" not in session_state:
        session_state.pitch_round = 1

    if "pitch_finished" not in session_state:
        session_state.pitch_finished = False


def get_pitch_options(session_state):
    center = session_state.pitch_center
    step = session_state.pitch_step

    return [
        max(250, center - step),
        center,
        min(10000, center + step)
    ]


def update_pitch_search(session_state, selected_pitch):
    session_state.pitch_center = selected_pitch
    session_state.pitch_step = session_state.pitch_step / 2
    session_state.pitch_round += 1

    if session_state.pitch_step <= 125:
        session_state.pitch_finished = True


def reset_pitch_search(session_state):
    session_state.pitch_center = 6000
    session_state.pitch_step = 2000
    session_state.pitch_round = 1
    session_state.pitch_finished = False


def get_final_pitch(session_state):
    return int(session_state.pitch_center)