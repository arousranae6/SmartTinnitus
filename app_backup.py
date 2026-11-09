import streamlit as st
import sqlite3
from datetime import datetime
import numpy as np
from scipy.io.wavfile import write
import io
# -----------------------------
# DATABASE CONNECTION
# -----------------------------

conn = sqlite3.connect("smarttinnitus.db", check_same_thread=False)
cursor = conn.cursor()

# -----------------------------
# CREATE PATIENT TABLE
# -----------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_code TEXT UNIQUE,
    age INTEGER,
    sex TEXT,
    created_at TEXT
)
""")

# -----------------------------
# CREATE TINNITUS PROFILE TABLE
# -----------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS tinnitus_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    laterality TEXT,
    tinnitus_type TEXT,
    temporal_pattern TEXT,
    duration_months INTEGER,
    annoyance_score INTEGER,
    sleep_score INTEGER,
    concentration_score INTEGER,
    stress_score INTEGER,
    estimated_pitch_hz REAL,
    loudness_relative REAL,
    created_at TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS stimulus_trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    stimulus_type TEXT,
    center_frequency_hz REAL,
    bandwidth_hz REAL,
    relative_level REAL,
    comfort_score INTEGER,
    tinnitus_noticeability INTEGER,
    listen_again TEXT,
    created_at TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
)
""")
conn.commit()


# -----------------------------
# PAGE TITLE
# -----------------------------

st.title("SmartTinnitus")

st.write(
    "Intelligent digital platform for personalized tinnitus profiling "
    "and adaptive acoustic stimulus optimization."
)

# =====================================================
# SECTION 1 — PATIENT CREATION
# =====================================================

st.header("1. Patient Profile")

patient_code = st.text_input("Patient ID")

age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    step=1
)

sex = st.selectbox(
    "Sex",
    ["Select", "Female", "Male", "Other"]
)

if st.button("Save Patient"):

    if patient_code == "":
        st.error("Please enter a Patient ID.")

    elif sex == "Select":
        st.error("Please select the patient's sex.")

    else:

        try:

            cursor.execute("""
            INSERT INTO patients
            (patient_code, age, sex, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                patient_code,
                age,
                sex,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            )

            conn.commit()

            st.success(
                f"Patient {patient_code} saved successfully."
            )

        except sqlite3.IntegrityError:

            st.error(
                "This Patient ID already exists."
            )

# =====================================================
# SECTION 2 — TINNITUS PROFILE
# =====================================================

st.header("2. Tinnitus Profile")

# Get patient IDs from database
cursor.execute("""
SELECT id, patient_code
FROM patients
ORDER BY patient_code
""")

patients = cursor.fetchall()

if len(patients) == 0:

    st.warning(
        "Please create a patient before completing the tinnitus profile."
    )

else:

    patient_options = {
        patient_code: patient_id
        for patient_id, patient_code in patients
    }

    selected_patient = st.selectbox(
        "Select Patient",
        list(patient_options.keys())
    )

    selected_patient_id = patient_options[selected_patient]

    # -----------------------------
    # TINNITUS CHARACTERISTICS
    # -----------------------------

    st.subheader("Tinnitus Characteristics")

    laterality = st.selectbox(
        "Where do you perceive your tinnitus?",
        [
            "Select",
            "Left ear",
            "Right ear",
            "Both ears",
            "Inside the head"
        ]
    )

    tinnitus_type = st.selectbox(
        "How would you describe the sound?",
        [
            "Select",
            "Tonal",
            "Buzzing",
            "Hissing",
            "Clicking",
            "Mixed",
            "Other"
        ]
    )

    temporal_pattern = st.selectbox(
        "Is your tinnitus:",
        [
            "Select",
            "Continuous",
            "Intermittent"
        ]
    )

    duration_months = st.number_input(
        "How long have you experienced tinnitus? (months)",
        min_value=0,
        max_value=1000,
        step=1
    )

    # -----------------------------
    # IMPACT SCORES
    # -----------------------------

    st.subheader("Impact")

    annoyance_score = st.slider(
        "Current tinnitus annoyance",
        min_value=0,
        max_value=10,
        value=0
    )

    sleep_score = st.slider(
        "Sleep disturbance",
        min_value=0,
        max_value=10,
        value=0
    )

    concentration_score = st.slider(
        "Concentration difficulty",
        min_value=0,
        max_value=10,
        value=0
    )

    stress_score = st.slider(
        "Stress associated with tinnitus",
        min_value=0,
        max_value=10,
        value=0
    )

    # -----------------------------
    # SAVE TINNITUS PROFILE
    # -----------------------------

    if st.button("Save Tinnitus Profile"):

        if laterality == "Select":

            st.error(
                "Please select tinnitus laterality."
            )

        elif tinnitus_type == "Select":

            st.error(
                "Please select tinnitus type."
            )

        elif temporal_pattern == "Select":

            st.error(
                "Please select the temporal pattern."
            )

        else:

            cursor.execute("""
            INSERT INTO tinnitus_profiles
            (
                patient_id,
                laterality,
                tinnitus_type,
                temporal_pattern,
                duration_months,
                annoyance_score,
                sleep_score,
                concentration_score,
                stress_score,
                estimated_pitch_hz,
                loudness_relative,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                selected_patient_id,
                laterality,
                tinnitus_type,
                temporal_pattern,
                duration_months,
                annoyance_score,
                sleep_score,
                concentration_score,
                stress_score,
                None,
                None,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            )

            conn.commit()

            st.success(
                f"Tinnitus profile for {selected_patient} saved successfully."
            )
            # =====================================================
# SECTION 3 — PURE TONE GENERATOR
# =====================================================

st.header("3. Pure Tone Generator")

frequency = st.slider(
    "Select frequency (Hz)",
    min_value=250,
    max_value=10000,
    value=4000,
    step=50
)

duration = st.slider(
    "Duration (seconds)",
    min_value=1,
    max_value=5,
    value=2
)

volume = st.slider(
    "Relative volume",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01
)

if st.button("Generate Tone"):

    sample_rate = 44100

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False
    )

    signal = volume * np.sin(
        2 * np.pi * frequency * t
    )

    signal_int16 = np.int16(
    signal * 32767
    )
    

    audio_buffer = io.BytesIO()

    write(
        audio_buffer,
        sample_rate,
        signal_int16
    )

    audio_buffer.seek(0)

    st.audio(
        audio_buffer,
        format="audio/wav"
    )

    st.success(
        f"Generated {frequency} Hz tone."
    )
   # =====================================================
# SECTION 4 — ADAPTIVE TINNITUS PITCH MATCHING
# =====================================================

st.header("4. Adaptive Tinnitus Pitch Matching")

st.write(
    "Listen to the tones and choose the one that is closest "
    "to the pitch of your tinnitus."
)

# -----------------------------
# INITIALIZE SESSION STATE
# -----------------------------

if "pitch_center" not in st.session_state:
    st.session_state.pitch_center = 6000

if "pitch_step" not in st.session_state:
    st.session_state.pitch_step = 2000

if "pitch_round" not in st.session_state:
    st.session_state.pitch_round = 1

if "pitch_finished" not in st.session_state:
    st.session_state.pitch_finished = False

# -----------------------------
# GENERATE CURRENT OPTIONS
# -----------------------------

center = st.session_state.pitch_center
step = st.session_state.pitch_step

pitch_options = [
    max(250, center - step),
    center,
    min(10000, center + step)
]

st.write(f"### Round {st.session_state.pitch_round}")

# -----------------------------
# AUDIO FUNCTION
# -----------------------------

def generate_tone(freq, duration=1.5, volume=0.05):

    sample_rate = 44100

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False
    )

    signal = volume * np.sin(
        2 * np.pi * freq * t
    )

    signal_int16 = np.int16(
        signal / np.max(np.abs(signal)) * 32767
    )

    audio_buffer = io.BytesIO()

    write(
        audio_buffer,
        sample_rate,
        signal_int16
    )

    audio_buffer.seek(0)

    return audio_buffer

# -----------------------------
# SHOW TONES
# -----------------------------

for freq in pitch_options:

    st.subheader(f"{int(freq)} Hz")

    st.audio(
        generate_tone(freq),
        format="audio/wav"
    )

# -----------------------------
# USER CHOICE
# -----------------------------

selected_pitch = st.radio(
    "Which tone is closest to your tinnitus?",
    pitch_options,
    format_func=lambda x: f"{int(x)} Hz",
    key=f"pitch_choice_{st.session_state.pitch_round}"
)

# -----------------------------
# NEXT ROUND
# -----------------------------

if not st.session_state.pitch_finished:

    if st.button("Next Round"):

        st.session_state.pitch_center = selected_pitch

        st.session_state.pitch_step = st.session_state.pitch_step / 2

        st.session_state.pitch_round += 1

        if st.session_state.pitch_step <= 125:

            st.session_state.pitch_finished = True

        st.rerun()

# -----------------------------
# FINAL RESULT
# -----------------------------


if st.session_state.pitch_finished:

    final_pitch = int(st.session_state.pitch_center)

    st.success(
        f"Estimated tinnitus pitch: {final_pitch} Hz"
    )

    # Get patients from database
    cursor.execute("""
    SELECT id, patient_code
    FROM patients
    ORDER BY patient_code
    """)

    patients_for_pitch = cursor.fetchall()

    if len(patients_for_pitch) > 0:

        patient_pitch_options = {
            patient_code: patient_id
            for patient_id, patient_code in patients_for_pitch
        }

        selected_pitch_patient = st.selectbox(
            "Select patient for this result",
            list(patient_pitch_options.keys()),
            key="pitch_patient"
        )

        selected_pitch_patient_id = patient_pitch_options[selected_pitch_patient]

        if st.button("Save Pitch Result"):

            cursor.execute("""
            UPDATE tinnitus_profiles
            SET estimated_pitch_hz = ?
            WHERE patient_id = ?
            """,
            (
                final_pitch,
                selected_pitch_patient_id
            )
            )

            conn.commit()

            st.success(
                f"Pitch result saved for {selected_pitch_patient}."
            )

    if st.button("Restart Pitch Test"):

        st.session_state.pitch_center = 6000
        st.session_state.pitch_step = 2000
        st.session_state.pitch_round = 1
        st.session_state.pitch_finished = False

        st.rerun()
        # =====================================================
# =====================================================
# SECTION 5 — LOUDNESS MATCHING
# =====================================================

st.header("5. Loudness Matching")

st.write(
    "Select a patient, then adjust the relative level until the tone "
    "sounds approximately as loud as the patient's tinnitus."
)

# Get patients with saved tinnitus profiles
cursor.execute("""
SELECT patients.id,
       patients.patient_code,
       tinnitus_profiles.estimated_pitch_hz
FROM patients
JOIN tinnitus_profiles
ON patients.id = tinnitus_profiles.patient_id
ORDER BY patients.patient_code
""")

patients_loudness = cursor.fetchall()

if len(patients_loudness) == 0:

    st.warning(
        "No tinnitus profiles are available yet."
    )

else:

    loudness_patient_options = {
        patient_code: {
            "patient_id": patient_id,
            "pitch": pitch
        }
        for patient_id, patient_code, pitch in patients_loudness
    }

    selected_loudness_patient = st.selectbox(
        "Select patient",
        list(loudness_patient_options.keys()),
        key="loudness_patient"
    )

    patient_data = loudness_patient_options[selected_loudness_patient]

    patient_id_loudness = patient_data["patient_id"]
    saved_pitch = patient_data["pitch"]

    if saved_pitch is None:

        st.warning(
            "This patient does not have a saved pitch result yet."
        )

    else:

        st.info(
            f"Saved tinnitus pitch: {int(saved_pitch)} Hz"
        )

        relative_level = st.slider(
            "Relative level",
            min_value=0.01,
            max_value=0.20,
            value=0.03,
            step=0.01,
            key="loudness_level"
        )

        if st.button("Generate Loudness Test Tone"):

            sample_rate = 44100
            duration = 1.5

            t = np.linspace(
                0,
                duration,
                int(sample_rate * duration),
                endpoint=False
            )

            signal = relative_level * np.sin(
                2 * np.pi * saved_pitch * t
            )

            signal_int16 = np.int16(
                signal * 32767
            )

            audio_buffer = io.BytesIO()

            write(
                audio_buffer,
                sample_rate,
                signal_int16
            )

            audio_buffer.seek(0)

            st.audio(
                audio_buffer,
                format="audio/wav"
            )

            st.info(
                f"Testing {int(saved_pitch)} Hz "
                f"at relative level {relative_level:.2f}"
            )

        if st.button("Save Loudness Match"):

            cursor.execute("""
            UPDATE tinnitus_profiles
            SET loudness_relative = ?
            WHERE patient_id = ?
            """,
            (
                relative_level,
                patient_id_loudness
            )
            )

            conn.commit()

            st.success(
                f"Loudness match saved for {selected_loudness_patient}."
            )
            # =====================================================
# SECTION 6 — PERSONALIZED ACOUSTIC STIMULUS GENERATOR
# =====================================================

st.header("6. Personalized Acoustic Stimulus Generator")

st.write(
    "Generate different acoustic stimuli based on the patient's "
    "saved tinnitus pitch."
)

# Get patients with saved tinnitus pitch
cursor.execute("""
SELECT patients.id,
       patients.patient_code,
       tinnitus_profiles.estimated_pitch_hz
FROM patients
JOIN tinnitus_profiles
ON patients.id = tinnitus_profiles.patient_id
WHERE tinnitus_profiles.estimated_pitch_hz IS NOT NULL
ORDER BY patients.patient_code
""")

stimulus_patients = cursor.fetchall()

if len(stimulus_patients) == 0:

    st.warning(
        "No patient with a saved tinnitus pitch is available."
    )

else:

    stimulus_patient_options = {
        patient_code: {
            "patient_id": patient_id,
            "pitch": pitch
        }
        for patient_id, patient_code, pitch in stimulus_patients
    }

    selected_stimulus_patient = st.selectbox(
        "Select patient",
        list(stimulus_patient_options.keys()),
        key="stimulus_patient"
    )

    stimulus_patient_data = stimulus_patient_options[selected_stimulus_patient]

    saved_pitch = stimulus_patient_data["pitch"]

    st.info(
        f"Saved tinnitus pitch: {int(saved_pitch)} Hz"
    )

    stimulus_type = st.selectbox(
        "Choose stimulus type",
        [
            "Pure Tone",
            "White Noise",
            "Narrow-Band Noise"
        ]
    )

    stimulus_duration = st.slider(
        "Stimulus duration (seconds)",
        min_value=1,
        max_value=5,
        value=2
    )

    stimulus_level = st.slider(
        "Relative level",
        min_value=0.01,
        max_value=0.10,
        value=0.03,
        step=0.01
    )

    bandwidth = st.slider(
        "Bandwidth for narrow-band noise (Hz)",
        min_value=100,
        max_value=2000,
        value=500,
        step=100
    )

    if st.button("Generate Personalized Stimulus"):

        sample_rate = 44100

        number_samples = int(
            sample_rate * stimulus_duration
        )

        t = np.linspace(
            0,
            stimulus_duration,
            number_samples,
            endpoint=False
        )

        # ----------------------------------
        # PURE TONE
        # ----------------------------------

        if stimulus_type == "Pure Tone":

            signal = stimulus_level * np.sin(
                2 * np.pi * saved_pitch * t
            )

        # ----------------------------------
        # WHITE NOISE
        # ----------------------------------

        elif stimulus_type == "White Noise":

            noise = np.random.normal(
                0,
                1,
                number_samples
            )

            noise = noise / np.max(
                np.abs(noise)
            )

            signal = stimulus_level * noise

        # ----------------------------------
        # NARROW-BAND NOISE
        # ----------------------------------

        elif stimulus_type == "Narrow-Band Noise":

            noise = np.random.normal(
                0,
                1,
                number_samples
            )

            frequencies = np.fft.rfftfreq(
                number_samples,
                d=1 / sample_rate
            )

            noise_fft = np.fft.rfft(noise)

            lower_frequency = (
                saved_pitch - bandwidth / 2
            )

            upper_frequency = (
                saved_pitch + bandwidth / 2
            )

            frequency_mask = (
                (frequencies >= lower_frequency)
                &
                (frequencies <= upper_frequency)
            )

            noise_fft[
                ~frequency_mask
            ] = 0

            filtered_noise = np.fft.irfft(
                noise_fft,
                n=number_samples
            )

            if np.max(np.abs(filtered_noise)) > 0:

                filtered_noise = (
                    filtered_noise
                    /
                    np.max(np.abs(filtered_noise))
                )

            signal = (
                stimulus_level
                *
                filtered_noise
            )

        # ----------------------------------
        # CONVERT TO WAV
        # ----------------------------------

        signal_int16 = np.int16(
            np.clip(signal, -1, 1) * 32767
        )

        audio_buffer = io.BytesIO()

        write(
            audio_buffer,
            sample_rate,
            signal_int16
        )

        audio_buffer.seek(0)

        st.audio(
            audio_buffer,
            format="audio/wav"
        )

        st.success(
            f"{stimulus_type} generated successfully."
        )
        # =====================================================
# PATIENT FEEDBACK
# =====================================================

st.subheader("Patient Feedback")

comfort_score = st.slider(
    "How comfortable was this sound?",
    min_value=0,
    max_value=10,
    value=5,
    key="comfort_score"
)

tinnitus_noticeability = st.slider(
    "How noticeable is your tinnitus now?",
    min_value=0,
    max_value=10,
    value=5,
    key="noticeability_score"
)

listen_again = st.radio(
    "Would you listen to this sound again?",
    ["Yes", "No"],
    key="listen_again"
)

if st.button("Save Stimulus Trial"):

    selected_patient_id = stimulus_patient_data["patient_id"]

    trial_bandwidth = None

    if stimulus_type == "Narrow-Band Noise":
        trial_bandwidth = bandwidth

    cursor.execute("""
    INSERT INTO stimulus_trials
    (
        patient_id,
        stimulus_type,
        center_frequency_hz,
        bandwidth_hz,
        relative_level,
        comfort_score,
        tinnitus_noticeability,
        listen_again,
        created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
        selected_patient_id,
        stimulus_type,
        saved_pitch,
        trial_bandwidth,
        stimulus_level,
        comfort_score,
        tinnitus_noticeability,
        listen_again,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    )

    conn.commit()

    st.success(
        f"Stimulus trial saved for {selected_stimulus_patient}."
    )
    # =====================================================
# SECTION 7 — SAVED STIMULUS TRIALS
# =====================================================

st.header("7. Saved Stimulus Trials")

cursor.execute("""
SELECT
    patients.patient_code,
    stimulus_trials.stimulus_type,
    stimulus_trials.center_frequency_hz,
    stimulus_trials.bandwidth_hz,
    stimulus_trials.relative_level,
    stimulus_trials.comfort_score,
    stimulus_trials.tinnitus_noticeability,
    stimulus_trials.listen_again,
    stimulus_trials.created_at
FROM stimulus_trials
JOIN patients
ON stimulus_trials.patient_id = patients.id
ORDER BY stimulus_trials.created_at DESC
""")

saved_trials = cursor.fetchall()

if len(saved_trials) == 0:

    st.info("No stimulus trials have been saved yet.")

else:

    import pandas as pd

    trials_df = pd.DataFrame(
        saved_trials,
        columns=[
            "Patient",
            "Stimulus Type",
            "Center Frequency (Hz)",
            "Bandwidth (Hz)",
            "Relative Level",
            "Comfort",
            "Tinnitus Noticeability",
            "Listen Again",
            "Date"
        ]
    )

    st.dataframe(
        trials_df,
        use_container_width=True
    )
    # =====================================================
# SECTION 8 — SIMPLE RESPONSE SCORING
# =====================================================

st.header("8. Stimulus Response Ranking")

cursor.execute("""
SELECT
    patients.patient_code,
    stimulus_trials.stimulus_type,
    stimulus_trials.center_frequency_hz,
    stimulus_trials.bandwidth_hz,
    stimulus_trials.relative_level,
    stimulus_trials.comfort_score,
    stimulus_trials.tinnitus_noticeability,
    stimulus_trials.listen_again
FROM stimulus_trials
JOIN patients
ON stimulus_trials.patient_id = patients.id
""")

ranking_data = cursor.fetchall()

if len(ranking_data) == 0:

    st.info("No stimulus trials are available for ranking.")

else:

    ranking_rows = []

    for row in ranking_data:

        (
            patient_code,
            stimulus_type,
            center_frequency,
            bandwidth,
            relative_level,
            comfort,
            noticeability,
            listen_again
        ) = row

        listen_bonus = 2 if listen_again == "Yes" else 0

        response_score = (
            comfort
            +
            (10 - noticeability)
            +
            listen_bonus
        )

        ranking_rows.append(
            [
                patient_code,
                stimulus_type,
                center_frequency,
                bandwidth,
                relative_level,
                comfort,
                noticeability,
                listen_again,
                response_score
            ]
        )

    ranking_df = pd.DataFrame(
        ranking_rows,
        columns=[
            "Patient",
            "Stimulus Type",
            "Frequency (Hz)",
            "Bandwidth (Hz)",
            "Relative Level",
            "Comfort",
            "Noticeability",
            "Listen Again",
            "Response Score"
        ]
    )

    ranking_df = ranking_df.sort_values(
        by="Response Score",
        ascending=False
    )

    st.dataframe(
        ranking_df,
        use_container_width=True
    )
    # =====================================================
# SECTION 9 — NEXT STIMULUS RECOMMENDATION
# =====================================================

st.header("9. Next Stimulus Recommendation")

# Get available patients
cursor.execute("""
SELECT id, patient_code
FROM patients
ORDER BY patient_code
""")

patients_recommendation = cursor.fetchall()

if len(patients_recommendation) == 0:

    st.info("No patients available.")

else:

    patient_recommendation_options = {
        patient_code: patient_id
        for patient_id, patient_code in patients_recommendation
    }

    selected_recommendation_patient = st.selectbox(
        "Select patient",
        list(patient_recommendation_options.keys()),
        key="recommendation_patient"
    )

    recommendation_patient_id = patient_recommendation_options[
        selected_recommendation_patient
    ]

    # Get this patient's trials
    cursor.execute("""
    SELECT
        stimulus_type,
        center_frequency_hz,
        bandwidth_hz,
        relative_level,
        comfort_score,
        tinnitus_noticeability,
        listen_again
    FROM stimulus_trials
    WHERE patient_id = ?
    """, (recommendation_patient_id,))

    patient_trials = cursor.fetchall()

    if len(patient_trials) == 0:

        st.warning(
            "This patient does not have any saved stimulus trials yet."
        )

    else:

        scored_trials = []

        for trial in patient_trials:

            (
                stimulus_type,
                center_frequency,
                bandwidth,
                relative_level,
                comfort,
                noticeability,
                listen_again
            ) = trial

            listen_bonus = 2 if listen_again == "Yes" else 0

            response_score = (
                comfort
                +
                (10 - noticeability)
                +
                listen_bonus
            )

            scored_trials.append(
                {
                    "stimulus_type": stimulus_type,
                    "frequency": center_frequency,
                    "bandwidth": bandwidth,
                    "level": relative_level,
                    "score": response_score
                }
            )

        # Find best previous trial
        best_trial = max(
            scored_trials,
            key=lambda x: x["score"]
        )

        st.subheader("Best Previous Trial")

        st.write(
            f"Stimulus type: **{best_trial['stimulus_type']}**"
        )

        st.write(
            f"Frequency: **{int(best_trial['frequency'])} Hz**"
        )

        if best_trial["bandwidth"] is not None:
            st.write(
                f"Bandwidth: **{int(best_trial['bandwidth'])} Hz**"
            )

        st.write(
            f"Relative level: **{best_trial['level']:.2f}**"
        )

        st.write(
            f"Response score: **{best_trial['score']}**"
        )

        # ---------------------------------------------
        # SIMPLE NEXT-STIMULUS LOGIC
        # ---------------------------------------------

        recommended_type = best_trial["stimulus_type"]
        recommended_frequency = best_trial["frequency"]
        recommended_level = best_trial["level"]

        if best_trial["bandwidth"] is not None:

            recommended_bandwidth = max(
                100,
                best_trial["bandwidth"] - 100
            )

        else:
            recommended_bandwidth = 500

        st.subheader("Suggested Next Trial")

        st.write(
            f"Stimulus type: **{recommended_type}**"
        )

        st.write(
            f"Frequency: **{int(recommended_frequency)} Hz**"
        )

        if recommended_type == "Narrow-Band Noise":

            st.write(
                f"Suggested bandwidth: "
                f"**{int(recommended_bandwidth)} Hz**"
            )

        st.write(
            f"Relative level: **{recommended_level:.2f}**"
        )
        # ---------------------------------------------
# GENERATE RECOMMENDED STIMULUS
# ---------------------------------------------

if st.button("Generate Recommended Stimulus"):

    sample_rate = 44100
    duration = 2

    number_samples = int(
        sample_rate * duration
    )

    t = np.linspace(
        0,
        duration,
        number_samples,
        endpoint=False
    )

    # PURE TONE
    if recommended_type == "Pure Tone":

        signal = recommended_level * np.sin(
            2 * np.pi * recommended_frequency * t
        )

    # WHITE NOISE
    elif recommended_type == "White Noise":

        noise = np.random.normal(
            0,
            1,
            number_samples
        )

        noise = noise / np.max(
            np.abs(noise)
        )

        signal = recommended_level * noise

    # NARROW-BAND NOISE
    elif recommended_type == "Narrow-Band Noise":

        noise = np.random.normal(
            0,
            1,
            number_samples
        )

        frequencies = np.fft.rfftfreq(
            number_samples,
            d=1 / sample_rate
        )

        noise_fft = np.fft.rfft(noise)

        lower_frequency = (
            recommended_frequency
            - recommended_bandwidth / 2
        )

        upper_frequency = (
            recommended_frequency
            + recommended_bandwidth / 2
        )

        frequency_mask = (
            (frequencies >= lower_frequency)
            &
            (frequencies <= upper_frequency)
        )

        noise_fft[
            ~frequency_mask
        ] = 0

        filtered_noise = np.fft.irfft(
            noise_fft,
            n=number_samples
        )

        if np.max(np.abs(filtered_noise)) > 0:

            filtered_noise = (
                filtered_noise
                /
                np.max(np.abs(filtered_noise))
            )

        signal = (
            recommended_level
            *
            filtered_noise
        )

    signal_int16 = np.int16(
        np.clip(signal, -1, 1) * 32767
    )

    audio_buffer = io.BytesIO()

    write(
        audio_buffer,
        sample_rate,
        signal_int16
    )

    audio_buffer.seek(0)

    st.audio(
        audio_buffer,
        format="audio/wav"
    )

    st.success(
        "Recommended stimulus generated."
    )
    # ---------------------------------------------
# FEEDBACK FOR RECOMMENDED STIMULUS
# ---------------------------------------------

st.subheader("Rate Recommended Stimulus")

recommended_comfort = st.slider(
    "How comfortable was this recommended sound?",
    min_value=0,
    max_value=10,
    value=5,
    key="recommended_comfort"
)

recommended_noticeability = st.slider(
    "How noticeable is your tinnitus after this sound?",
    min_value=0,
    max_value=10,
    value=5,
    key="recommended_noticeability"
)

recommended_listen_again = st.radio(
    "Would you listen to this recommended sound again?",
    ["Yes", "No"],
    key="recommended_listen_again"
)

if st.button("Save Recommended Trial"):

    cursor.execute("""
    INSERT INTO stimulus_trials
    (
        patient_id,
        stimulus_type,
        center_frequency_hz,
        bandwidth_hz,
        relative_level,
        comfort_score,
        tinnitus_noticeability,
        listen_again,
        created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
        recommendation_patient_id,
        recommended_type,
        recommended_frequency,
        recommended_bandwidth
        if recommended_type == "Narrow-Band Noise"
        else None,
        recommended_level,
        recommended_comfort,
        recommended_noticeability,
        recommended_listen_again,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()

    st.success(
        f"Recommended trial saved for {selected_recommendation_patient}."
    )
