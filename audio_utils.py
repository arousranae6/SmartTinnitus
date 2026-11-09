import numpy as np
import io
from scipy.io.wavfile import write


SAMPLE_RATE = 44100


def signal_to_wav_buffer(signal, sample_rate=SAMPLE_RATE):

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

    return audio_buffer


def generate_pure_tone(
    frequency,
    duration=1.5,
    volume=0.05,
    sample_rate=SAMPLE_RATE
):

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False
    )

    signal = volume * np.sin(
        2 * np.pi * frequency * t
    )

    return signal_to_wav_buffer(
        signal,
        sample_rate
    )


def generate_white_noise(
    duration=2,
    volume=0.03,
    sample_rate=SAMPLE_RATE
):

    number_samples = int(
        sample_rate * duration
    )

    noise = np.random.normal(
        0,
        1,
        number_samples
    )

    max_value = np.max(
        np.abs(noise)
    )

    if max_value > 0:
        noise = noise / max_value

    signal = volume * noise

    return signal_to_wav_buffer(
        signal,
        sample_rate
    )


def generate_narrow_band_noise(
    center_frequency,
    bandwidth=500,
    duration=2,
    volume=0.03,
    sample_rate=SAMPLE_RATE
):

    number_samples = int(
        sample_rate * duration
    )

    noise = np.random.normal(
        0,
        1,
        number_samples
    )

    frequencies = np.fft.rfftfreq(
        number_samples,
        d=1 / sample_rate
    )

    noise_fft = np.fft.rfft(
        noise
    )

    lower_frequency = (
        center_frequency
        -
        bandwidth / 2
    )

    upper_frequency = (
        center_frequency
        +
        bandwidth / 2
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

    max_value = np.max(
        np.abs(filtered_noise)
    )

    if max_value > 0:
        filtered_noise = (
            filtered_noise
            /
            max_value
        )

    signal = (
        volume
        *
        filtered_noise
    )

    return signal_to_wav_buffer(
        signal,
        sample_rate
    )


def generate_stimulus(
    stimulus_type,
    frequency,
    duration=2,
    volume=0.03,
    bandwidth=500
):

    if stimulus_type == "Pure Tone":

        return generate_pure_tone(
            frequency=frequency,
            duration=duration,
            volume=volume
        )

    elif stimulus_type == "White Noise":

        return generate_white_noise(
            duration=duration,
            volume=volume
        )

    elif stimulus_type == "Narrow-Band Noise":

        return generate_narrow_band_noise(
            center_frequency=frequency,
            bandwidth=bandwidth,
            duration=duration,
            volume=volume
        )

    else:

        raise ValueError(
            "Unknown stimulus type."
        )