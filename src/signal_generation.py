import numpy as np

from src.array_model import steering_vector


def generate_source_signal(num_samples, frequency=0.05):
    """
    Generate a complex baseband source signal.

    Parameters
    ----------
    num_samples : int
        Number of time samples.

    frequency : float
        Normalized signal frequency.

    Returns
    -------
    numpy.ndarray
        Complex source signal.
    """

    n = np.arange(num_samples)

    return np.exp(1j * 2 * np.pi * frequency * n)


def generate_awgn_noise(signal, snr_db):
    """
    Generate complex AWGN for a desired signal-to-noise ratio.

    Parameters
    ----------
    signal : numpy.ndarray
        Reference signal used to determine signal power.

    snr_db : float
        Desired signal-to-noise ratio in dB.

    Returns
    -------
    numpy.ndarray
        Complex Gaussian noise with the same shape as signal.
    """

    signal_power = np.mean(np.abs(signal) ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape)
        + 1j * np.random.randn(*signal.shape)
    )

    return noise


def generate_received_signal(
    num_elements,
    num_samples,
    spacing,
    wavelength,
    angle_deg,
    frequency=0.05,
    snr_db=10.0
):
    """
    Generate noisy signals received by a ULA from a single source.

    Returns
    -------
    numpy.ndarray
        Array data with shape (num_elements, num_samples).
    """

    source_signal = generate_source_signal(
        num_samples,
        frequency
    )

    steering = steering_vector(
        num_elements,
        spacing,
        wavelength,
        angle_deg
    )

    clean_signal = (
        steering[:, np.newaxis]
        * source_signal[np.newaxis, :]
    )

    noise = generate_awgn_noise(
        clean_signal,
        snr_db
    )

    received_signal = clean_signal + noise

    return received_signal

def generate_multiple_received_signal(
    num_elements,
    num_samples,
    spacing,
    wavelength,
    angles_deg,
    snr_db=10.0,
    seed=42
):
    """
    Generate noisy signals received by a ULA from
    multiple independent sources.

    Parameters
    ----------
    num_elements : int
        Number of antenna elements.

    num_samples : int
        Number of snapshots.

    spacing : float
        Antenna element spacing.

    wavelength : float
        Signal wavelength.

    angles_deg : list or numpy.ndarray
        Directions of arrival in degrees.

    snr_db : float
        Desired signal-to-noise ratio in dB.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Received array data with shape
        (num_elements, num_samples).
    """

    rng = np.random.default_rng(seed)

    angles_deg = np.asarray(angles_deg)

    num_sources = len(angles_deg)

    # Generate independent complex Gaussian source signals.
    source_signals = (
        rng.standard_normal((num_sources, num_samples))
        + 1j * rng.standard_normal((num_sources, num_samples))
    ) / np.sqrt(2)

    # Normalize each source to unit average power.
    source_signals /= np.sqrt(
        np.mean(np.abs(source_signals) ** 2, axis=1)
    )[:, np.newaxis]

    # Build the steering matrix.
    steering_matrix = np.column_stack([
        steering_vector(
            num_elements=num_elements,
            spacing=spacing,
            wavelength=wavelength,
            angle_deg=angle
        )
        for angle in angles_deg
    ])

    # Received signal:
    # X = A @ S
    clean_signal = steering_matrix @ source_signals

    # Generate complex AWGN.
    signal_power = np.mean(np.abs(clean_signal) ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(clean_signal.shape)
        + 1j * rng.standard_normal(clean_signal.shape)
    )

    received_signal = clean_signal + noise

    return received_signal