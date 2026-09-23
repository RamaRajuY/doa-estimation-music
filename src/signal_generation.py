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


def generate_received_signal(
    num_elements,
    num_samples,
    spacing,
    wavelength,
    angle_deg,
    frequency=0.05
):
    """
    Generate the signals received by a ULA from a single source.

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

    received_signal = (
        steering[:, np.newaxis]
        * source_signal[np.newaxis, :]
    )

    return received_signal