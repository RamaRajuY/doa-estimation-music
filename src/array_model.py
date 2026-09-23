import numpy as np


def steering_vector(num_elements, spacing, wavelength, angle_deg):
    """
    Generate the steering vector for a Uniform Linear Array (ULA).

    Parameters
    ----------
    num_elements : int
        Number of antenna elements.

    spacing : float
        Distance between adjacent antenna elements.

    wavelength : float
        Wavelength of the received signal.

    angle_deg : float
        Direction of arrival in degrees.

    Returns
    -------
    numpy.ndarray
        Complex steering vector.
    """

    angle_rad = np.deg2rad(angle_deg)

    element_indices = np.arange(num_elements)

    phase_shift = (
        -2j
        * np.pi
        * spacing
        / wavelength
        * np.sin(angle_rad)
        * element_indices
    )

    return np.exp(phase_shift)