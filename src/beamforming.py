import numpy as np

from src.array_model import steering_vector


def calculate_beamforming_spectrum(
    covariance_matrix,
    num_elements,
    spacing,
    wavelength,
    angle_grid
):
    """
    Calculate the conventional Bartlett beamforming spectrum.

    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        Spatial covariance matrix.

    num_elements : int
        Number of antenna elements.

    spacing : float
        Antenna element spacing.

    wavelength : float
        Signal wavelength.

    angle_grid : numpy.ndarray
        Angles in degrees over which to evaluate the spectrum.

    Returns
    -------
    numpy.ndarray
        Conventional beamforming spectrum.
    """

    beamforming_spectrum = np.zeros(angle_grid.shape)

    for index, angle in enumerate(angle_grid):

        steering = steering_vector(
            num_elements=num_elements,
            spacing=spacing,
            wavelength=wavelength,
            angle_deg=angle
        )

        numerator = (
            steering.conj().T
            @ covariance_matrix
            @ steering
        )

        denominator = (
            steering.conj().T
            @ steering
        )

        beamforming_spectrum[index] = (
            np.real(numerator / denominator)
        )

    return beamforming_spectrum