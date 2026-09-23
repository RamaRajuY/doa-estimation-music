import numpy as np

from src.array_model import steering_vector


def calculate_music_spectrum(
    eigenvectors,
    num_sources,
    num_elements,
    spacing,
    wavelength,
    angle_grid
):
    """
    Calculate the MUSIC pseudospectrum.

    Parameters
    ----------
    eigenvectors : numpy.ndarray
        Eigenvector matrix from the covariance matrix.

    num_sources : int
        Number of signal sources.

    num_elements : int
        Number of antenna elements.

    spacing : float
        Antenna element spacing.

    wavelength : float
        Signal wavelength.

    angle_grid : numpy.ndarray
        Angles, in degrees, over which to evaluate MUSIC.

    Returns
    -------
    numpy.ndarray
        MUSIC pseudospectrum.
    """

    # np.linalg.eigh() returns eigenvectors in the same
    # order as the ascending eigenvalues.
    noise_subspace = eigenvectors[:, :-num_sources]

    music_spectrum = np.zeros(angle_grid.shape)

    noise_projection = (
        noise_subspace @ noise_subspace.conj().T
    )

    for index, angle in enumerate(angle_grid):

        steering = steering_vector(
            num_elements=num_elements,
            spacing=spacing,
            wavelength=wavelength,
            angle_deg=angle
        )

        denominator = (
            steering.conj().T
            @ noise_projection
            @ steering
        )

        music_spectrum[index] = 1.0 / np.abs(denominator)

    return music_spectrum


def estimate_doa(angle_grid, music_spectrum):
    """
    Estimate DOA from the maximum MUSIC spectrum value.

    Parameters
    ----------
    angle_grid : numpy.ndarray
        Scanned angles in degrees.

    music_spectrum : numpy.ndarray
        MUSIC pseudospectrum.

    Returns
    -------
    float
        Estimated direction of arrival in degrees.
    """

    peak_index = np.argmax(music_spectrum)

    return angle_grid[peak_index]