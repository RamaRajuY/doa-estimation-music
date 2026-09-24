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
def estimate_multiple_doa(
    angle_grid,
    music_spectrum,
    num_sources,
    min_peak_separation_deg=2.0
):
    """
    Estimate multiple DOAs from the MUSIC pseudospectrum.

    Parameters
    ----------
    angle_grid : numpy.ndarray
        Scanned angles in degrees.

    music_spectrum : numpy.ndarray
        MUSIC pseudospectrum.

    num_sources : int
        Number of expected signal sources.

    min_peak_separation_deg : float
        Minimum angular separation between detected peaks.

    Returns
    -------
    numpy.ndarray
        Estimated DOAs in ascending order.
    """

    # Find local maxima in the MUSIC spectrum.
    peak_indices = []

    for i in range(1, len(music_spectrum) - 1):
        if (
            music_spectrum[i] > music_spectrum[i - 1]
            and music_spectrum[i] > music_spectrum[i + 1]
        ):
            peak_indices.append(i)

    if not peak_indices:
        raise RuntimeError("No MUSIC spectrum peaks were found.")

    # Rank peaks from strongest to weakest.
    peak_indices.sort(
        key=lambda i: music_spectrum[i],
        reverse=True
    )

    selected_indices = []

    for index in peak_indices:

        angle = angle_grid[index]

        too_close = any(
            abs(angle - angle_grid[selected])
            < min_peak_separation_deg
            for selected in selected_indices
        )

        if not too_close:
            selected_indices.append(index)

        if len(selected_indices) == num_sources:
            break

    if len(selected_indices) < num_sources:
        raise RuntimeError(
            f"Only {len(selected_indices)} distinct peaks found; "
            f"expected {num_sources}."
        )

    return np.sort(angle_grid[selected_indices])