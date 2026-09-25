import numpy as np


def estimate_esprit_doa(
    covariance_matrix,
    num_sources,
    spacing,
    wavelength,
):
    """
    Estimate DOAs using the ESPRIT algorithm.

    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        Spatial covariance matrix.

    num_sources : int
        Number of signal sources.

    spacing : float
        Antenna element spacing.

    wavelength : float
        Signal wavelength.

    Returns
    -------
    numpy.ndarray
        Estimated DOAs in degrees.
    """

    # --------------------------------------------------------
    # Eigenvalue decomposition
    # --------------------------------------------------------

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance_matrix
    )

    # np.linalg.eigh() returns eigenvalues in ascending order.
    # The eigenvectors corresponding to the largest
    # eigenvalues form the signal subspace.
    signal_subspace = eigenvectors[:, -num_sources:]

    # --------------------------------------------------------
    # Selection matrices
    # --------------------------------------------------------

    subarray_1 = signal_subspace[:-1, :]
    subarray_2 = signal_subspace[1:, :]

    # --------------------------------------------------------
    # Solve the rotational-invariance equation
    #
    # subarray_2 = subarray_1 @ Psi
    # --------------------------------------------------------

    psi = np.linalg.pinv(subarray_1) @ subarray_2

    # --------------------------------------------------------
    # Find eigenvalues of Psi
    # --------------------------------------------------------

    rotation_eigenvalues = np.linalg.eigvals(psi)

    # --------------------------------------------------------
    # Extract spatial phase shifts
    # --------------------------------------------------------

    spatial_phases = np.angle(rotation_eigenvalues)

    # Our steering-vector convention is:
    #
    # exp(-j * 2*pi*d/lambda*sin(theta))
    #
    # Therefore the minus sign is required here.
    sin_theta = -(
        spatial_phases
        / (
            2
            * np.pi
            * spacing
            / wavelength
        )
    )

    # Protect against very small numerical errors.
    sin_theta = np.clip(
        sin_theta,
        -1.0,
        1.0,
    )

    estimated_doas = np.rad2deg(
        np.arcsin(sin_theta)
    )

    return np.sort(estimated_doas)