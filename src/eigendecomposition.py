import numpy as np


def calculate_eigendecomposition(covariance_matrix):
    """
    Perform eigenvalue decomposition of a Hermitian
    spatial covariance matrix.

    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        Spatial covariance matrix.

    Returns
    -------
    eigenvalues : numpy.ndarray
        Eigenvalues in ascending order.

    eigenvectors : numpy.ndarray
        Corresponding eigenvectors.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance_matrix
    )

    return eigenvalues, eigenvectors