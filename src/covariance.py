import numpy as np


def calculate_covariance_matrix(received_signal):
    """
    Calculate the spatial covariance matrix.

    Parameters
    ----------
    received_signal : numpy.ndarray
        Complex array data with shape
        (num_elements, num_snapshots).

    Returns
    -------
    numpy.ndarray
        Spatial covariance matrix.
    """

    num_snapshots = received_signal.shape[1]

    covariance_matrix = (
        received_signal @ received_signal.conj().T
        / num_snapshots
    )

    return covariance_matrix