import numpy as np

from src.signal_generation import generate_received_signal

from src.covariance import calculate_covariance_matrix

from src.eigendecomposition import calculate_eigendecomposition

NUM_ELEMENTS = 8
NUM_SAMPLES = 1000

WAVELENGTH = 1.0
SPACING = WAVELENGTH / 2

DOA = 30.0
SNR_DB = 20.0


received_signal = generate_received_signal(
    num_elements=NUM_ELEMENTS,
    num_samples=NUM_SAMPLES,
    spacing=SPACING,
    wavelength=WAVELENGTH,
    angle_deg=DOA,
    snr_db=SNR_DB
)

covariance_matrix = calculate_covariance_matrix(received_signal)

eigenvalues, eigenvectors = calculate_eigendecomposition(
    covariance_matrix
)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvector matrix shape:")
print(eigenvectors.shape)

print("\nCovariance matrix shape:")
print(covariance_matrix.shape)

print("\nCovariance matrix:")
print(covariance_matrix)

print("Received signal shape:")
print(received_signal.shape)

print("\nSNR:")
print(f"{SNR_DB} dB")

print("\nFirst sample from each antenna:")
print(received_signal[:, 0])

print("\nMagnitude of first sample:")
print(np.abs(received_signal[:, 0]))

print("\nPhase of first sample:")
print(np.angle(received_signal[:, 0], deg=True))