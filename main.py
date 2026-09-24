import numpy as np
import matplotlib.pyplot as plt

from src.signal_generation import generate_multiple_received_signal
from src.covariance import calculate_covariance_matrix
from src.eigendecomposition import calculate_eigendecomposition
from src.music import (
    calculate_music_spectrum,
    estimate_multiple_doa
)

NUM_ELEMENTS = 8
NUM_SAMPLES = 1000

WAVELENGTH = 1.0
SPACING = WAVELENGTH / 2

DOAS = [-20.0, 35.0]

SNR_DB = 10.0
NUM_SOURCES = len(DOAS)

received_signal = generate_multiple_received_signal(
    num_elements=NUM_ELEMENTS,
    num_samples=NUM_SAMPLES,
    spacing=SPACING,
    wavelength=WAVELENGTH,
    angles_deg=DOAS,
    snr_db=SNR_DB,
    seed=42
)

covariance_matrix = calculate_covariance_matrix(received_signal)

eigenvalues, eigenvectors = calculate_eigendecomposition(
    covariance_matrix
)

ANGLE_GRID = np.linspace(-90.0, 90.0, 1801)
NUM_SOURCES = len(DOAS)


music_spectrum = calculate_music_spectrum(
    eigenvectors=eigenvectors,
    num_sources=NUM_SOURCES,
    num_elements=NUM_ELEMENTS,
    spacing=SPACING,
    wavelength=WAVELENGTH,
    angle_grid=ANGLE_GRID
)


estimated_doas = estimate_multiple_doa(
    angle_grid=ANGLE_GRID,
    music_spectrum=music_spectrum,
    num_sources=NUM_SOURCES
)

print("\nEstimated DOAs:")

for doa in estimated_doas:
    print(f"{doa:.1f} degrees")

# Normalize the MUSIC spectrum and convert to dB
music_spectrum_db = 10 * np.log10(
    music_spectrum / np.max(music_spectrum)
)

# Plot MUSIC spectrum
plt.figure(figsize=(10, 6))

plt.plot(
    ANGLE_GRID,
    music_spectrum_db,
    label="MUSIC Spectrum"
)

for doa in estimated_doas:
    plt.axvline(
        doa,
        linestyle="--",
        label=f"Estimated DOA = {doa:.1f}°"
    )

plt.xlabel("Angle (degrees)")
plt.ylabel("Normalized MUSIC Spectrum (dB)")
plt.title("MUSIC Direction of Arrival Estimation")

plt.grid(True)
plt.legend()

plt.xlim(-90, 90)
plt.ylim(-60, 5)

plt.tight_layout()

plt.savefig(
    "results/music_spectrum.png",
    dpi=300
)

plt.show()

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