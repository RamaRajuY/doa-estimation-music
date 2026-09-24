import numpy as np
import matplotlib.pyplot as plt

from src.signal_generation import generate_multiple_received_signal
from src.covariance import calculate_covariance_matrix
from src.eigendecomposition import calculate_eigendecomposition
from src.music import (
    calculate_music_spectrum,
    estimate_multiple_doa,
)


# ============================================================
# Experiment configuration
# ============================================================

NUM_ELEMENTS = 8
NUM_SNAPSHOTS = 1000

WAVELENGTH = 1.0
SPACING = WAVELENGTH / 2

TRUE_DOAS = np.array([-20.0, 35.0])

NUM_SOURCES = len(TRUE_DOAS)

SNR_DB = 10.0

RANDOM_SEED = 42

ANGLE_GRID = np.linspace(-90.0, 90.0, 1801)


# ============================================================
# Generate received array data
# ============================================================

received_signal = generate_multiple_received_signal(
    num_elements=NUM_ELEMENTS,
    num_samples=NUM_SNAPSHOTS,
    spacing=SPACING,
    wavelength=WAVELENGTH,
    angles_deg=TRUE_DOAS,
    snr_db=SNR_DB,
    seed=RANDOM_SEED,
)


# ============================================================
# Calculate spatial covariance matrix
# ============================================================

covariance_matrix = calculate_covariance_matrix(
    received_signal
)


# ============================================================
# Eigenvalue decomposition
# ============================================================

eigenvalues, eigenvectors = calculate_eigendecomposition(
    covariance_matrix
)


# ============================================================
# MUSIC spectrum
# ============================================================

music_spectrum = calculate_music_spectrum(
    eigenvectors=eigenvectors,
    num_sources=NUM_SOURCES,
    num_elements=NUM_ELEMENTS,
    spacing=SPACING,
    wavelength=WAVELENGTH,
    angle_grid=ANGLE_GRID,
)


# ============================================================
# DOA estimation
# ============================================================

estimated_doas = estimate_multiple_doa(
    angle_grid=ANGLE_GRID,
    music_spectrum=music_spectrum,
    num_sources=NUM_SOURCES,
)


# ============================================================
# Calculate DOA errors
# ============================================================

doa_errors = estimated_doas - TRUE_DOAS


# ============================================================
# Display experiment summary
# ============================================================

print("=" * 50)
print("        MUSIC DOA ESTIMATION")
print("=" * 50)

print("\nArray configuration")
print(f"Number of antennas : {NUM_ELEMENTS}")
print(f"Element spacing    : {SPACING:.2f} λ")

print("\nSignal configuration")
print(f"Number of sources  : {NUM_SOURCES}")
print(f"True DOAs          : {TRUE_DOAS}")
print(f"SNR                : {SNR_DB:.1f} dB")
print(f"Snapshots          : {NUM_SNAPSHOTS}")

print("\nEigenvalues")
print(np.round(eigenvalues, 4))

print("\nResults")

for i, (true_doa, estimated_doa, error) in enumerate(
    zip(TRUE_DOAS, estimated_doas, doa_errors),
    start=1,
):
    print(
        f"Source {i}: "
        f"True = {true_doa:.1f}°, "
        f"Estimated = {estimated_doa:.1f}°, "
        f"Error = {error:.1f}°"
    )


# ============================================================
# Plot MUSIC spectrum
# ============================================================

music_spectrum_db = 10 * np.log10(
    music_spectrum / np.max(music_spectrum)
)

plt.figure(figsize=(10, 6))

plt.plot(
    ANGLE_GRID,
    music_spectrum_db,
    label="MUSIC Spectrum",
)

# True DOA markers
for true_doa in TRUE_DOAS:
    plt.axvline(
        true_doa,
        linestyle=":",
        label=f"True DOA = {true_doa:.1f}°",
    )

# Estimated DOA markers
for estimated_doa in estimated_doas:
    plt.axvline(
        estimated_doa,
        linestyle="--",
        label=f"Estimated DOA = {estimated_doa:.1f}°",
    )

plt.xlabel("Angle (degrees)")
plt.ylabel("Normalized MUSIC Spectrum (dB)")

plt.title("Two-Source MUSIC Direction of Arrival Estimation")

plt.xlim(-90, 90)
plt.ylim(-60, 5)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/music_spectrum_two_sources.png",
    dpi=300,
)

plt.show()