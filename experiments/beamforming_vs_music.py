import numpy as np
import matplotlib.pyplot as plt

from src.signal_generation import generate_multiple_received_signal
from src.covariance import calculate_covariance_matrix
from src.eigendecomposition import calculate_eigendecomposition
from src.music import calculate_music_spectrum
from src.beamforming import calculate_beamforming_spectrum


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
# Generate the same received data used for both methods
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
# Calculate covariance matrix
# ============================================================

covariance_matrix = calculate_covariance_matrix(
    received_signal
)


# ============================================================
# Calculate eigenvalue decomposition for MUSIC
# ============================================================

eigenvalues, eigenvectors = calculate_eigendecomposition(
    covariance_matrix
)


# ============================================================
# Conventional beamforming
# ============================================================

beamforming_spectrum = calculate_beamforming_spectrum(
    covariance_matrix=covariance_matrix,
    num_elements=NUM_ELEMENTS,
    spacing=SPACING,
    wavelength=WAVELENGTH,
    angle_grid=ANGLE_GRID,
)


# ============================================================
# MUSIC
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
# Normalize spectra and convert to dB
# ============================================================

beamforming_spectrum_db = 10 * np.log10(
    beamforming_spectrum / np.max(beamforming_spectrum)
)

music_spectrum_db = 10 * np.log10(
    music_spectrum / np.max(music_spectrum)
)


# ============================================================
# Display experiment information
# ============================================================

print("=" * 60)
print("     CONVENTIONAL BEAMFORMING VS MUSIC")
print("=" * 60)

print("\nExperiment configuration")
print(f"Number of antennas : {NUM_ELEMENTS}")
print(f"Snapshots          : {NUM_SNAPSHOTS}")
print(f"Element spacing    : {SPACING:.2f} λ")
print(f"Number of sources  : {NUM_SOURCES}")
print(f"True DOAs          : {TRUE_DOAS}")
print(f"SNR                : {SNR_DB:.1f} dB")


# ============================================================
# Plot comparison
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    ANGLE_GRID,
    beamforming_spectrum_db,
    label="Conventional Beamforming",
)

plt.plot(
    ANGLE_GRID,
    music_spectrum_db,
    label="MUSIC",
)

for true_doa in TRUE_DOAS:
    plt.axvline(
        true_doa,
        linestyle=":",
        label=f"True DOA = {true_doa:.1f}°",
    )

plt.xlabel("Angle (degrees)")
plt.ylabel("Normalized Spatial Spectrum (dB)")

plt.title(
    "Conventional Beamforming versus MUSIC"
)

plt.xlim(-90, 90)
plt.ylim(-40, 5)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/beamforming_vs_music.png",
    dpi=300,
)

plt.show()