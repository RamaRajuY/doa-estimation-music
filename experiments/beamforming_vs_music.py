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

def calculate_3db_peak_width(
    angle_grid,
    spectrum_db,
    peak_angle,
    search_window_deg=10.0
):
    """
    Calculate the -3 dB spectral peak width around a source.

    The function first finds the actual local maximum near the
    expected peak angle, then finds the -3 dB crossing points
    on both sides using linear interpolation.

    Parameters
    ----------
    angle_grid : numpy.ndarray
        Angle values in degrees.

    spectrum_db : numpy.ndarray
        Normalized spectrum in dB.

    peak_angle : float
        Expected source angle.

    search_window_deg : float
        Search window around the expected source angle.

    Returns
    -------
    float
        -3 dB spectral peak width in degrees.
    """

    # Find samples inside the local search window.
    mask = (
        (angle_grid >= peak_angle - search_window_deg)
        & (angle_grid <= peak_angle + search_window_deg)
    )

    local_indices = np.where(mask)[0]

    if len(local_indices) < 3:
        raise RuntimeError(
            "Not enough samples in the peak search window."
        )

    # Find the actual local peak.
    local_peak_index = local_indices[
        np.argmax(spectrum_db[local_indices])
    ]

    threshold = spectrum_db[local_peak_index] - 3.0

    # --------------------------------------------------------
    # Find left -3 dB crossing
    # --------------------------------------------------------

    left_index = local_peak_index

    while (
        left_index > 0
        and spectrum_db[left_index] > threshold
    ):
        left_index -= 1

    if left_index == 0:
        raise RuntimeError(
            "Left -3 dB crossing was not found."
        )

    x1 = angle_grid[left_index]
    x2 = angle_grid[left_index + 1]

    y1 = spectrum_db[left_index]
    y2 = spectrum_db[left_index + 1]

    left_crossing = x1 + (
        (threshold - y1)
        / (y2 - y1)
        * (x2 - x1)
    )

    # --------------------------------------------------------
    # Find right -3 dB crossing
    # --------------------------------------------------------

    right_index = local_peak_index

    while (
        right_index < len(spectrum_db) - 1
        and spectrum_db[right_index] > threshold
    ):
        right_index += 1

    if right_index == len(spectrum_db) - 1:
        raise RuntimeError(
            "Right -3 dB crossing was not found."
        )

    x1 = angle_grid[right_index - 1]
    x2 = angle_grid[right_index]

    y1 = spectrum_db[right_index - 1]
    y2 = spectrum_db[right_index]

    right_crossing = x1 + (
        (threshold - y1)
        / (y2 - y1)
        * (x2 - x1)
    )

    return right_crossing - left_crossing

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
# Calculate -3 dB beamwidths around the true DOAs
# ============================================================

print("\n-3 dB Beamwidths")

for true_doa in TRUE_DOAS:

    beamforming_width = calculate_3db_peak_width(
        ANGLE_GRID,
        beamforming_spectrum_db,
        true_doa,
    )

    music_width = calculate_3db_peak_width(
        ANGLE_GRID,
        music_spectrum_db,
        true_doa,
    )

    print(
        f"\nSource at {true_doa:.1f}°"
    )

    print(
    f"Conventional Beamforming : "
    f"{beamforming_width:.3f}°"
    )

    print(
    f"MUSIC spectral peak width : "
    f"{music_width:.3f}°"
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