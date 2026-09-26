import numpy as np
import matplotlib.pyplot as plt
import time

from src.signal_generation import generate_multiple_received_signal
from src.music import calculate_music_spectrum, estimate_multiple_doa
from src.esprit import estimate_esprit_doa


# --------------------------------------------------------
# Simulation parameters
# --------------------------------------------------------

NUM_ELEMENTS = 8
SPACING = 0.5
WAVELENGTH = 1.0

TRUE_DOAS = np.array([-20.0, 35.0])
NUM_SOURCES = len(TRUE_DOAS)

NUM_SNAPSHOTS = 1000
SNR_DB = 10.0
SEED = 42

ANGLE_GRID = np.linspace(-90.0, 90.0, 1801)


# --------------------------------------------------------
# Generate one common received signal
# --------------------------------------------------------

received_signal = generate_multiple_received_signal(
    num_elements=NUM_ELEMENTS,
    angles_deg=TRUE_DOAS,
    num_samples=NUM_SNAPSHOTS,
    spacing=SPACING,
    wavelength=WAVELENGTH,
    snr_db=SNR_DB,
    seed=SEED,
)


# --------------------------------------------------------
# Calculate one common covariance matrix
# --------------------------------------------------------

num_snapshots = received_signal.shape[1]

covariance_matrix = (
    received_signal @ received_signal.conj().T
) / num_snapshots


# --------------------------------------------------------
# MUSIC estimation
# --------------------------------------------------------

music_start = time.perf_counter()

eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

music_spectrum = calculate_music_spectrum(
    eigenvectors=eigenvectors,
    num_sources=NUM_SOURCES,
    num_elements=NUM_ELEMENTS,
    spacing=SPACING,
    wavelength=WAVELENGTH,
    angle_grid=ANGLE_GRID,
)

music_doas = estimate_multiple_doa(
    angle_grid=ANGLE_GRID,
    music_spectrum=music_spectrum,
    num_sources=NUM_SOURCES,
)

music_time = time.perf_counter() - music_start


# --------------------------------------------------------
# ESPRIT estimation
# --------------------------------------------------------

esprit_start = time.perf_counter()

esprit_doas = estimate_esprit_doa(
    covariance_matrix=covariance_matrix,
    num_sources=NUM_SOURCES,
    spacing=SPACING,
    wavelength=WAVELENGTH,
)

esprit_time = time.perf_counter() - esprit_start


# --------------------------------------------------------
# Calculate estimation errors
# --------------------------------------------------------

music_errors = np.abs(music_doas - TRUE_DOAS)
esprit_errors = np.abs(esprit_doas - TRUE_DOAS)

music_rmse = np.sqrt(np.mean((music_doas - TRUE_DOAS) ** 2))
esprit_rmse = np.sqrt(np.mean((esprit_doas - TRUE_DOAS) ** 2))


# --------------------------------------------------------
# Print results
# --------------------------------------------------------

print("MUSIC vs ESPRIT Comparison")
print("-" * 40)

print("\nMUSIC")
for i in range(NUM_SOURCES):
    print(
        f"Source {i + 1}: "
        f"True = {TRUE_DOAS[i]:.2f}°, "
        f"Estimated = {music_doas[i]:.2f}°, "
        f"Error = {music_errors[i]:.2f}°"
    )

print(f"MUSIC RMSE = {music_rmse:.4f}°")
print(f"MUSIC execution time = {music_time * 1000:.3f} ms")

print("\nESPRIT")
for i in range(NUM_SOURCES):
    print(
        f"Source {i + 1}: "
        f"True = {TRUE_DOAS[i]:.2f}°, "
        f"Estimated = {esprit_doas[i]:.2f}°, "
        f"Error = {esprit_errors[i]:.2f}°"
    )

print(f"ESPRIT RMSE = {esprit_rmse:.4f}°")
print(f"ESPRIT execution time = {esprit_time * 1000:.3f} ms")


# --------------------------------------------------------
# Plot MUSIC spectrum and mark ESPRIT estimates
# --------------------------------------------------------

normalized_music = music_spectrum / np.max(music_spectrum)
music_spectrum_db = 10.0 * np.log10(normalized_music)

plt.figure(figsize=(10, 5))
plt.plot(
    ANGLE_GRID,
    music_spectrum_db,
    label="MUSIC spectrum",
)

for doa in TRUE_DOAS:
    plt.axvline(
        doa,
        linestyle="--",
        label=f"True DOA = {doa:.0f}°",
    )

for doa in esprit_doas:
    plt.axvline(
        doa,
        linestyle=":",
        label=f"ESPRIT estimate = {doa:.2f}°",
    )

plt.xlabel("Angle (degrees)")
plt.ylabel("Normalized MUSIC spectrum (dB)")
plt.title("MUSIC vs ESPRIT DOA Estimation")
plt.xlim(-90, 90)
plt.ylim(-50, 2)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/esprit_vs_music.png",
    dpi=300,
)

plt.show()
