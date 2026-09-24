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

ANGLE_GRID = np.linspace(-90.0, 90.0, 1801)

SNR_VALUES = np.array([
    -10,
    -5,
    0,
    5,
    10,
    15,
    20,
])

NUM_TRIALS = 50


# ============================================================
# Store results
# ============================================================

rmse_results = []

source_1_errors = []
source_2_errors = []


# ============================================================
# Run experiment
# ============================================================

for snr_db in SNR_VALUES:

    trial_errors_source_1 = []
    trial_errors_source_2 = []

    for trial in range(NUM_TRIALS):

        seed = 1000 + trial

        received_signal = generate_multiple_received_signal(
            num_elements=NUM_ELEMENTS,
            num_samples=NUM_SNAPSHOTS,
            spacing=SPACING,
            wavelength=WAVELENGTH,
            angles_deg=TRUE_DOAS,
            snr_db=snr_db,
            seed=seed,
        )

        covariance_matrix = calculate_covariance_matrix(
            received_signal
        )

        eigenvalues, eigenvectors = (
            calculate_eigendecomposition(
                covariance_matrix
            )
        )

        music_spectrum = calculate_music_spectrum(
            eigenvectors=eigenvectors,
            num_sources=NUM_SOURCES,
            num_elements=NUM_ELEMENTS,
            spacing=SPACING,
            wavelength=WAVELENGTH,
            angle_grid=ANGLE_GRID,
        )

        estimated_doas = estimate_multiple_doa(
            angle_grid=ANGLE_GRID,
            music_spectrum=music_spectrum,
            num_sources=NUM_SOURCES,
        )

        errors = estimated_doas - TRUE_DOAS

        trial_errors_source_1.append(errors[0])
        trial_errors_source_2.append(errors[1])

    # Store RMSE for each source.
    rmse_source_1 = np.sqrt(
        np.mean(np.square(trial_errors_source_1))
    )

    rmse_source_2 = np.sqrt(
        np.mean(np.square(trial_errors_source_2))
    )

    # Combined RMSE.
    combined_rmse = np.sqrt(
    (
        np.mean(np.square(trial_errors_source_1))
        + np.mean(np.square(trial_errors_source_2))
    )
    / 2
)

    source_1_errors.append(rmse_source_1)
    source_2_errors.append(rmse_source_2)
    rmse_results.append(combined_rmse)


# ============================================================
# Print results
# ============================================================

print("=" * 60)
print("             DOA ERROR VS SNR")
print("=" * 60)

print("\nExperiment configuration")
print(f"Number of antennas : {NUM_ELEMENTS}")
print(f"Snapshots          : {NUM_SNAPSHOTS}")
print(f"Number of sources  : {NUM_SOURCES}")
print(f"True DOAs          : {TRUE_DOAS}")
print(f"Trials per SNR     : {NUM_TRIALS}")

print("\nResults")
print(
    f"{'SNR (dB)':>10}"
    f"{'Source 1 RMSE':>20}"
    f"{'Source 2 RMSE':>20}"
    f"{'Combined RMSE':>20}"
)

for snr, rmse1, rmse2, combined in zip(
    SNR_VALUES,
    source_1_errors,
    source_2_errors,
    rmse_results,
):
    print(
        f"{snr:>10.1f}"
        f"{rmse1:>20.3f}"
        f"{rmse2:>20.3f}"
        f"{combined:>20.3f}"
    )


# ============================================================
# Plot RMSE versus SNR
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    SNR_VALUES,
    source_1_errors,
    marker="o",
    label="Source 1 (-20°)",
)

plt.plot(
    SNR_VALUES,
    source_2_errors,
    marker="s",
    label="Source 2 (35°)",
)

plt.plot(
    SNR_VALUES,
    rmse_results,
    marker="^",
    label="Combined RMSE",
)

plt.xlabel("SNR (dB)")
plt.ylabel("DOA RMSE (degrees)")

plt.title("MUSIC DOA Estimation Error versus SNR")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/doa_error_vs_snr.png",
    dpi=300,
)

plt.show()