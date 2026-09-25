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

FIRST_DOA = 10.0

SECOND_DOAS = np.array([
    30.0,
    20.0,
    15.0,
    12.0,
    11.0,
])

SNR_DB = 10.0

NUM_SOURCES = 2

ANGLE_GRID = np.linspace(-20.0, 40.0, 1201)

RANDOM_SEED = 42


# ============================================================
# Run angular resolution experiments
# ============================================================

print("=" * 65)
print("              MUSIC ANGULAR RESOLUTION")
print("=" * 65)

print("\nExperiment configuration")
print(f"Number of antennas : {NUM_ELEMENTS}")
print(f"Snapshots          : {NUM_SNAPSHOTS}")
print(f"SNR                : {SNR_DB:.1f} dB")
print(f"First DOA          : {FIRST_DOA:.1f}°")

for second_doa in SECOND_DOAS:

    true_doas = np.array([
        FIRST_DOA,
        second_doa,
    ])

    separation = abs(second_doa - FIRST_DOA)

    received_signal = generate_multiple_received_signal(
        num_elements=NUM_ELEMENTS,
        num_samples=NUM_SNAPSHOTS,
        spacing=SPACING,
        wavelength=WAVELENGTH,
        angles_deg=true_doas,
        snr_db=SNR_DB,
        seed=RANDOM_SEED,
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

    # Compare sorted estimated DOAs with sorted true DOAs.
    max_allowed_error_deg = 0.5

    sorted_true_doas = np.sort(true_doas)
    sorted_estimated_doas = np.sort(estimated_doas)

    doa_errors = np.abs(
        sorted_estimated_doas - sorted_true_doas
    )

    max_doa_error = np.max(doa_errors)

    resolved = (
        max_doa_error <= max_allowed_error_deg
    )

    print(
        f"\nTrue DOAs = "
        f"[{FIRST_DOA:.1f}°, {second_doa:.1f}°]"
    )

    print(
        f"Angular separation = "
        f"{separation:.1f}°"
    )

    print(
        f"Estimated DOAs = "
        f"{estimated_doas}"
    )

    print(
        f"Maximum DOA error = "
        f"{max_doa_error:.2f}°"
    )

    print(
        "Resolution status = "
        f"{'RESOLVED' if resolved else 'NOT RESOLVED'}"
    )


# ============================================================
# Plot all angular-resolution cases
# ============================================================

fig, axes = plt.subplots(
    len(SECOND_DOAS),
    1,
    figsize=(10, 14),
)

for ax, second_doa in zip(axes, SECOND_DOAS):

    true_doas = np.array([
        FIRST_DOA,
        second_doa,
    ])

    received_signal = generate_multiple_received_signal(
        num_elements=NUM_ELEMENTS,
        num_samples=NUM_SNAPSHOTS,
        spacing=SPACING,
        wavelength=WAVELENGTH,
        angles_deg=true_doas,
        snr_db=SNR_DB,
        seed=RANDOM_SEED,
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

    music_spectrum_db = 10 * np.log10(
        music_spectrum / np.max(music_spectrum)
    )

    ax.plot(
        ANGLE_GRID,
        music_spectrum_db,
    )

    ax.axvline(
        FIRST_DOA,
        linestyle=":",
    )

    ax.axvline(
        second_doa,
        linestyle=":",
    )

    ax.set_title(
        f"DOA separation = "
        f"{abs(second_doa - FIRST_DOA):.1f}°"
    )

    ax.set_ylabel("MUSIC Spectrum (dB)")
    ax.set_ylim(-40, 5)
    ax.grid(True)


axes[-1].set_xlabel("Angle (degrees)")

plt.tight_layout()

plt.savefig(
    "results/music_angular_resolution_study.png",
    dpi=300,
)

plt.show()