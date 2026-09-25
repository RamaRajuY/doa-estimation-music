import numpy as np

from src.signal_generation import (
    generate_multiple_received_signal
)
from src.covariance import calculate_covariance_matrix
from src.esprit import estimate_esprit_doa


# ============================================================
# Experiment configuration
# ============================================================

NUM_ELEMENTS = 8
NUM_SNAPSHOTS = 1000

WAVELENGTH = 1.0
SPACING = WAVELENGTH / 2

TRUE_DOAS = np.array([
    -20.0,
    35.0,
])

NUM_SOURCES = len(TRUE_DOAS)

SNR_DB = 10.0

RANDOM_SEED = 42


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
# Calculate covariance matrix
# ============================================================

covariance_matrix = calculate_covariance_matrix(
    received_signal
)


# ============================================================
# ESPRIT DOA estimation
# ============================================================

estimated_doas = estimate_esprit_doa(
    covariance_matrix=covariance_matrix,
    num_sources=NUM_SOURCES,
    spacing=SPACING,
    wavelength=WAVELENGTH,
)


# ============================================================
# Calculate errors
# ============================================================

errors = estimated_doas - TRUE_DOAS


# ============================================================
# Display results
# ============================================================

print("=" * 60)
print("                 ESPRIT DOA ESTIMATION")
print("=" * 60)

print("\nExperiment configuration")
print(f"Number of antennas : {NUM_ELEMENTS}")
print(f"Element spacing    : {SPACING:.2f} λ")
print(f"Number of sources  : {NUM_SOURCES}")
print(f"True DOAs          : {TRUE_DOAS}")
print(f"SNR                : {SNR_DB:.1f} dB")
print(f"Snapshots          : {NUM_SNAPSHOTS}")

print("\nResults")

for i, (true_doa, estimated_doa, error) in enumerate(
    zip(TRUE_DOAS, estimated_doas, errors),
    start=1,
):
    print(
        f"Source {i}: "
        f"True = {true_doa:.1f}°, "
        f"Estimated = {estimated_doa:.2f}°, "
        f"Error = {error:.2f}°"
    )
    