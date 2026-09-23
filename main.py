import numpy as np

from src.signal_generation import generate_received_signal


NUM_ELEMENTS = 8
NUM_SAMPLES = 1000

WAVELENGTH = 1.0
SPACING = WAVELENGTH / 2

DOA = 30.0


received_signal = generate_received_signal(
    num_elements=NUM_ELEMENTS,
    num_samples=NUM_SAMPLES,
    spacing=SPACING,
    wavelength=WAVELENGTH,
    angle_deg=DOA
)


print("Received signal shape:")
print(received_signal.shape)

print("\nFirst sample from each antenna:")
print(received_signal[:, 0])

print("\nMagnitude of first sample:")
print(np.abs(received_signal[:, 0]))

print("\nPhase of first sample:")
print(np.angle(received_signal[:, 0], deg=True))