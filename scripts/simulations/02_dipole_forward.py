
import numpy as np
import matplotlib.pyplot as plt

from qsm_vascular.dipole import create_dipole_kernel

# Simulation parameters
N = 64
voxel_size = 1.0  # mm

# Generate the dipole kernel using our reusable function
D = create_dipole_kernel(N=N, voxel_size=voxel_size)

# Extract the kx-kz plane (ky = 0)
D_slice = D[:, 0, :]

# Shift zero frequency to the center for visualization
D_slice_shifted = np.fft.fftshift(D_slice)

plt.figure(figsize=(6, 5))

plt.imshow(
    D_slice_shifted,
    cmap="seismic",
    vmin=-2/3,
    vmax=2/3,
    origin="lower"
)

plt.colorbar(label="Dipole kernel D")
plt.xlabel("kz frequency index")
plt.ylabel("kx frequency index")
plt.title("Dipole kernel in the kx-kz plane (ky = 0)")

plt.tight_layout()
plt.show()
