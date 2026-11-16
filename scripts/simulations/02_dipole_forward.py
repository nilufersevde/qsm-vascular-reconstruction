
import numpy as np
import matplotlib.pyplot as plt

from qsm_vascular.dipole import dipole_forward
from qsm_vascular.phantoms import create_cylindrical_phantom

# Simulation parameters
N = 64
voxel_size = 1.0  # mm

# Generate the susceptibility phantom
chi = create_cylindrical_phantom(
    N=N,
    voxel_size=voxel_size,
    radius=5.0,
    chi_vessel=0.4
)

b = dipole_forward(chi, voxel_size=voxel_size)

# Expected field for a z-aligned cylindrical vessel
b_expected = (chi - chi.mean()) / 3
# Compare numerical and expected fields
error = np.abs(b - b_expected)

print("Maximum absolute error:", error.max())


# Visualize susceptibility and simulated local field
z_center = N // 2

chi_slice = chi[:, :, z_center]
b_slice = b[:, :, z_center]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Susceptibility phantom
im1 = axes[0].imshow(
    chi_slice,
    cmap="gray",
    origin="lower",
    vmin=0,
    vmax=0.4
)

axes[0].set_title("Susceptibility phantom")
axes[0].set_xlabel("Y voxel index")
axes[0].set_ylabel("X voxel index")
fig.colorbar(im1, ax=axes[0], label="Susceptibility (ppm)")

# Simulated magnetic field
im2 = axes[1].imshow(
    b_slice,
    cmap="seismic",
    origin="lower",
    vmin=-0.14,
    vmax=0.14
)

axes[1].set_title("Simulated local field")
axes[1].set_xlabel("Y voxel index")
axes[1].set_ylabel("X voxel index")
fig.colorbar(im2, ax=axes[1], label="Field shift (ppm)")

plt.tight_layout()
plt.show()
