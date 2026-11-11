import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
N = 64
voxel_size = 1.0  # mm

# Initialize susceptibility volume
chi = np.zeros((N, N, N))

# Vessel parameters
radius = 5.0  # mm
chi_vessel = 0.4  # ppm

# Center of the volume
center = N // 2

# Create the x and y coordinates
x = (np.arange(N) - center) * voxel_size
y = (np.arange(N) - center) * voxel_size  #np.arange(N) creates an array of voxel indices

X, Y = np.meshgrid(x, y, indexing="ij")
circle_mask = X**2 + Y**2 <= radius**2

chi[:, :, :] = np.where(circle_mask[:, :, None], chi_vessel, 0.0)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# XY plane: perpendicular to the vessel
axes[0].imshow(chi[:, :, N // 2], cmap="gray", vmin=0, vmax=0.4)
axes[0].set_title("XY cross-section")
axes[0].set_xlabel("Y voxel index")
axes[0].set_ylabel("X voxel index")

# XZ plane: parallel to the vessel
axes[1].imshow(chi[:, N // 2, :], cmap="gray", vmin=0, vmax=0.4)
axes[1].set_title("XZ cross-section")
axes[1].set_xlabel("Z voxel index")
axes[1].set_ylabel("X voxel index")

plt.tight_layout()
plt.show()

print("Maximum susceptibility:", chi.max())
print("Number of vessel voxels:", np.count_nonzero(chi))