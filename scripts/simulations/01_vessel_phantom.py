import numpy as np
import matplotlib.pyplot as plt

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

print("Volume shape:", chi.shape)
print("Maximum susceptibility:", chi.max())
print("Number of vessel voxels:", np.count_nonzero(chi))