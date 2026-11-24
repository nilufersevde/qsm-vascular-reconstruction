
import numpy as np
import matplotlib.pyplot as plt

from qsm_vascular.phantoms import create_cylindrical_phantom
from qsm_vascular.dipole import dipole_forward
from qsm_vascular.inversion import tikhonov_inversion


# 1. Simulation parameters
N = 64
voxel_size = 1.0
radius = 5.0
chi_vessel = 0.4
lambda_reg = 0.1

# 2. Generate ground-truth susceptibility
chi_true = create_cylindrical_phantom(
    N=N,
    voxel_size=voxel_size,
    radius=radius,
    chi_vessel=chi_vessel,
)


# 3. Simulate magnetic field
b = dipole_forward(
    chi_true,
    voxel_size=voxel_size,
)


# 4. Reconstruct susceptibility
chi_recon = tikhonov_inversion(
    b,
    lambda_reg=lambda_reg,
    voxel_size=voxel_size,
)


# 5. Reference-aware comparison
vessel_mask = chi_true > 0
background_mask = ~vessel_mask

true_contrast = (
    chi_true[vessel_mask].mean()
    - chi_true[background_mask].mean()
)

recon_contrast = (
    chi_recon[vessel_mask].mean()
    - chi_recon[background_mask].mean()
)

bias = recon_contrast - true_contrast

relative_bias = 100 * bias / true_contrast

print(f"True vessel contrast: {true_contrast:.6f} ppm")
print(f"Reconstructed contrast: {recon_contrast:.6f} ppm")
print(f"Contrast bias: {bias:.6f} ppm")
print(f"Relative bias: {relative_bias:.2f}%")

print("\nRegularization sweep")
print("Lambda   Reconstructed contrast (ppm)")


# 6. Regularization sweep
for lam in [0.001, 0.01, 0.05, 0.1, 0.2]:
    chi_result = tikhonov_inversion(
        b,
        lambda_reg=lam,
        voxel_size=voxel_size,
    )

    contrast = (
        chi_result[vessel_mask].mean()
        - chi_result[background_mask].mean()
    )

    print(f"{lam:.3f}    {contrast:.6f}")

# 7. Visualize central XY slices
z_center = N // 2

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

images = [
    chi_true[:, :, z_center],
    b[:, :, z_center],
    chi_recon[:, :, z_center],
]

titles = [
    "Ground-truth susceptibility",
    "Simulated local field",
    "Tikhonov reconstruction",
]

for ax, image, title in zip(axes, images, titles):
    im = ax.imshow(
        image,
        cmap="seismic",
        origin="lower",
        vmin=-0.4,
        vmax=0.4,
    )

    ax.set_title(title)
    ax.set_xlabel("Y voxel index")
    ax.set_ylabel("X voxel index")

    fig.colorbar(im, ax=ax, label="ppm")

plt.tight_layout()
plt.show()
