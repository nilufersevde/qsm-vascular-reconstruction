import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

k = np.linspace(-1, 1, 501)

kx, kz = np.meshgrid(k, k)

# Calculate the squared distance from the origin
k_squared = kx**2 + kz**2

# Create an array to store dipole kernel values
D = np.zeros_like(k_squared)

# Identify all points except the origin
nonzero = k_squared > 0

# Evaluate the dipole kernel
D[nonzero] = (
    1/3 - kz[nonzero]**2 / k_squared[nonzero]
)

# Create a figure and a set of axes
fig, ax = plt.subplots(figsize=(7, 6))

# Display the dipole kernel as a colored image
im = ax.imshow(
    D,
    extent=[-1, 1, -1, 1],
    origin="lower",
    cmap="RdBu_r",
    vmin=-2/3,
    vmax=2/3
)

# Draw lines where the dipole kernel equals zero
ax.contour(
    kx,
    kz,
    D,
    levels=[0],
    colors="black",
    linewidths=1.5
)

# Label the spatial-frequency axes
ax.set_xlabel("$k_x$", fontsize=13)
ax.set_ylabel("$k_z$", fontsize=13)

# Add a descriptive title
ax.set_title("Fourier-Space Dipole Kernel ($k_y = 0$)", fontsize=14)

# Add a color bar showing the kernel values
fig.colorbar(im, ax=ax, label="$D(k)$")

# Adjust spacing so labels fit
plt.tight_layout()

output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(
    output_dir / "dipole_kernel_2d.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

ax.set_xlabel("$k_x$", fontsize=13)
ax.set_ylabel("$k_z$", fontsize=13)
ax.set_title("Dipole Kernel ($k_y = 0$)", fontsize=15)

fig.colorbar(im, ax=ax, label="$D(\\mathbf{k})$")