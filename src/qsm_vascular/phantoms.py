import numpy as np


def create_cylindrical_phantom(
    N=64,
    voxel_size=1.0,
    radius=5.0,
    chi_vessel=0.4
):
    """Create a cylindrical susceptibility phantom along the z-axis."""

    chi = np.zeros((N, N, N))

    center = N // 2

    x = (np.arange(N) - center) * voxel_size
    y = (np.arange(N) - center) * voxel_size

    X, Y = np.meshgrid(x, y, indexing="ij")

    circle_mask = X**2 + Y**2 <= radius**2

    chi[:, :, :] = np.where(
        circle_mask[:, :, None],
        chi_vessel,
        0.0
    )

    return chi