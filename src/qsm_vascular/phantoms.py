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



def create_tilted_cylindrical_phantom(
    N=64,
    voxel_size=1.0,
    radius=5.0,
    chi_vessel=0.4,
    angle_deg=30.0,
):
    # Convert the angle to radians
    theta = np.deg2rad(angle_deg)

    # Create spatial coordinates
    coords = (np.arange(N) - N // 2) * voxel_size

    X, Y, Z = np.meshgrid(
        coords, coords, coords,
        indexing="ij"
    )

    # Squared distance from the tilted cylinder axis
    d_perp_squared = (
    X**2 + Y**2 + Z**2
    - (X * np.sin(theta) + Z * np.cos(theta))**2
    )

    # Allow for floating-point errors at the cylinder boundary
    tolerance = 1e-12 * radius**2

    vessel_mask = d_perp_squared <= radius**2 + tolerance

    # Assign susceptibility values
    chi = np.where(vessel_mask, chi_vessel, 0.0)

    return chi
