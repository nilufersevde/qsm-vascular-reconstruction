import numpy as np


def create_dipole_kernel(N=64, voxel_size=1.0):
    """Create a 3D Fourier-space dipole kernel for B0 along z."""

    kx = np.fft.fftfreq(N, d=voxel_size)
    ky = np.fft.fftfreq(N, d=voxel_size)
    kz = np.fft.fftfreq(N, d=voxel_size)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

    K2 = KX**2 + KY**2 + KZ**2

    D = np.zeros_like(K2)

    mask = K2 != 0
    D[mask] = 1/3 - KZ[mask]**2 / K2[mask]

    return D