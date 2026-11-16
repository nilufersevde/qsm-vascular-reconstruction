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



def dipole_forward(chi, voxel_size=1.0):
    """
    Simulate the normalized local magnetic field from susceptibility.

    Parameters
    ----------
    chi : np.ndarray
        3D susceptibility distribution in ppm.
    voxel_size : float
        Isotropic voxel size in mm.

    Returns
    -------
    b : np.ndarray
        Simulated normalized local field in ppm.
    """

    N = chi.shape[0]

    D = create_dipole_kernel(
        N=N,
        voxel_size=voxel_size
    )

    chi_hat = np.fft.fftn(chi)

    b_hat = D * chi_hat

    b = np.fft.ifftn(b_hat).real

    return b
