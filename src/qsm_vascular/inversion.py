
import numpy as np

from qsm_vascular.dipole import create_dipole_kernel


def tikhonov_inversion(b, lambda_reg, voxel_size=1.0):
    """
    Reconstruct susceptibility using zero-order Tikhonov regularization.

    Parameters
    ----------
    b : np.ndarray
        3D normalized local magnetic field in ppm.
    lambda_reg : float
        Positive regularization parameter.
    voxel_size : float
        Isotropic voxel size in mm.

    Returns
    -------
    chi_recon : np.ndarray
        Reconstructed mean-zero susceptibility map in ppm.
    """

    if lambda_reg <= 0:
        raise ValueError("lambda_reg must be positive.")

    D = create_dipole_kernel(
    shape=b.shape,
    voxel_size=voxel_size
    )

    b_hat = np.fft.fftn(b)

    filter_tikhonov = D / (D**2 + lambda_reg)

    chi_hat = filter_tikhonov * b_hat

    chi_recon = np.fft.ifftn(chi_hat).real

    return chi_recon
