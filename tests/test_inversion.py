
import numpy as np

from qsm_vascular.phantoms import create_cylindrical_phantom
from qsm_vascular.dipole import dipole_forward
from qsm_vascular.inversion import tikhonov_inversion


def test_tikhonov_parallel_cylinder():
    chi_true = create_cylindrical_phantom(
        N=64,
        voxel_size=1.0,
        radius=5.0,
        chi_vessel=0.4,
    )

    b = dipole_forward(chi_true, voxel_size=1.0)

    lambda_reg = 0.01

    chi_recon = tikhonov_inversion(
        b,
        lambda_reg=lambda_reg,
        voxel_size=1.0,
    )

    chi_expected = (chi_true - chi_true.mean()) / (
        1 + 9 * lambda_reg
    )

    np.testing.assert_allclose(
        chi_recon,
        chi_expected,
        atol=1e-12,
        rtol=0,
    )
