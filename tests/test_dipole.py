
import numpy as np

from qsm_vascular.dipole import (
    create_dipole_kernel,
    dipole_forward,
)
from qsm_vascular.phantoms import create_cylindrical_phantom


def test_dipole_kernel_origin():
    D = create_dipole_kernel(shape=(64, 64, 64), voxel_size=1.0)

    assert D[0, 0, 0] == 0.0


def test_forward_output_shape():
    chi = create_cylindrical_phantom(N=64)

    b = dipole_forward(chi, voxel_size=1.0)

    assert b.shape == chi.shape


def test_forward_parallel_cylinder():
    chi = create_cylindrical_phantom(
        N=64,
        voxel_size=1.0,
        radius=5.0,
        chi_vessel=0.4,
    )

    b = dipole_forward(chi, voxel_size=1.0)

    b_expected = (chi - chi.mean()) / 3

    np.testing.assert_allclose(
        b,
        b_expected,
        atol=1e-12,
        rtol=0,
    )

def test_non_cubic_kernel():
    D = create_dipole_kernel(
        shape=(64, 48, 32),
        voxel_size=1.0
    )

    assert D.shape == (64, 48, 32)
    assert D[0, 0, 0] == 0.0