import numpy as np

from qsm_vascular.phantoms import (
    create_cylindrical_phantom,
    create_tilted_cylindrical_phantom,
)


def test_tilted_cylinder_zero_angle():
    chi_original = create_cylindrical_phantom()

    chi_tilted = create_tilted_cylindrical_phantom(
        angle_deg=0
    )

    np.testing.assert_array_equal(
        chi_original,
        chi_tilted
    )


def test_tilted_cylinder_ninety_degrees():
    chi_original = create_cylindrical_phantom()

    chi_expected = np.swapaxes(chi_original, 0, 2)

    chi_tilted = create_tilted_cylindrical_phantom(
        angle_deg=90
    )

    np.testing.assert_array_equal(
        chi_expected,
        chi_tilted
    )