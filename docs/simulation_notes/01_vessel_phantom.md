# Simulation 01 — Cylindrical Vessel Susceptibility Phantom

## 1. Objective

The objective of this simulation is to construct a simple three-dimensional susceptibility phantom representing a straight cerebral vessel.

The phantom provides a known ground-truth susceptibility distribution for evaluating the dipole forward model and subsequent QSM reconstruction algorithms.

This initial phantom is deliberately idealized. It allows us to study the mathematical behavior of the forward and inverse problems before introducing more realistic vessel geometries, noise, and partial-volume effects.

## 2. Simulation parameters

| Parameter | Value |
|---|---|
| Volume dimensions | 64 × 64 × 64 voxels |
| Voxel size | 1 × 1 × 1 mm³ |
| Physical field of view | 64 × 64 × 64 mm³ |
| Vessel radius | 5 mm |
| Vessel diameter | 10 mm |
| Vessel susceptibility | +0.4 ppm |
| Background susceptibility | 0 ppm |
| Vessel orientation | Along the z-axis |
| Main magnetic field direction | z-axis |
| Noise | None |

The susceptibility value of +0.4 ppm is an illustrative simulation parameter, not a measured value from a specific subject or vessel.

## 3. Spatial coordinate system

The susceptibility phantom is represented by a three-dimensional NumPy array:

```python
chi = np.zeros((N, N, N))
```

Here, `chi[i, j, k]` represents the susceptibility assigned to a voxel at array indices `(i, j, k)`.

The volume has 64 voxels along each axis. Spatial coordinates are constructed using:

```python
center = N // 2

x = (np.arange(N) - center) * voxel_size
y = (np.arange(N) - center) * voxel_size
```

For `N = 64` and `voxel_size = 1.0`, the coordinates along each axis range from −32 mm to +31 mm.

The center of the cylindrical cross-section is located at $x = 0$ and $y = 0$.

The coordinate grids are generated using:

```python
X, Y = np.meshgrid(x, y, indexing="ij")
```

The `indexing="ij"` argument ensures that array dimensions correspond directly to the x and y coordinate indices.

## 4. Mathematical definition of the phantom

The vessel is modeled as a cylinder with a circular cross-section. Its susceptibility distribution is defined as

$$
\chi(x,y,z) =
\begin{cases}
\chi_{\mathrm{vessel}}, & x^2+y^2 \leq R^2, \\
\chi_{\mathrm{background}}, & x^2+y^2 > R^2.
\end{cases}
$$

where:

- $x,y,z$ are spatial coordinates in millimeters.
- $R$ is the vessel radius in millimeters.
- $\chi_{\mathrm{vessel}} = 0.4\ \mathrm{ppm}$.
- $\chi_{\mathrm{background}} = 0\ \mathrm{ppm}$.
- $\chi(x,y,z)$ is the susceptibility at a given position.

For our simulation,

$$
R = 5\ \mathrm{mm}
$$

and the vessel mask satisfies

$$
x^2+y^2 \leq 25\ \mathrm{mm}^2.
$$

Notice that the condition does not depend on $z$. Therefore, the vessel extends through the entire simulated volume along the z-axis.

## 5. Python implementation

The reusable function is located in `src/qsm_vascular/phantoms.py`.

```python
import numpy as np


def create_cylindrical_phantom(
    N=64,
    voxel_size=1.0,
    radius=5.0,
    chi_vessel=0.4,
):
    chi = np.zeros((N, N, N))

    center = N // 2

    x = (np.arange(N) - center) * voxel_size
    y = (np.arange(N) - center) * voxel_size

    X, Y = np.meshgrid(x, y, indexing="ij")

    circle_mask = X**2 + Y**2 <= radius**2

    chi[:, :, :] = np.where(
        circle_mask[:, :, None],
        chi_vessel,
        0.0,
    )

    return chi
```

The function returns a three-dimensional susceptibility array.

### 5.1 Circular vessel mask

```python
circle_mask = X**2 + Y**2 <= radius**2
```

This creates a two-dimensional Boolean mask. Each position is assigned:

- `True` if it lies inside or on the circular boundary.
- `False` if it lies outside the circle.

The circle is determined by

$$
x^2+y^2\leq R^2.
$$

### 5.2 Extending the mask along z

The circular mask initially has shape `(64, 64)`.

The expression

```python
circle_mask[:, :, None]
```

adds a third dimension, resulting in shape `(64, 64, 1)`.

NumPy broadcasting allows the mask to be applied across all 64 z-slices of the susceptibility volume. Consequently, the same circular cross-section appears at every position along z.

### 5.3 Assigning susceptibility values

```python
np.where(circle_mask[:, :, None], chi_vessel, 0.0)
```

This assigns:

- 0.4 ppm to voxels inside the cylindrical vessel.
- 0 ppm to voxels in the surrounding background.

The resulting array has shape `(64, 64, 64)`.

## 6. Geometrical interpretation

The phantom is a straight cylinder aligned with the z-axis.

### XY cross-section

A slice perpendicular to the vessel axis shows a circular region with susceptibility +0.4 ppm. Outside the circle, susceptibility is zero.

### XZ or YZ cross-section

A longitudinal slice through the vessel center shows a rectangular region. This occurs because the circular vessel mask is repeated along the entire z-dimension.

The vessel therefore has constant cross-sectional geometry throughout the simulated volume.

## 7. Basic numerical checks

```python
print("Shape:", chi.shape)
print("Minimum susceptibility:", chi.min())
print("Maximum susceptibility:", chi.max())
print("Mean susceptibility:", chi.mean())
print("Vessel voxels:", np.count_nonzero(chi))
```

Expected results for the specified parameters:

| Quantity | Value |
|---|---|
| Array shape | (64, 64, 64) |
| Minimum susceptibility | 0 ppm |
| Maximum susceptibility | 0.4 ppm |
| Vessel voxels per XY slice | 81 |
| Total vessel voxels | 5,184 |
| Total volume voxels | 262,144 |
| Mean susceptibility | 0.00791015625 ppm |

The vessel occupies approximately 1.98% of the simulated volume.

The mean susceptibility is

$$
\overline{\chi}
=
\frac{1}{N_xN_yN_z}
\sum_{i,j,k}\chi_{ijk}.
$$

For the present phantom,

$$
\overline{\chi}
=
\frac{5184\times0.4}{64^3}
=
0.00791015625\ \mathrm{ppm}.
$$

This mean value later becomes important when validating the dipole forward simulation.

## 8. Assumptions and limitations

1. The vessel is a perfectly straight cylinder.
2. The vessel has constant radius along its length.
3. Susceptibility inside the vessel is spatially uniform.
4. The surrounding background has zero susceptibility.
5. The vessel is aligned with the main magnetic field $B_0$.
6. The vessel occupies the entire z-dimension.
7. Vessel boundaries are represented by a binary voxel mask.
8. No partial-volume averaging is modeled.
9. No measurement noise is included.
10. No anatomical tissue structure is represented.

The voxelized circular boundary approximates the ideal continuous cylindrical geometry. This phantom is suitable for controlled numerical experiments, but is not a realistic anatomical model of cerebral vasculature.

## 9. Connection to the dipole forward model

The susceptibility phantom is the input to the dipole forward model:

$$
\hat{b}(\mathbf{k})
=
D(\mathbf{k})\hat{\chi}(\mathbf{k}).
$$

Here, $\chi$ is the known susceptibility distribution, while $b$ is the simulated local magnetic field perturbation normalized by $B_0$.

Because the phantom is constant along z, its Fourier transform has nonzero components only at $k_z=0$.

This provides a useful special case for validating the FFT-based dipole forward implementation.

The corresponding validation is documented in `docs/simulation_notes/02_dipole_forward.md`.

## 10. Planned extensions

- Different vessel radii and diameters.
- Different vessel orientations relative to $B_0$.
- Finite-length cylinders.
- Subvoxel vessel positioning.
- Partial-volume effects.
- More realistic susceptibility contrasts.
- Multiple vessels and more complex geometries.

These extensions will help investigate how vessel geometry and image resolution influence QSM reconstruction accuracy.
