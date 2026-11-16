
# Dipole Forward Simulation — Implementation and Validation Notes

## 1. Fourier-domain forward model

The local magnetic field perturbation is related to the magnetic susceptibility distribution through

$$
\hat{b}(\mathbf{k}) = D(\mathbf{k})\hat{\chi}(\mathbf{k})
$$

where:

- $\chi(\mathbf{r})$ is the spatial susceptibility distribution, expressed in ppm.
- $\mathbf{r}=(x,y,z)$ is a position in physical space.
- $\mathbf{k}=(k_x,k_y,k_z)$ is a spatial-frequency vector.
- $\hat{\chi}(\mathbf{k})$ is the Fourier transform of susceptibility.
- $D(\mathbf{k})$ is the dipole kernel.
- $b(\mathbf{r})=\Delta B(\mathbf{r})/B_0$ is the normalized local magnetic field perturbation, expressed in ppm when using ppm susceptibility values.

For $B_0$ aligned with the z-axis, the dipole kernel is

$$
D(\mathbf{k}) =
\frac{1}{3}
-
\frac{k_z^2}{k_x^2+k_y^2+k_z^2}
$$

for nonzero frequency vectors.

## 2. Zero spatial frequency and mean susceptibility

The zero-frequency component represents the spatially constant component of susceptibility.

Any susceptibility distribution can be decomposed as

$$
\chi(\mathbf{r})
=
\overline{\chi}
+
\left(\chi(\mathbf{r})-\overline{\chi}\right)
$$

where $\overline{\chi}$ is the mean susceptibility across the simulated volume.

The first term is constant everywhere. The second term describes variations around the mean and has zero spatial average.

With NumPy's default FFT convention,

$$
\hat{\chi}(\mathbf{0})
=
N_xN_yN_z\overline{\chi}
$$

where $N_x$, $N_y$, and $N_z$ are the number of voxels along the three spatial axes.

## 3. Dipole kernel at the origin

The dipole-kernel formula is undefined at $\mathbf{k}=\mathbf{0}$ because both the numerator and denominator of its second term are zero.

In our discrete QSM forward model, we define

$$
D(\mathbf{0})=0
$$

Consequently, the constant susceptibility component contributes nothing to the simulated local field, and the simulated field has zero spatial mean.

This convention does not imply that the original susceptibility distribution has zero mean.

It is also distinct from choosing an anatomical susceptibility reference, such as cerebrospinal fluid (CSF).

## 4. Special case: Cylinder parallel to B0

Our initial phantom is a straight cylindrical vessel extending through the entire z-dimension.

Because susceptibility does not vary along z, its Fourier coefficients are zero whenever $k_z\neq 0$.

For nonzero frequency vectors with $k_z=0$,

$$
D(k_x,k_y,0)=\frac{1}{3}
$$

The constant susceptibility component is removed because $D(\mathbf{0})=0$.

Therefore, the simulated local field satisfies

$$
\boxed{
b(\mathbf{r})
=
\frac{1}{3}
\left(
\chi(\mathbf{r})-\overline{\chi}
\right)
}
$$

This relationship is specific to our z-invariant phantom and periodic FFT forward model. It does not generally hold for arbitrary vessel orientations or finite-length cylinders.

## 5. Numerical validation

### Simulation parameters

| Parameter | Value |
|---|---|
| Volume size | 64 × 64 × 64 voxels |
| Voxel size | 1 × 1 × 1 mm³ |
| Vessel radius | 5 mm |
| Vessel susceptibility | +0.4 ppm |
| Background susceptibility | 0 ppm |
| Vessel orientation | Parallel to $B_0$ |
| Main field direction | z-axis |
| Noise | None |

### Numerical forward simulation

The local field was calculated using the following operations:

```python
chi_hat = np.fft.fftn(chi)
b_hat = D * chi_hat
b = np.fft.ifftn(b_hat).real
```

### Analytical validation

For the z-invariant cylindrical phantom, the expected local field is

$$
b_{\mathrm{expected}}(\mathbf{r})
=
\frac{\chi(\mathbf{r})-\overline{\chi}}{3}
$$

The maximum absolute difference was calculated as

$$
E_{\max}
=
\max_{\mathbf{r}}
\left|
b(\mathbf{r})-b_{\mathrm{expected}}(\mathbf{r})
\right|
$$

The observed result was

$$
E_{\max}
=
5.551115123125783\times10^{-17}\ \mathrm{ppm}
$$

The agreement is consistent with floating-point precision and supports the numerical correctness of the implementation for this particular test case.

It does not establish physical accuracy for arbitrary geometries or realistic acquisition conditions.

## 6. Assumptions and limitations

- The simulation uses an idealized cylindrical vessel.
- The cylinder extends through the full z-dimension.
- The main magnetic field $B_0$ is aligned with the z-axis.
- The susceptibility distribution is defined on an isotropic voxel grid.
- The FFT-based implementation assumes periodic boundary conditions.
- The zero-frequency dipole-kernel value is defined as zero.
- Noise, phase acquisition, and background-field removal are not modeled.
- The analytical validation applies only to the present special-case geometry.

## 7. Points to revisit

- Physical interpretation of the zero-frequency convention and boundary conditions.
- Susceptibility referencing in in-vivo QSM.
- Magic-angle nulls and the ill-posed inverse problem.
- Finite-length vessels and different orientations relative to $B_0$.
- Partial-volume effects and acquisition resolution.
- Literature references supporting the forward-model assumptions.
