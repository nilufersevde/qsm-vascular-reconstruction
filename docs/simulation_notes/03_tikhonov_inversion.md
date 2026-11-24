# Simulation 03 — Zero-Order Tikhonov Inversion

## 1. Objective

Reconstruct a known cylindrical susceptibility phantom from its simulated local magnetic field using **zero-order Tikhonov regularization**. This is a simple, noise-free baseline before experiments with tilted vessels, smaller diameters, and noise.

Our pipeline is:

1. Generate ground-truth susceptibility, `chi_true`.
2. Compute the normalized local field, `b = dipole_forward(chi_true)`.
3. Reconstruct susceptibility, `chi_recon = tikhonov_inversion(b, lambda_reg)`.
4. Compare the reconstructed vessel-to-background contrast with its known true value.

## 2. Simulation parameters

| Parameter | Value |
| --- | --- |
| Array shape | 64 × 64 × 64 voxels |
| Voxel size | 1 mm isotropic |
| Vessel radius | 5 mm |
| Vessel axis | z-axis, parallel to B0 |
| Vessel susceptibility | +0.4 ppm |
| Background susceptibility | 0 ppm |
| Simulated noise | None |
| Main illustrated regularization value | `lambda_reg = 0.1` |

The vessel susceptibility value is an illustrative simulation setting, not a patient measurement.

## 3. Forward model and inversion problem

The Fourier-domain dipole forward model is

$$
\hat{b}(\mathbf{k}) = D(\mathbf{k})\hat{\chi}(\mathbf{k}).
$$

Here:

- $\chi(\mathbf{r})$ is the susceptibility map in spatial coordinates $\mathbf{r}=(x,y,z)$.
- $\hat{\chi}(\mathbf{k})$ is its Fourier transform, with spatial-frequency coordinates $\mathbf{k}=(k_x,k_y,k_z)$.
- $D(\mathbf{k})$ is the dipole kernel for $B_0$ aligned with z.
- $b(\mathbf{r})$ is the normalized local magnetic field perturbation, expressed in ppm when the susceptibility values are expressed in ppm.

For nonzero frequencies,

$$
D(\mathbf{k})=\frac{1}{3}-\frac{k_z^2}{k_x^2+k_y^2+k_z^2}.
$$

Our discrete implementation sets $D(\mathbf{0})=0$. Direct inversion by dividing by $D$ is unstable near the kernel's zeros.

## 4. Zero-order Tikhonov regularization

Instead of direct inversion, we minimize

$$
\chi_\lambda = \underset{\chi}{\operatorname{arg\,min}}\left(\|A\chi-b\|_2^2+\lambda\|\chi\|_2^2\right),
$$

where $A$ is the dipole forward operator, $b$ is the given field, and $\lambda>0$ controls regularization.

- **Data fidelity:** $\|A\chi-b\|_2^2$ penalizes disagreement between predicted and observed fields.
- **Regularization:** $\lambda\|\chi\|_2^2$ penalizes large susceptibility values.
- **Zero-order** means no spatial derivatives are used in the regularization penalty.

For this Fourier-diagonal forward operator, the solution is

$$
\boxed{\hat{\chi}_\lambda(\mathbf{k})=\frac{D(\mathbf{k})}{D(\mathbf{k})^2+\lambda}\hat{b}(\mathbf{k})}.
$$

The denominator stays positive for $\lambda>0$, avoiding division by zero.

## 5. Why reconstructed susceptibility is attenuated

In our noise-free simulation,

$$
\hat{b}(\mathbf{k})=D(\mathbf{k})\hat{\chi}_{\mathrm{true}}(\mathbf{k}).
$$

Substituting this into the Tikhonov solution gives

$$
\hat{\chi}_\lambda(\mathbf{k}) = \underbrace{\frac{D(\mathbf{k})^2}{D(\mathbf{k})^2+\lambda}}_{H_\lambda(\mathbf{k})}\hat{\chi}_{\mathrm{true}}(\mathbf{k}).
$$

$H_\lambda$ is the **retention or attenuation factor** for a susceptibility Fourier component. For $\lambda>0$, it lies between 0 and 1. Frequencies at which $D$ is small are suppressed more strongly.

### Special case: our z-aligned cylinder

The idealized cylinder is unchanged along z. Its nonzero susceptibility Fourier coefficients therefore have $k_z=0$. At those frequencies, except at $\mathbf{k}=\mathbf{0}$,

$$
D=\frac{1}{3}.
$$

Hence

$$
H_\lambda=\frac{1/9}{1/9+\lambda}=\boxed{\frac{1}{1+9\lambda}}.
$$

The zero-frequency component is removed because $D(\mathbf{0})=0$. Transforming back to spatial coordinates therefore gives

$$
\boxed{\chi_\lambda(\mathbf{r})=\frac{\chi_{\mathrm{true}}(\mathbf{r})-\overline{\chi}_{\mathrm{true}}}{1+9\lambda}}.
$$

This formula is specific to the z-invariant cylinder and the chosen discrete periodic FFT model. It is not a general Tikhonov reconstruction formula for tilted or finite vessels.

## 6. Reference-aware evaluation

The dipole model eliminates the constant (zero-frequency) susceptibility component. Consequently, we compare **vessel-to-background contrast**, not the raw reconstructed vessel value, with ground truth.

We identify voxels using the **ground-truth** image:

```python
vessel_mask = chi_true > 0
background_mask = ~vessel_mask
```

The contrast is

$$
\Delta\chi_{\mathrm{recon}}=\operatorname{mean}(\chi_{\mathrm{recon}}\text{ in vessel})-\operatorname{mean}(\chi_{\mathrm{recon}}\text{ in background}).
$$

The true contrast is $0.4-0=0.4$ ppm. We compute

$$
\mathrm{Bias}=\Delta\chi_{\mathrm{recon}}-\Delta\chi_{\mathrm{true}},
$$

$$
\mathrm{Relative\ bias}=100\times\frac{\mathrm{Bias}}{\Delta\chi_{\mathrm{true}}}\%.
$$

For our z-invariant phantom, the predicted reconstructed contrast is

$$
\boxed{\Delta\chi_{\mathrm{recon}}=\frac{0.4}{1+9\lambda}\ \mathrm{ppm}}.
$$

## 7. Observed numerical results

The simulation script was run with `lambda_reg = 0.1` and printed:

```text
True vessel contrast: 0.400000 ppm
Reconstructed contrast: 0.210526 ppm
Contrast bias: -0.189474 ppm
Relative bias: -47.37%
```

The same run performed a regularization sweep:

| Lambda | Reconstructed vessel-to-background contrast (ppm) |
| ---: | ---: |
| 0.001 | 0.396432 |
| 0.010 | 0.366972 |
| 0.050 | 0.275862 |
| 0.100 | 0.210526 |
| 0.200 | 0.142857 |

The results agree with $0.4/(1+9\lambda)$. Increasing $\lambda$ decreases the reconstructed contrast in this noise-free special case. The reconstruction retains the circular vessel shape because all its nonzero-frequency coefficients are attenuated equally.

### Automated validation

At the most recent reported test run, all five tests passed, including the Tikhonov analytical check and the non-cubic kernel test. This is a reported result from the local development environment, not an independent run recorded in this document.

## 8. Relevant project files

- `src/qsm_vascular/phantoms.py` — ground-truth cylindrical phantom.
- `src/qsm_vascular/dipole.py` — Fourier dipole kernel and forward model.
- `src/qsm_vascular/inversion.py` — zero-order Tikhonov inversion.
- `scripts/simulations/03_tikhonov_inversion.py` — reconstruction, contrast, sweep, and plotting.
- `tests/test_inversion.py` — analytical test of Tikhonov inversion.

## 9. Limitations

- The vessel is a large, idealized cylinder parallel to $B_0$.
- The susceptibility is uniform inside the vessel and zero outside.
- There is no noise or simulated GRE acquisition.
- Forward simulation and inversion use the same discrete dipole kernel, which makes this an **inverse-crime baseline** rather than an independent physical validation.
- The nonzero susceptibility frequencies of this phantom do not probe the dipole kernel's magic-angle zeros.
- Reference choices and vessel segmentation will become more challenging with realistic data.

## 10. Next step

Implement a tilted-cylinder phantom in `src/qsm_vascular/phantoms.py`, test its geometry, and validate its simulated field with attention to padding, boundaries, and resolution. After that, study orientation-dependent Tikhonov reconstruction error, including angles around the idealized 35.3-degree spectral threshold.
