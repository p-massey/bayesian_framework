# Phase Determination for the Three Images of SN H0pe
## Methodology from Chen et al. (2024), *ApJ* 970:102
### DOI: 10.3847/1538-4357/ad50a5 | arXiv: 2403.19029

---

## Overview

SN H0pe is a triply imaged Type Ia supernova at *z* = 1.78, strongly lensed by the galaxy cluster PLCK G165.7+67.0. Its three images are labelled **A** (northernmost, faintest), **B** (brightest), and **C** (intermediate brightness, southernmost). The phase of each image — defined as **rest-frame days after peak B-band brightness** — was determined spectroscopically from a single epoch of JWST/NIRSpec observations acquired on UT 2023 April 22 (Epoch 2, DDT programme 4446; PI: B. Frye). The analysis was performed **blind** to photometric light-curve data and to the lens-model predictions throughout.

---

## 1. Observational Data

### 1.1 Instrument configuration

| Grating/Filter | Wavelength coverage (observed) | Spectral resolution *R* | Rest-frame coverage at *z* = 1.78 |
|---|---|---|---|
| PRISM/CLEAR | 0.6–5.3 μm | ~100 | ~0.22–1.9 μm |
| G140M/F100LP | 0.97–1.84 μm | ~1000 | ~0.35–0.66 μm |
| G235M/F170LP | 1.66–3.17 μm | ~1000 | ~0.57–1.1 μm |

All three gratings targeted all three SN images simultaneously using the Micro-Shutter Assembly (MSA). Each slitlet comprised three open shutters placed end-to-end (0.20″ in the dispersion direction × 0.46″ per shutter in the spatial direction; 0.007″ gap between shutters; total height 1.52″).

The IRS2 detector readout mode was used, which substantially reduces 1/*f* noise.

### 1.2 Archive and pipeline

- Stage 2 calibrated data were retrieved from the Mikulski Archive for Space Telescopes (MAST; doi: 10.17909/rqdx-3976).
- Additional reduction used the JWST pipeline (`jwst` v1.10.2 / context file `jwst_1087.pmap`) to produce two-dimensional (2D) spectral images.
- The pipeline applied a slit-loss throughput correction for each SN image based on its planned position within the MSA shutters.

---

## 2. Spectral Extraction

The extraction procedure must be reproduced in full before phase-fitting can begin; errors in the extraction propagate into the phase uncertainties.

### 2.1 Custom background subtraction

The default JWST in-scene/off-scene nod background subtraction (`BKGSUB` step) was **skipped** because the MSA slitlets are dominated by the extended host-galaxy continuum — there are no blank-sky shutters in this observation.

A custom 2D background model was constructed as follows:

1. For each pixel column (i.e., each wavelength) in the 2D spectrum — with `BKGSUB` turned off — the **minimum flux** was identified across all pixels in that column that lie within MSA shutters at the same spatial position along the slit.
2. Because individual minimum-flux values can be biased by bad pixels (cosmic-ray masks, hot pixels), the resulting 2D minimum-flux array was **smoothed along the dispersion direction** using a median filter with a kernel width of **10 pixels** to produce a smooth 2D background model.
3. This 2D model was subtracted from the original pipeline 2D spectrum.

### 2.2 Simultaneous SN + host-galaxy source extraction

Because the SN point source and the spatially extended host-galaxy nucleus are co-located along the slit, a **simultaneous two-kernel fit** was applied to disentangle the two at each wavelength:

- The SN was modelled as a **Gaussian** kernel (appropriate for an unresolved point source).
- The host galaxy was modelled by whichever of a **Gaussian, Moffat** (Moffat 1969), or **Voigt** (Whiting 1968) profile minimised the least-squares statistic for that observation.
- Both kernels were fit simultaneously to the flux in the 2D spectrum as a function of wavelength. Uncertainties in the kernel functions were **not** propagated at this stage.
- Effective PSFs for the NIRSpec MSA configuration were generated using `webbpsf`.

Once the two-component spatial model was established, **optimal extraction** (Horne 1986) was applied to the background-subtracted 2D spectra using the MOSE (MOS Optimal Spectral Extraction) notebook implementation. Uncertainties on the extracted 1D spectra include contributions from: background subtraction residuals, host-galaxy contamination, bad pixels (cosmic rays), slit loss, and flux calibration.

Saturated pixels and other image artefacts were flagged in the 2D spectra prior to extraction. Individual calibrated 2D exposures were coadded before extraction.

> **Key wavelength range for phase fitting:** The rest-frame range **3600–10000 Å** (shown as green shading in Figure 4 of the paper) was used for all template matching and phase measurements.

---

## 3. SN Classification

Before measuring phases, the SN type must be confirmed because the phase-fitting templates are type-specific.

### 3.1 Template matching with NGSF

The extracted spectra of Images B and C (the two high-SNR images) were cross-matched with the SN Ia template library in the **Next Generation Super-Fit (NGSF)** software package (Goldwasser et al. 2022) over the rest-frame wavelength range 3600–10000 Å.

- Image B most closely matched **SN 2013dy** at phase +6.2 d (reduced χ² = 1.71) and **SN 2012cg** at +9.6 d.
- Image C most closely matched **SN 1994D** at phase +28.6 d (reduced χ² = 0.69) and **SN 2013dy** at +41.3 d.
- All five top matches for both images were Type Ia, subtype **Ia-norm** (or in one case Ia-91T-like).

### 3.2 Feature identification

Two diagnostic features confirming Type Ia classification were measured directly in the G140M spectrum of Image B:

| Feature | Rest-frame λ (observed) | Observer-frame λ | Velocity |
|---|---|---|---|
| Si II λ6355 (blueshifted) | 6117 ± 2 Å | ~17017 Å | −11.23 ± 0.11 × 10³ km s⁻¹ |
| Ca II H & K (blueshifted) | 3790 ± 2 Å | ~10544 Å | −12.54 ± 0.19 × 10³ km s⁻¹ |

The Si II λ6355 absorption equivalent width was measured to be **56 ± 8 Å** in the rest frame from Image B. Both the velocity and EW are consistent with a normal-velocity (Ia-norm) or 1991T-like SN Ia.

---

## 4. Phase Measurement: Three Methods

Three independent methods were applied. The **Hsiao07 template fit** is the primary/final result; SALT3-NIR and SNID serve as validation.

### 4.1 Method 1: Hsiao07 Spectral Template Fitting (primary)

#### 4.1.1 Template

The Hsiao07 templates (Hsiao et al. 2007) are a spectrophotometric time series for SNe Ia, constructed from a library of ~600 spectra of ~100 SNe Ia. They provide a spectral energy distribution as a function of rest-frame phase (days relative to B-band peak) and wavelength. Templates extend to **+80 days** post-maximum. The implementation used is that within the **SNCosmo** package (Barbary et al. 2023).

#### 4.1.2 Simultaneous three-image joint fit

All three SN images were fit **simultaneously** with a shared set of **11 free parameters**:

| Parameter | Symbol | Description |
|---|---|---|
| Overall normalization | α | Amplitude of the Hsiao07 template |
| Colour excess | E(B−V) | Host-galaxy dust reddening |
| Total-to-selective extinction | R_V | Extinction law slope |
| Flux ratio A/B | f_A/f_B | Ratio of SN fluxes between Images A and B |
| Flux ratio A/C | f_A/f_C | Ratio of SN fluxes between Images A and C |
| Phase of Image A | t_A | Rest-frame days after B-band peak (image A) |
| Phase of Image B | t_B | Rest-frame days after B-band peak (image B) |
| Phase of Image C | t_C | Rest-frame days after B-band peak (image C) |
| Background A | bg_A | Free additive offset for Image A spectrum |
| Background B | bg_B | Free additive offset for Image B spectrum |
| Background C | bg_C | Free additive offset for Image C spectrum |

The individual per-image normalizations are derived from the shared α and the flux ratios as: α_A = α, α_B = α/(f_A/f_B), α_C = α/(f_A/f_C).

#### 4.1.3 MCMC sampling

The model was sampled using the **`emcee`** affine-invariant ensemble sampler (Foreman-Mackey et al. 2013). The posterior distributions of all 11 parameters were obtained from the MCMC chain. The reported phases and their **68% confidence intervals** (i.e., 1σ) are derived directly from the marginalised posterior distributions.

As a cross-check, each spectrum was also fit **independently** (4 parameters per image: α, E(B−V), R_V, t_i). Results from independent fits are tabulated alongside the joint-fit results and are consistent.

#### 4.1.4 Pre-systematic results (before uncertainty corrections)

From the joint MCMC fit alone:

| Image | Phase t_i (raw MCMC) |
|---|---|
| B | listed in Table B1 (paper) |
| C | listed in Table B1 (paper) |
| A | listed in Table B1 (paper) |

*(Final values after systematic corrections are reported in Section 7 and given in Section 5 below.)*

---

### 4.2 Method 2: SALT3-NIR Model (validation)

The SALT3-NIR model (Pierel et al. 2022) was developed from ~1200 spectra of 380 SNe Ia (original SALT3; Kenworthy et al. 2021), extended to 2 μm by incorporating 166 additional SNe Ia with NIR data.

Model parameters:

| Parameter | Symbol | Description |
|---|---|---|
| Overall normalization | x_0 | Amplitude |
| Light-curve shape (stretch) | x_1 | Width–luminosity relation |
| Colour | c | Colour–luminosity relation |
| Phase | t_i | Rest-frame days after B-band peak |
| Background | bg_i | Free additive offset |

The same simultaneous and individual fitting strategies as for Hsiao07 were applied, using MCMC.

**Important caveat:** The SALT3-NIR model is only valid for phases ≤ 50 days. For phases beyond 50 days, the likelihood was set equal to the likelihood at 50 days. Because Image A's inferred phase from this model (~42.8 d) approaches this boundary, the joint SALT3-NIR fit may be biased for the co-constrained parameters. Consequently, the SALT3-NIR model is **not used as the primary constraint** and is reported for consistency only.

---

### 4.3 Method 3: SNID (Blondin & Tonry 2007) (validation, continuum-independent)

SNID cross-correlates a **continuum-subtracted** observed spectrum with a library of continuum-subtracted template spectra. This method is insensitive to the SN continuum shape (and thus to dust extinction), probing only spectral line features.

Two template libraries were used:

1. **SNID built-in "template 2.0"** library of observed nearby SN spectra.
2. **Custom SALT3-NIR library**: 1750 synthetic SN Ia spectra spanning 70 phases × 25 combinations of stretch (x_1) and colour (c) parameters.

Results:

| Image | Matching templates (built-in library) | Favoured subtype |
|---|---|---|
| B | 176 (all Ia; 163 Ia-norm) | Ia-norm |
| C | 165 (164 Ia; 107 Ia-norm) | Ia-norm |
| A | None (SNR too low) | Cannot determine |

SNID phases are consistent with Hsiao07 and SALT3-NIR results but with **larger uncertainties** because the continuum information is discarded. SNID therefore decisively confirms the SN Ia classification but is not used as the primary phase constraint.

---

## 5. Uncertainty Budget

The total uncertainty on each phase has **three components** that are combined through a simulation-based approach:

1. **Statistical uncertainty** from the MCMC posterior (template fitting).
2. **Template model systematic** — the bias and scatter introduced by imperfect SN Ia templates.
3. **Microlensing and millilensing systematic** — wavelength-dependent spectral distortions from compact mass structures.

### 5.1 Template simulation uncertainty (Section 6.1)

To calibrate how reliably the Hsiao07 template recovers true SN phases, the same fitting procedure was applied to a set of well-observed nearby SNe Ia from the SNID built-in library.

**Selection of test SNe:**
SNe were required to have at least one spectrum in each of three phase bins that bracket the inferred phases of the three SN H0pe images:

| Bin | Phase range | Corresponding SN H0pe image |
|---|---|---|
| 1 | (−10, +10) days | Image B |
| 2 | (+10, +30) days | Image C |
| 3 | (+30, +60) days | Image A |

This selection yielded **228 sets** of three spectra (one per bin per set).

**Adding noise to match NIRSpec conditions:**
For each template spectrum in each set:
1. The residuals between the actual NIRSpec spectra and the smoothed spectra were computed (smoothed = 120 Å rolling window).
2. A residual value was drawn **randomly** from this empirical residual distribution (assuming the residuals are uncorrelated with wavelength) and added to the template spectrum.
3. The template spectrum was rescaled to match the **signal-to-noise ratio** of the corresponding NIRSpec SN H0pe spectrum.

**Fitting and bias estimation:**
Each set of three noise-added template spectra was then fit simultaneously with the Hsiao07 template (same 11-parameter MCMC procedure as in Section 4.1). The difference between inferred phase and actual (true) phase was recorded for each simulation. The resulting distribution of residuals, computed within a 10-day rolling window, provides the **68% and 95% confidence intervals** on the phase bias as a function of true phase. In particular, it showed that:
- Phase recovery is reliable (small bias) at phases < 30 days.
- At phases > 30 days (relevant for Image A), uncertainties are substantially larger due to both the slow evolution of late-time SN Ia spectral features and the low SNR of Image A's spectrum.

### 5.2 Microlensing and millilensing uncertainties (Section 6.2)

An additional 4000 simulations were performed to account for the effects of compact mass structures along the line of sight.

**Microlensing** (stellar-mass objects):
- Four theoretical models of microlensing caustic magnification maps from Suyu et al. (2020) and Huber et al. (2021) were used.
- For each model, **1000 SN Ia spectra** were simulated at random positions in the source plane.
- At each position, the wavelength-dependent magnification factor — derived by comparing the simulated spectrum at that position with the un-microlensed input spectrum — was computed. This gives spectrally-varying distortions to the SN continuum shape.
- Microlensing introduces systematic magnitude offsets at the level of: σ_m = 0.04 mag (Image A), **0.12 mag (Image B)**, 0.04 mag (Image C) across rest-frame 3600–10000 Å.

**Millilensing** (dark-matter subhaloes):
- Magnification probability distributions were computed for a range of dark-matter subhalo mass functions and substructure fractions consistent with theoretical predictions and observations (Gilman et al. 2020).
- A magnification value was drawn from these distributions and applied as a **grey (wavelength-independent) multiplicative factor** to each simulated SN spectrum (because the SN photosphere at ~10¹⁴–10¹⁵ cm is much smaller than the scale of millilens caustics, which therefore cannot produce chromatic effects).
- Millilensing contributes σ_m = 0.04 mag to all three images.

**Fitting the simulated spectra:**
Gaussian noise was added to each simulated spectrum to match the signal-to-noise ratios of the actual three SN H0pe spectra. Each set of three simulated spectra (one per SN image) was then fit simultaneously with the Hsiao07 template using the same MCMC procedure. The difference between inferred and input parameters gives the systematic offset attributable to microlensing and millilensing for that realisation.

---

## 6. Combining Statistical and Systematic Uncertainties

The final posterior distributions on the phases, time delays, and magnifications are constructed by **modifying each step in the MCMC chain** to fold in both sources of systematic uncertainty.

### 6.1 Procedure

For each step *i* in the original Hsiao07 MCMC chain, the raw parameter vector is:

```
X^i = [α_A^i, α_B^i, α_C^i, E(B−V)^i, R_V^i, t_A^i, t_B^i, t_C^i]
```

**Step 1 — Apply template bias correction:**
A nearby SN Ia is randomly selected from the simulation sample (Section 5.1) whose **inferred phase** lies within ±10 days of the current MCMC phase for each image. The median offset between inferred and actual phase for that SN provides corrections δt^j_A, δt^j_B, δt^j_C. Corresponding corrections to normalisations δα^j are also applied. (Extinction parameters R_V and E(B−V) are assumed to have been correctly recovered in the simulation, so no correction is applied to them from this step.)

**Step 2 — Apply microlensing and millilensing correction:**
A set of three simulated spectra (one per image) is randomly selected from the microlensing/millilensing simulation sample (Section 5.2). The median offsets in all parameters (δt^k, δα^k, δE(B−V)^k, δR_V^k) from that simulation are extracted.

**Step 3 — Construct modified parameter vector:**
The corrected parameter vector for this MCMC step becomes:

```
Y^i = X^i + δ^j + δ^k

Explicitly:
Y_1^i = α_A^i + δα_A^j + δα_A^k
Y_2^i = α_B^i + δα_B^j + δα_B^k
Y_3^i = α_C^i + δα_C^j + δα_C^k
Y_4^i = E(B−V)^i + δE(B−V)^k
Y_5^i = R_V^i + δR_V^k
Y_6^i = t_A^i + δt_A^j + δt_A^k
Y_7^i = t_B^i + δt_B^j + δt_B^k
Y_8^i = t_C^i + δt_C^j + δt_C^k
```

The collection of all modified vectors {**Y**^i} constitutes the **final posterior distribution** from which phases, time delays, and magnifications are reported.

### 6.2 Relative time delays

The observer-frame relative time delays are computed from the modified phase posteriors as:

```
Δt_AB = (t_B − t_A) × (1 + z)     [observer-frame days]
Δt_BC = (t_C − t_B) × (1 + z)     [observer-frame days]
```

where *z* = 1.782 is the SN redshift. The factor (1 + z) converts from rest-frame to observer-frame days.

---

## 7. Final Results

### 7.1 Phases (rest-frame days after B-band peak)

| Image | Phase (pre-unblinding, Hsiao07) | Phase (post-unblinding, extended bin) |
|---|---|---|
| B | **6.5 +2.4/−1.8 d** | 6.2 +2.5/−1.7 d |
| C | **24.3 +3.9/−3.9 d** | 24.5 +3.6/−3.4 d |
| A | **50.6 +16.1/−15.3 d** | 54.5 +18.7/−17.0 d |

SALT3-NIR comparison: t_B = 5.6 +2.3/−2.1 d, t_C = 26.7 +2.5/−4.2 d, t_A = 42.8 +11.4/−9.7 d. All consistent within uncertainties.

### 7.2 Observer-frame relative time delays (Hsiao07, pre-unblinding)

| Delay | Value |
|---|---|
| Δt_AB (= t_B − t_A, observer frame) | −122.3 +43.7/−43.8 d |
| Δt_BC (= t_C − t_B, observer frame) | +49.3 +12.2/−14.7 d |
| Δt_AC | −73.6 +44.7/−46.1 d |

### 7.3 Host-galaxy extinction from the fit

From the joint Hsiao07 fit: **E(B−V) = 0.27 ± 0.02, R_V = 2.73 ± 0.17**, implying A_V ≳ 0.7 mag — the SN exploded in a dusty environment.

---

## 8. Post-Unblinding Robustness Test (Section 8)

After the blind analysis was unblinded by a third party, an additional test was performed: the third phase bin used to select nearby SN Ia comparison spectra was extended from (30, 60) d to **(30, 80) d**. This tests sensitivity to the upper boundary of the simulation phase range, motivated by the lack of template spectra beyond ~40 days for Image A's low SNR.

The resulting shifts in the time delays were 11.5 d (Δt_AB) and 1.0 d (Δt_BC), which correspond to ~0.25σ and ~0.1σ, respectively — statistically insignificant. The impact on the combined H₀ inference (Pascale et al. 2024) was < 1%.

---

## 9. Summary of Software and Data Products

| Component | Software/resource |
|---|---|
| Pipeline reduction | `jwst` v1.10.2, context `jwst_1087.pmap` |
| Optimal extraction | MOSE notebook (Horne 1986 algorithm) |
| PSF models | `webbpsf` |
| Template fitting | `sncosmo` (Barbary et al. 2023) + Hsiao07 template |
| MCMC sampling | `emcee` (Foreman-Mackey et al. 2013) |
| SALT3-NIR fitting | `sncosmo` + SALT3-NIR (Pierel et al. 2022) |
| SN classification | NGSF (Goldwasser et al. 2022) |
| Continuum-independent matching | SNID (Blondin & Tonry 2007) |
| Spectral data | MAST doi: 10.17909/rqdx-3976 |

---

## 10. Key Assumptions and Caveats

1. The residuals added to template spectra during simulation (Section 5.1) are assumed to be **spectrally uncorrelated** (white noise). Any correlated residuals (e.g., from imperfect host subtraction or spectral features) are not modelled.
2. The Hsiao07 template assumes a **standard SN Ia** spectral evolution; intrinsic SN diversity is treated as a source of scatter, not a systematic bias.
3. The analysis uses only **NIRSpec spectra from a single epoch** (2023 April 22). No imaging data or lens-model predictions were incorporated.
4. The SALT3-NIR model's validity ceiling of 50 days limits its use for Image A and means its joint-fit result may be biased.
5. The **microlensing treatment** assumes four specific theoretical caustic models; uncertainties from model selection are not fully quantified.
6. The **grey (achromatic) assumption** for millilensing is valid given the small SN photosphere relative to millilens caustic scales.

---

## References

- Barbary, K. et al. 2023, SNCosmo v2.10.1, Zenodo, doi: 10.5281/zenodo.8091892
- Blondin, S. & Tonry, J. L. 2007, ApJ, 666, 1024
- Foreman-Mackey, D. et al. 2013, PASP, 125, 306
- Gilman, D. et al. 2020, MNRAS, 491, 6077
- Goldwasser, S. et al. 2022, TNS AstroNote, 191, 1
- Horne, K. 1986, PASP, 98, 609
- Hsiao, E. Y. et al. 2007, ApJ, 663, 1187
- Huber, S. et al. 2021, A&A, 646, A110
- Huber, S. et al. 2019, A&A, 631, A161
- Kenworthy, W. D. et al. 2021, ApJ, 923, 265
- Moffat, A. F. J. 1969, A&A, 3, 455
- Pierel, J. D. R. et al. 2022, ApJ, 939, 11 (SALT3-NIR)
- Suyu, S. H. et al. 2020, arXiv:2002.08378
- Whiting, E. 1968, JQSRT, 8, 1379
