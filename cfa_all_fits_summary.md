# CfA Supernova Spectra: Nested Sampling Phase Determination Results

We have completed the single-spectrum nested sampling phase determination analysis on the entire CfA supernova database. Using the rest-frame phase range restriction of **$-12$ to $+35$ days**, a total of **1,808 spectra** were fitted in parallel using the `salt3` template model inside `sncosmo` and the `dynesty` nested sampler.

The analysis was executed using the parallelized script [fit_all_cfa_parallel.py](file:///Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/scripts/fit_all_cfa_parallel.py). The outputs were compiled and visualized in the Jupyter notebook [cfa_all_fits_summary.ipynb](file:///Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/notebooks/cfa_all_fits_summary.ipynb).

---

## 1. Quality Cuts on the Results

To filter out poor template fits (due to peculiar subclasses like 91bg, extreme reddening, or bad extractions) and unconstrained nested sampler results, we define the following standard quality cuts:
1. **Goodness of Fit**: $\chi^2_{\text{red}} \le 10.0$ (removes bad fits and severe template mismatch).
2. **Phase Constraint**: $\sigma_{t0} \le 1.0$ day (removes unconstrained fits where the sampler failed to converge or the spectrum lacks diagnostic features).

Applying these quality cuts cleans the sample from **1,808** to **1,542** spectra ($\approx 85\%$ survival rate). 

---

## 2. Statistical Metrics Before & After Quality Cuts

The table below breaks down the phase determination residuals ($A_{\text{spec}} - A_{\text{true}}$ in rest-frame days) for each spectroscopic subtype of Type Ia Supernovae before and after quality cuts:

### A. Raw Results (Before Quality Cuts, N = 1,808)
| Subtype | Count | Median Bias (days) | Standard Deviation (days) | Robust MAD (days) |
| :--- | :---: | :---: | :---: | :---: |
| **All Spectra** | 1,808 | $-1.15$ | $7.06$ | $2.77$ |
| **Normal (N)** | 943 | $-0.78$ | $5.97$ | $2.17$ |
| **91T-like** | 201 | $-1.12$ | $4.41$ | $1.64$ |
| **High-Velocity (HV)** | 342 | $-2.92$ | $5.08$ | $2.70$ |
| **91bg-like** | 114 | $+6.43$ | $11.72$ | $7.92$ |
| **Peculiar (pec)** | 55 | $-5.24$ | $11.02$ | $7.13$ |
| **Unknown** | 153 | $-1.96$ | $10.02$ | $4.05$ |

### B. Cleaned Results (After Quality Cuts, N = 1,542)
| Subtype | Count | Median Bias (days) | Standard Deviation (days) | Robust MAD (days) |
| :--- | :---: | :---: | :---: | :---: |
| **All Spectra** | 1,542 | $-1.15$ | $4.80$ | $2.42$ |
| **Normal (N)** | 873 | $-0.78$ | $3.78$ | $2.12$ |
| **91T-like** | 193 | $-1.12$ | $3.11$ | $1.63$ |
| **High-Velocity (HV)** | 273 | $-2.81$ | $3.49$ | $2.30$ |
| **91bg-like** | 51 | $+3.74$ | $10.46$ | $6.58$ |
| **Peculiar (pec)** | 34 | $-5.32$ | $10.24$ | $5.03$ |
| **Unknown** | 118 | $-1.67$ | $7.45$ | $2.96$ |

> [!TIP]
> **Key Impact of Quality Cuts**:
> * For **Normal SNe Ia**, the standard deviation scatter drops significantly from **$5.97$ days to $3.78$ days** (a reduction of 2.2 days), while preserving 93% of the sample.
> * For **91bg-like SNe**, the count drops from 114 to 51 (as expected, 91bg spectra have strong Titanium absorption lines that a normal SN Ia template cannot fit, leading to high $\chi^2_{\text{red}}$ values).

---

## 3. Comparison Plots (SpecSim Style)

To match the visualization format of the `cfa_spectra_pipeline` comparisons (e.g. `specsim_comparison_sigma_0.15.png`), we have plotted the results using a 1x2 panel layout:
* **Left Panel**: True phase $t_{\text{LC}}$ vs. Estimated phase $t_{\text{est}}$ with a dashed red 1:1 line.
* **Right Panel**: True phase $t_{\text{LC}}$ vs. Residual with a horizontal zero line, running median (solid red line), and shaded running MAD region.

### A. Raw Nested Sampling Results (No Quality Cuts)
![Raw Nested Sampling Comparison](/Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/outputs/nested_sampling_comparison_raw.png)

### B. Cleaned Nested Sampling Results (With Quality Cuts)
![Cleaned Nested Sampling Comparison](/Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/outputs/nested_sampling_comparison_clean.png)

---

## 4. Key Files Created
* **Comparison Plotter Script**: [plot_nested_sampling_comparison.py](file:///Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/scripts/plot_nested_sampling_comparison.py)
* **Parallel Fitting Script**: [fit_all_cfa_parallel.py](file:///Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/scripts/fit_all_cfa_parallel.py)
* **Jupyter Analysis Notebook**: [cfa_all_fits_summary.ipynb](file:///Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/notebooks/cfa_all_fits_summary.ipynb)
* **Fitted Results CSV**: [cfa_fits_results.csv](file:///Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/data/cfa_fits_results.csv)
