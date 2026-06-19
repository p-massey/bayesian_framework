# Likelihood Formulations in SALT3 Fitting Framework

This document details the mathematical formulations, parameterizations, and physical motivations for the likelihood functions implemented in this codebase.

---

## 1. Analytical Optimization of Amplitude ($x_0$)

To reduce the dimensionality of the parameter space and improve nested sampling convergence, we solve for the overall amplitude scaling parameter $x_0$ analytically at each likelihood evaluation.

In the standard **SALT3** model, the flux scales linearly with $x_0$:
$$F_{\text{model}}(\lambda) = x_0 \times F_{\text{unit}}(\lambda)$$

The weighted chi-squared ($\chi^2$) is defined as:
$$\chi^2 = \sum_i \frac{\left(F_i - x_0 F_{\text{unit}, i}\right)^2}{\sigma_i^2}$$

Differentiating with respect to $x_0$ and setting the derivative to zero:
$$\frac{\partial \chi^2}{\partial x_0} = \sum_i \frac{-2 F_{\text{unit}, i} \left(F_i - x_0 F_{\text{unit}, i}\right)}{\sigma_i^2} = 0$$

Solving for $x_0$:
$$x_0 = \frac{\sum_i F_i F_{\text{unit}, i} w_i}{\sum_i F_{\text{unit}, i}^2 w_i}$$

where $w_i = \frac{1}{\sigma_i^2}$ are the weights. 

By solving for $x_0$ analytically, we reduce the dimensions of our parameter space from 4 to 3 ($t_0, x_1, c$), allowing the nested sampler to converge significantly faster and avoiding parameter degeneracies.

---

## 2. Nuisance Likelihood (Standard Formulation)

Located in [salt3.py](file:///Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/src/fitting/salt3.py#L96-L131) and `tests/test_methods.py` under the function `log_likelihood_nuisance`.

### Formulation
$$\ln \mathcal{L} = -0.5 \times \chi^2$$

### Parameterization
In this formulation, the time of maximum $t_0$ is defined as a rest-frame phase offset relative to the observation MJD ($T_{\text{spec}}$):
$$t_{0, \text{MJD}} = T_{\text{spec}} - t_{0, \text{offset}} \times (1 + z)$$

where $t_{0, \text{offset}}$ is constrained by uniform priors (typically $[-50, 50]$ days rest-frame).

### Use Case
This represents the standard Gaussian likelihood. It is appropriate when the noise model ($\sigma_i$) is assumed to fully capture the uncertainties and there is negligible template mismatch.

---

## 3. Scaled Likelihood (Template-Mismatch Formulation)

Located in the parallel fitting script [fit_all_cfa_parallel.py](file:///Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/scripts/fit_all_cfa_parallel.py).

### Formulation
$$\ln \mathcal{L} = -0.5 \times \frac{\chi^2}{\chi^2_{\text{red}}}$$

### Parameterization
Here, $t_0$ is parameterized as an absolute MJD, and the prior bounds are centered around the observation MJD ($T_{\text{spec}}$):
$$t_0 \in \left[T_{\text{spec}} - 50(1+z), T_{\text{spec}} + 20(1+z)\right]$$

### The Role of $\chi^2_{\text{red}}$
Before running the nested sampler, the script performs a quick pre-fit optimization (using `scipy.optimize.minimize`) to find the minimum chi-squared ($\chi^2_{\text{min}}$) and calculates the reduced chi-squared:
$$\chi^2_{\text{red}} = \frac{\chi^2_{\text{min}}}{N_{\text{pixels}} - 4}$$

The log-likelihood evaluated during nested sampling is then scaled by this factor.

### Physical Motivation
Optical spectra typically have thousands of pixels, meaning the formal statistical errors are very small. This results in extremely large chi-squared values ($\chi^2 \sim 1000 - 10000$) due to systematic template mismatch. 

If unscaled, this makes the likelihood surface extremely steep, causing the nested sampler's posterior to collapse to an unrealistically narrow spike, yielding phase uncertainties of $\approx 0.001$ days and rendering the sampler highly sensitive to local minima. 

Scaling the likelihood by $\chi^2_{\text{red}}$ inflates the parameter uncertainties to reflect the systematic limitations of the SALT3 template, stretching the posterior distribution to represent a physically realistic age constraint.
