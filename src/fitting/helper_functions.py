import os
import argparse
import numpy as np
import sncosmo
import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt


def load_flm_spectrum(file_path):
    data = np.genfromtxt(file_path, invalid_raise=False)

    wavelength = data[:, 0]
    flux = data[:, 1]

    # Check if we have a valid 3rd column with non-zero values
    if data.shape[1] >= 3 and not np.all(data[:, 2] == 0):
        flux_err = data[:, 2]
    else:
        # If 3rd column is missing or all zeros, create a dummy error (10% of flux)
        # np.abs to ensure errors are positive
        flux_err = 0.1 * np.abs(flux)
        # Ensure there are not zero errors
        flux_err[flux_err == 0] = np.median(flux_err) if np.any(flux_err > 0) else 1.0

    # Only filter for NaN/Inf values
    mask = np.isfinite(flux) & np.isfinite(flux_err)

    return wavelength[mask], flux[mask], flux_err[mask]


def fit_spectrum_with_dynesty(wavelength, flux, flux_err, model_name, live_point_number, redshift=None):
    model = sncosmo.Model(source=model_name)

    # Define parameters based on model type
    if 'salt' in model_name.lower():
        # SALT models (salt2, salt3) use x1 and c
        params = ['t0', 'x1', 'c', 'log10_x0']
        priors = {
            't0': (-20, 50),
            'x1': (-3, 3),
            'c': (-0.8, 0.8),
            'log10_x0': (-20, -1)
        }
    else:
        # Hsiao model params
        params = ['t0', 'log10_x0']
        priors = {
            't0': (-20, 50),
            'log10_x0': (-20, -1)
        }

    # Add redshift as a param if it's not fixed
    if redshift is None:
        params.insert(0, 'z')
        priors['z'] = (0.005, 0.1)
    else:
        model.set(z=redshift)

    def prior_transform(utheta):
        theta = np.zeros(len(params))
        for i, p in enumerate(params):
            low, high = priors[p]
            theta[i] = utheta[i] * (high - low) + low
        return theta

    def loglike(theta):
        param_dict = dict(zip(params, theta))

        # Handle the amplitude naming convention
        if 'log10_x0' in param_dict:
            val = 10 ** param_dict.pop('log10_x0')
            # For Hsiao, sncosmo expects 'amplitude'
            # For SALT, it expects 'x0'
            if 'salt' in model_name.lower():
                param_dict['x0'] = val
            else:
                param_dict['amplitude'] = val

        try:
            model.set(**param_dict)
            model_flux = model.flux(0.0, wavelength)
            # Avoid division by zero if flux_err is 0 (safety check)
            chisq = np.sum(((flux - model_flux) / (flux_err + 1e-20)) ** 2)
            return -0.5 * chisq if np.isfinite(chisq) else -1e10
        except:
            return -1e10

    sampler = dynesty.NestedSampler(loglike, prior_transform, len(params), nlive=live_point_number, sample='rslice', bootstrap=0)
    sampler.run_nested()

    return sampler.results, params


def fit_spectrum_with_dynesty_nox0(wavelength, flux, flux_err, model_name='salt3', redshift=None):
    """
    Fits a single spectrum using dynesty and sncosmo.
    Treats x0 as a nuisance parameter by analytically optimizing it.
    """
    model = sncosmo.Model(source=model_name)

    if redshift is not None:
        model.set(z=redshift)
        params = ['t0', 'x1', 'c']
        priors = {
            't0': (-20, 50),
            'x1': (-3, 3),
            'c': (-0.8, 0.8)
        }
    else:
        params = ['z', 't0', 'x1', 'c']
        priors = {
            'z': (0.005, 0.1),
            't0': (-20, 50),
            'x1': (-3, 3),
            'c': (-0.8, 0.8)
        }

    obs_time = 0.0

    def prior_transform(utheta):
        theta = np.zeros_like(utheta)
        for i, p in enumerate(params):
            low, high = priors[p]
            theta[i] = utheta[i] * (high - low) + low
        return theta

    def loglike(theta):
        param_dict = dict(zip(params, theta))
        model.set(**param_dict)

        try:
            # Set unit x0 to get the model shape
            model.set(x0=1.0)
            model_flux_unit = model.flux(obs_time, wavelength)
        except:
            return -1e10

        # Analytically marginalize over x0 (Nuisance parameter)
        # x0_best = sum(data * model / err^2) / sum(model^2 / err^2)
        w = 1.0 / flux_err ** 2
        num = np.sum(flux * model_flux_unit * w)
        den = np.sum(model_flux_unit ** 2 * w)

        if den <= 0:
            return -1e10

        x0_best = num / den
        if x0_best <= 0:
            return -1e10

        model_flux = x0_best * model_flux_unit
        chisq = np.sum(((flux - model_flux) / flux_err) ** 2)

        # Note: Proper marginalization would include -0.5 * ln(den)
        # but for parameter estimation of other variables, chisq is the primary driver.
        return -0.5 * chisq if not np.isnan(chisq) else -1e10

    ndim = len(params)
    sampler = dynesty.NestedSampler(loglike, prior_transform, ndim, nlive=500, sample='rslice', bootstrap=0)

    print(f"Starting dynesty fit with {ndim} parameters (x0 as nuisance)...")
    sampler.run_nested()
    return sampler.results, params
