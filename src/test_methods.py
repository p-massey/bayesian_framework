import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from datetime import datetime
import sncosmo
import dynesty
from joblib import Parallel, delayed

# --- CONFIGURATION ---
SPECTRA_DIR = "data/spectra/all_spectra"
PARAM_FILE = os.path.join("data/spectra/test", "cfasnIa_param.dat")
NLIVE = 100
MODEL_NAME = 'salt3'
N_CORES = 8  # Specifically using 8 cores as requested

def parse_param_file(file_path):
    params = {}
    if not os.path.exists(file_path):
        print(f"Error: Param file {file_path} not found.")
        return params
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            sn_name = parts[0].lower()
            try:
                z = float(parts[1])
                mjd_max = float(parts[2])
                params[sn_name] = {'z': z, 'mjd_max': mjd_max}
            except ValueError:
                continue
    return params

def date_str_to_mjd(date_str):
    parts = date_str.split('.')
    yyyymmdd = parts[0]
    decimal = float("0." + parts[1]) if len(parts) > 1 else 0.0
    dt = datetime.strptime(yyyymmdd, "%Y%m%d")
    t = Time(dt)
    mjd = t.mjd + decimal
    return mjd

def load_flm_spectrum(file_path):
    # Handles both .flm and .dat (assumes 2 or 3 columns)
    data = np.genfromtxt(file_path, invalid_raise=False)
    if len(data.shape) < 2:
        return np.array([]), np.array([]), np.array([])
        
    wavelength = data[:, 0]
    flux = data[:, 1]
    
    if data.shape[1] < 3:
        flux_err = 0.1 * np.abs(flux) 
    else:
        flux_err = data[:, 2]
        
    mask = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    return wavelength[mask], flux[mask], flux_err[mask]

def fit_full(wavelength, flux, flux_err, model_name='salt3', redshift=None):
    model = sncosmo.Model(source=model_name)
    if redshift is not None:
        model.set(z=redshift)
        params = ['t0', 'x1', 'c', 'log10_x0']
        priors = {'t0': (-100, 100), 'x1': (-6, 6), 'c': (-1.5, 1.5), 'log10_x0': (-20, -2)}
    else:
        params = ['z', 't0', 'x1', 'c', 'log10_x0']
        priors = {'z': (0.005, 0.1), 't0': (-100, 100), 'x1': (-6, 6), 'c': (-1.5, 1.5), 'log10_x0': (-20, -2)}

    def prior_transform(utheta):
        theta = np.zeros_like(utheta)
        for i, p in enumerate(params):
            low, high = priors[p]
            theta[i] = utheta[i] * (high - low) + low
        return theta

    def loglike(theta):
        param_dict = dict(zip(params, theta))
        if 'log10_x0' in param_dict:
            param_dict['x0'] = 10**param_dict.pop('log10_x0')
        model.set(**param_dict)
        try:
            model_flux = model.flux(0.0, wavelength)
        except:
            return -1e10
        chisq = np.sum(((flux - model_flux) / flux_err)**2)
        return -0.5 * chisq if not np.isnan(chisq) else -1e10

    sampler = dynesty.NestedSampler(loglike, prior_transform, len(params), nlive=NLIVE, sample='rslice', bootstrap=0)
    sampler.run_nested(print_progress=False)
    return sampler.results, params

def fit_nuisance(wavelength, flux, flux_err, model_name='salt3', redshift=None):
    model = sncosmo.Model(source=model_name)
    if redshift is not None:
        model.set(z=redshift)
        params = ['t0', 'x1', 'c']
        priors = {'t0': (-100, 100), 'x1': (-6, 6), 'c': (-1.5, 1.5)}
    else:
        params = ['z', 't0', 'x1', 'c']
        priors = {'z': (0.005, 0.1), 't0': (-100, 100), 'x1': (-6, 6), 'c': (-1.5, 1.5)}

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
            model.set(x0=1.0)
            model_flux_unit = model.flux(0.0, wavelength)
        except:
            return -1e10
        w = 1.0 / flux_err**2
        num = np.sum(flux * model_flux_unit * w)
        den = np.sum(model_flux_unit**2 * w)
        if den <= 0: return -1e10
        x0_best = num / den
        if x0_best <= 0: return -1e10
        model_flux = x0_best * model_flux_unit
        chisq = np.sum(((flux - model_flux) / flux_err)**2)
        return -0.5 * chisq if not np.isnan(chisq) else -1e10

    sampler = dynesty.NestedSampler(loglike, prior_transform, len(params), nlive=NLIVE, sample='rslice', bootstrap=0)
    sampler.run_nested(print_progress=False)
    return sampler.results, params

def process_single_file(filename, sn_params):
    file_path = os.path.join(SPECTRA_DIR, filename)
    
    # Robust parsing
    if filename.startswith('snf') or filename.startswith('sne'):
        parts = filename.split('-')
        sn_id = f"{parts[0]}-{parts[1]}"
        date_str = parts[2]
        param_lookup = sn_id.lower()
    else:
        parts = filename.split('-')
        sn_id = parts[0]
        date_str = parts[1]
        param_lookup = sn_id[2:] if sn_id.startswith('sn') else sn_id
    
    if param_lookup not in sn_params:
        return None
        
    p = sn_params[param_lookup]
    if p['mjd_max'] > 90000:
        return None
    
    z = p['z']
    try:
        mjd_obs = date_str_to_mjd(date_str)
    except:
        return None
        
    true_age = mjd_obs - p['mjd_max']
    
    wave, flux, err = load_flm_spectrum(file_path)
    if len(wave) < 10:
        return None
        
    mask = (wave > 3500) & (wave < 8000)
    wave, flux, err = wave[mask], flux[mask], err[mask]
    if len(wave) < 10:
        return None
    
    # 1. Full Fit
    try:
        res_f, par_f = fit_full(wave, flux, err, redshift=z)
        weights_f = np.exp(res_f.logwt - res_f.logz[-1])
        s_f = dynesty.utils.resample_equal(res_f.samples, weights_f)
        age_f = -np.mean(s_f[:, par_f.index('t0')])
        age_f_err = np.std(s_f[:, par_f.index('t0')])
    except Exception as e:
        age_f, age_f_err = np.nan, np.nan
        
    # 2. Nuisance Fit
    try:
        res_n, par_n = fit_nuisance(wave, flux, err, redshift=z)
        weights_n = np.exp(res_n.logwt - res_n.logz[-1])
        s_n = dynesty.utils.resample_equal(res_n.samples, weights_n)
        age_n = -np.mean(s_n[:, par_n.index('t0')])
        age_n_err = np.std(s_n[:, par_n.index('t0')])
    except Exception as e:
        age_n, age_n_err = np.nan, np.nan
    
    return {
        'filename': filename, 'true_age': true_age,
        'full_age': age_f, 'full_age_err': age_f_err,
        'nuis_age': age_n, 'nuis_age_err': age_n_err
    }

def run_test():
    sn_params = parse_param_file(PARAM_FILE)
    flm_files = sorted([f for f in os.listdir(SPECTRA_DIR) if f.endswith('.flm') or f.endswith('.dat')])
    
    print(f"Found {len(flm_files)} files in {SPECTRA_DIR}. Processing with {N_CORES} cores...")

    # Run in parallel with exactly N_CORES
    results_raw = Parallel(n_jobs=N_CORES, verbose=10)(
        delayed(process_single_file)(f, sn_params) for f in flm_files
    )
    
    # Filter out None values
    results = [r for r in results_raw if r is not None]

    df = pd.DataFrame(results)
    df.to_csv("outputs/csvs/allcfa_results.csv", index=False)
    
    # Plotting
    df = df.dropna()
    if df.empty:
        print("No valid results to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.errorbar(df['true_age'], df['full_age'], yerr=df['full_age_err'], fmt='o', label='Full Fit (log10_x0 sampled)', color='blue', alpha=0.5)
    ax1.errorbar(df['true_age'], df['nuis_age'], yerr=df['nuis_age_err'], fmt='s', label='Nuisance (x0 marginalized)', color='red', alpha=0.5)
    
    min_age = min(df['true_age'].min(), df['full_age'].min(), df['nuis_age'].min()) - 5
    max_age = max(df['true_age'].max(), df['full_age'].max(), df['nuis_age'].max()) + 5
    x_range = np.linspace(min_age, max_age, 100)
    ax1.plot(x_range, x_range, 'k--', label='1:1 Line')
    ax1.set_ylabel('Inferred Age (days)')
    ax1.set_title(f'Method Comparison on Random Spectra ({len(df)} files)')
    ax1.legend()
    ax1.grid(True, ls=':', alpha=0.6)
    
    ax2.errorbar(df['true_age'], df['full_age'] - df['true_age'], yerr=df['full_age_err'], fmt='o', color='blue', alpha=0.5)
    ax2.errorbar(df['true_age'], df['nuis_age'] - df['true_age'], yerr=df['nuis_age_err'], fmt='s', color='red', alpha=0.5)
    ax2.axhline(0, color='k', ls='--')
    ax2.set_xlabel('True Age (days)')
    ax2.set_ylabel('Residual (days)')
    ax2.grid(True, ls=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("outputs/plots/allcfa_comparison.png")
    print(f"\nPipeline complete. Processed {len(df)} matched files.")
    print("Results saved to outputs/csvs/allcfa_test_results.csv and outputs/plots/allcfa_comparison.png")

if __name__ == "__main__":
    run_test()
