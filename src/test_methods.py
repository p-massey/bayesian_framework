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
N_CORES = 8  # Optimized for your 8-core production runs

def parse_param_file(file_path):
    """Parses the SN parameter file for redshift and MJD of maximum."""
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
    """Converts YYYYMMDD.DD strings to MJD."""
    parts = date_str.split('.')
    yyyymmdd = parts[0]
    decimal = float("0." + parts[1]) if len(parts) > 1 else 0.0
    dt = datetime.strptime(yyyymmdd, "%Y%m%d")
    t = Time(dt)
    return t.mjd + decimal

def load_flm_spectrum(file_path):
    """Loads spectrum from .flm or .dat files."""
    try:
        data = np.genfromtxt(file_path, invalid_raise=False)
        if len(data.shape) < 2:
            return np.array([]), np.array([]), np.array([])
        
        wavelength = data[:, 0]
        flux = data[:, 1]
        flux_err = data[:, 2] if data.shape[1] >= 3 else 0.1 * np.abs(flux)
        
        mask = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
        return wavelength[mask], flux[mask], flux_err[mask]
    except Exception:
        return np.array([]), np.array([]), np.array([])

def fit_full(wavelength, flux, flux_err, model_name='salt3', redshift=None):
    """Standard Nested Sampling fit including x0 sampling."""
    model = sncosmo.Model(source=model_name)
    if redshift is not None:
        model.set(z=redshift)
        params = ['t0', 'x1', 'c', 'log10_x0']
        priors = {'t0': (-100, 100), 'x1': (-6, 6), 'c': (-1.5, 1.5), 'log10_x0': (-20, -2)}
    else:
        params = ['z', 't0', 'x1', 'c', 'log10_x0']
        priors = {'z': (0.005, 0.1), 't0': (-100, 100), 'x1': (-6, 6), 'c': (-1.5, 1.5), 'log10_x0': (-20, -2)}

    def pt(u):
        t = np.zeros_like(u)
        for i, p in enumerate(params):
            low, high = priors[p]
            t[i] = u[i] * (high - low) + low
        return t

    def ll(t):
        p_dict = dict(zip(params, t))
        if 'log10_x0' in p_dict: p_dict['x0'] = 10**p_dict.pop('log10_x0')
        model.set(**p_dict)
        try:
            m_flux = model.flux(0.0, wavelength)
            chisq = np.sum(((flux - m_flux) / flux_err)**2)
            return -0.5 * chisq if not np.isnan(chisq) else -1e10
        except:
            return -1e10

    sampler = dynesty.NestedSampler(ll, pt, len(params), nlive=NLIVE, sample='rslice')
    sampler.run_nested(print_progress=False)
    return sampler.results, params

def fit_nuisance(wavelength, flux, flux_err, model_name='salt3', redshift=None):
    """Nested Sampling fit with x0 analytically marginalized."""
    model = sncosmo.Model(source=model_name)
    if redshift is not None:
        model.set(z=redshift)
        params = ['t0', 'x1', 'c']
        priors = {'t0': (-100, 100), 'x1': (-6, 6), 'c': (-1.5, 1.5)}
    else:
        params = ['z', 't0', 'x1', 'c']
        priors = {'z': (0.005, 0.1), 't0': (-100, 100), 'x1': (-6, 6), 'c': (-1.5, 1.5)}

    def pt(u):
        t = np.zeros_like(u)
        for i, p in enumerate(params):
            low, high = priors[p]
            t[i] = u[i] * (high - low) + low
        return t

    def ll(t):
        p_dict = dict(zip(params, t))
        model.set(**p_dict)
        try:
            model.set(x0=1.0)
            m_flux_unit = model.flux(0.0, wavelength)
            w = 1.0 / flux_err**2
            num = np.sum(flux * m_flux_unit * w)
            den = np.sum(m_flux_unit**2 * w)
            if den <= 0: return -1e10
            x0_best = num / den
            if x0_best <= 0: return -1e10
            chisq = np.sum(((flux - x0_best * m_flux_unit) / flux_err)**2)
            return -0.5 * chisq if not np.isnan(chisq) else -1e10
        except:
            return -1e10

    sampler = dynesty.NestedSampler(ll, pt, len(params), nlive=NLIVE, sample='rslice')
    sampler.run_nested(print_progress=False)
    return sampler.results, params

def process_single_file(filename, sn_params):
    """Processes one spectrum: identifies SN, loads data, and runs both fits."""
    file_path = os.path.join(SPECTRA_DIR, filename)
    
    # Identify SN and age
    if filename.startswith(('snf', 'sne')):
        parts = filename.split('-')
        sn_id, date_str = f"{parts[0]}-{parts[1]}", parts[2]
        param_lookup = sn_id.lower()
    else:
        parts = filename.split('-')
        sn_id, date_str = parts[0], parts[1]
        param_lookup = sn_id[2:] if sn_id.startswith('sn') else sn_id

    if param_lookup not in sn_params:
        return None
        
    p = sn_params[param_lookup]
    if p['mjd_max'] > 90000: return None
    
    try:
        true_age = date_str_to_mjd(date_str) - p['mjd_max']
    except Exception:
        return None
    
    # Load and Pre-process
    wave, flux, err = load_flm_spectrum(file_path)
    mask = (wave > 3500) & (wave < 8000)
    wave, flux, err = wave[mask], flux[mask], err[mask]
    if len(wave) < 10: return None
    
    # Execute Fits
    res = {'filename': filename, 'true_age': true_age}
    
    for method, fit_func in [('full', fit_full), ('nuis', fit_nuisance)]:
        try:
            results, par_names = fit_func(wave, flux, err, redshift=p['z'])
            weights = np.exp(results.logwt - results.logz[-1])
            samples = dynesty.utils.resample_equal(results.samples, weights)
            t0_samples = samples[:, par_names.index('t0')]
            res[f'{method}_age'] = -np.mean(t0_samples)
            res[f'{method}_age_err'] = np.std(t0_samples)
        except Exception:
            res[f'{method}_age'], res[f'{method}_age_err'] = np.nan, np.nan
            
    return res

def run_test():
    """Main pipeline execution."""
    os.makedirs("outputs/csvs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    sn_params = parse_param_file(PARAM_FILE)
    flm_files = sorted([f for f in os.listdir(SPECTRA_DIR) if f.endswith(('.flm', '.dat'))])
    
    print(f"Found {len(flm_files)} candidate files. Starting {N_CORES}-core parallel processing...")

    results_raw = Parallel(n_jobs=N_CORES, verbose=10)(
        delayed(process_single_file)(f, sn_params) for f in flm_files
    )
    
    results = [r for r in results_raw if r is not None]
    if not results:
        print("No valid fits completed.")
        return

    df = pd.DataFrame(results).dropna()
    output_csv = "outputs/csvs/allcfa_results.csv"
    df.to_csv(output_csv, index=False)
    
    # Plotting
    if df.empty:
        print("DataFrame is empty after dropping NaNs.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.errorbar(df['true_age'], df['full_age'], yerr=df['full_age_err'], fmt='o', label='Full (x0 sampled)', color='blue', alpha=0.5)
    ax1.errorbar(df['true_age'], df['nuis_age'], yerr=df['nuis_age_err'], fmt='s', label='Nuisance (x0 marginalized)', color='red', alpha=0.5)
    
    lims = [min(df['true_age'].min(), df['full_age'].min()) - 5, max(df['true_age'].max(), df['full_age'].max()) + 5]
    ax1.plot(lims, lims, 'k--', label='1:1 Line')
    ax1.set_ylabel('Inferred Age (days)')
    ax1.set_title(f'Method Comparison: CfA Spectra (N={len(df)})')
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
    
    print(f"\nProcessing complete. Results: {output_csv}, Plot: outputs/plots/allcfa_comparison.png")

if __name__ == "__main__":
    run_test()
