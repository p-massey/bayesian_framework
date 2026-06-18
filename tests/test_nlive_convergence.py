import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sncosmo
import dynesty
import time
from joblib import Parallel, delayed

# --- CONFIGURATION ---
SPECTRA_DIR = "data/all_spectra"
PARAM_FILE = "data/cfasnIa_param.dat"
MODEL_NAME = 'salt3'
N_CORES = 16

# Test files (selected for variety)
TEST_FILES = [
    "sn1994ae-19941129.51-fast.flm",  # High SNR
    "sn1995al-19951124.51-mmt.flm",   # Medium SNR
    "sn1996bl-19961016.29-fast.flm"    # Lower SNR
]

NLIVE_VALUES = [20, 50, 100, 150, 200, 300, 400, 500]

def parse_param_file(file_path):
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = line.split()
            sn_name = parts[0].lower()
            params[sn_name] = {'z': float(parts[1]), 'mjd_max': float(parts[2])}
    return params

def load_flm_spectrum(file_path):
    data = np.genfromtxt(file_path, invalid_raise=False)
    wave, flux = data[:, 0], data[:, 1]
    err = data[:, 2] if data.shape[1] >= 3 else 0.1 * np.abs(flux)
    mask = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    return wave[mask], flux[mask], err[mask]

def run_fit(wavelength, flux, flux_err, nlive, method='nuis', redshift=None):
    model = sncosmo.Model(source=MODEL_NAME)
    if redshift: model.set(z=redshift)
    
    if method == 'full':
        params = ['t0', 'x1', 'c', 'log10_x0']
        priors = {'t0': (-50, 20), 'x1': (-6, 6), 'c': (-1.5, 1.5), 'log10_x0': (-20, -2)}
    else:
        params = ['t0', 'x1', 'c']
        priors = {'t0': (-50, 20), 'x1': (-6, 6), 'c': (-1.5, 1.5)}

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
            if method == 'nuis':
                model.set(x0=1.0)
                m_flux_unit = model.flux(0.0, wavelength)
                w = 1.0 / flux_err**2
                num, den = np.sum(flux * m_flux_unit * w), np.sum(m_flux_unit**2 * w)
                if den <= 0: return -1e10
                x0_best = num / den
                if x0_best <= 0: return -1e10
                chisq = np.sum(((flux - x0_best * m_flux_unit) / flux_err)**2)
            else:
                m_flux = model.flux(0.0, wavelength)
                chisq = np.sum(((flux - m_flux) / flux_err)**2)
            return -0.5 * chisq if not np.isnan(chisq) else -1e10
        except: return -1e10

    start_time = time.time()
    sampler = dynesty.NestedSampler(ll, pt, len(params), nlive=nlive, sample='rslice')
    sampler.run_nested(print_progress=False)
    duration = time.time() - start_time
    
    res = sampler.results
    weights = np.exp(res.logwt - res.logz[-1])
    samples = dynesty.utils.resample_equal(res.samples, weights)
    
    out = {
        'duration': duration,
        'logz': res.logz[-1],
        'logzerr': res.logzerr[-1]
    }
    
    for i, name in enumerate(params):
        s = samples[:, i]
        if name == 't0':
            out['age_mean'] = -np.mean(s)
            out['age_std'] = np.std(s)
        elif name == 'log10_x0':
            x0_s = 10**s
            out['x0_mean'] = np.mean(x0_s)
            out['x0_std'] = np.std(x0_s)
        else:
            out[f'{name}_mean'] = np.mean(s)
            out[f'{name}_std'] = np.std(s)
            
    if method == 'nuis':
        x0_list = []
        for sample in samples:
            p_dict = dict(zip(params, sample))
            model.set(**p_dict)
            model.set(x0=1.0)
            m_flux_unit = model.flux(0.0, wavelength)
            w = 1.0 / flux_err**2
            num, den = np.sum(flux * m_flux_unit * w), np.sum(m_flux_unit**2 * w)
            x0_list.append(num/den if den > 0 else np.nan)
        out['x0_mean'] = np.nanmean(x0_list)
        out['x0_std'] = np.nanstd(x0_list)
        
    return out

def test_task(filename, nlive, method, sn_params):
    sn_id = filename.split('-')[0]
    param_lookup = sn_id[2:] if sn_id.startswith('sn') else sn_id
    p = sn_params.get(param_lookup.lower())
    if not p: return None
    
    wave, flux, err = load_flm_spectrum(os.path.join(SPECTRA_DIR, filename))
    mask = (wave > 3500) & (wave < 8000)
    wave, flux, err = wave[mask], flux[mask], err[mask]
    
    res = run_fit(wave, flux, err, nlive, method=method, redshift=p['z'])
    res.update({
        'filename': filename, 
        'nlive': nlive, 
        'method': method, 
        'z_mean': p['z'],
        'z_std': 0.0  # Redshift is fixed in this test
    })
    return res

if __name__ == "__main__":
    sn_params = parse_param_file(PARAM_FILE)
    tasks = []
    for f in TEST_FILES:
        for nl in NLIVE_VALUES:
            for m in ['full', 'nuis']:
                tasks.append((f, nl, m))
                
    print(f"Starting convergence test: {len(tasks)} fits on {N_CORES} cores...")
    results_raw = Parallel(n_jobs=N_CORES)(delayed(test_task)(f, nl, m, sn_params) for f, nl, m in tasks)
    
    df = pd.DataFrame([r for r in results_raw if r])
    os.makedirs("outputs/analysis", exist_ok=True)
    df.to_csv("outputs/analysis/nlive_convergence_results.csv", index=False)
    
    # --- PLOTTING ---
    params_to_plot = ['age', 'x1', 'c', 'x0']
    n_params = len(params_to_plot)
    n_files = len(TEST_FILES)
    
    fig, axes = plt.subplots(n_files, n_params, figsize=(5*n_params, 4*n_files), sharex=True)
    
    for i, f in enumerate(TEST_FILES):
        sub = df[df['filename'] == f]
        for j, p in enumerate(params_to_plot):
            ax = axes[i, j] if n_files > 1 else axes[j]
            for m, col, marker in [('full', 'blue', 'o'), ('nuis', 'red', 's')]:
                m_sub = sub[sub['method'] == m]
                if f'{p}_mean' in m_sub.columns:
                    ax.errorbar(m_sub['nlive'], m_sub[f'{p}_mean'], yerr=m_sub[f'{p}_std'], 
                                 fmt=marker+'-', color=col, label=f"{m.capitalize()}")
            
            if i == 0:
                ax.set_title(f"Convergence: {p.upper()}")
            if j == 0:
                ax.set_ylabel(f"{f}\nValue")
            if i == n_files - 1:
                ax.set_xlabel("NLIVE")
            ax.grid(True, ls=':', alpha=0.6)
            if i == 0 and j == 0:
                ax.legend()

    plt.tight_layout()
    plt.savefig("outputs/analysis/nlive_all_params_convergence.png")
    
    # --- SUMMARY ---
    print("\n--- Final Convergence Results (Highest NLIVE) ---")
    max_nlive = max(NLIVE_VALUES)
    summary_df = df[df['nlive'] == max_nlive].copy()
    cols_to_show = ['filename', 'method', 'age_mean', 'age_std', 'x1_mean', 'x1_std', 'c_mean', 'c_std', 'x0_mean', 'x0_std', 'logz']
    # Filter only columns that exist
    cols_to_show = [c for c in cols_to_show if c in summary_df.columns]
    print(summary_df[cols_to_show].to_string(index=False))
    
    print(f"\nConvergence test complete.")
    print(f"CSV saved to: outputs/analysis/nlive_convergence_results.csv")
    print(f"Plot saved to: outputs/analysis/nlive_all_params_convergence.png")
