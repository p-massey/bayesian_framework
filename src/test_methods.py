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
SPECTRA_DIR = "data/all_spectra"
PARAM_FILE = os.path.join("data", "cfasnIa_param.dat")
NLIVE = 100
MODEL_NAME = 'salt3'
N_CORES = 16  # Optimized for 16-core cluster runs

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

def fit_full(wavelength, flux, flux_err, model_name='salt3', redshift=None, checkpoint_file=None):
    """Standard Nested Sampling fit including x0 sampling."""
    model = sncosmo.Model(source=model_name)
    if redshift is not None:
        model.set(z=redshift)
        params = ['t0', 'x1', 'c', 'log10_x0']
        # Age = -t0. To cover age (-20, 50), t0 must be (-50, 20)
        priors = {'t0': (-50, 20), 'x1': (-6, 6), 'c': (-1.5, 1.5), 'log10_x0': (-20, -2)}
    else:
        params = ['z', 't0', 'x1', 'c', 'log10_x0']
        priors = {'z': (0.005, 0.1), 't0': (-50, 20), 'x1': (-6, 6), 'c': (-1.5, 1.5), 'log10_x0': (-20, -2)}

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
    sampler.run_nested(print_progress=False, checkpoint_file=checkpoint_file)
    return sampler.results, params

def fit_nuisance(wavelength, flux, flux_err, model_name='salt3', redshift=None, checkpoint_file=None):
    """Nested Sampling fit with x0 analytically marginalized."""
    model = sncosmo.Model(source=model_name)
    if redshift is not None:
        model.set(z=redshift)
        params = ['t0', 'x1', 'c']
        priors = {'t0': (-50, 20), 'x1': (-6, 6), 'c': (-1.5, 1.5)}
    else:
        params = ['z', 't0', 'x1', 'c']
        priors = {'z': (0.005, 0.1), 't0': (-50, 20), 'x1': (-6, 6), 'c': (-1.5, 1.5)}

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
    sampler.run_nested(print_progress=False, checkpoint_file=checkpoint_file)
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
    
    # Ensure checkpoint directory exists
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Execute Fits
    res = {'filename': filename, 'true_age': true_age, 'z_true': p['z']}
    
    for method, fit_func in [('full', fit_full), ('nuis', fit_nuisance)]:
        # Define a unique checkpoint file for this fit
        checkpoint_file = os.path.join(checkpoint_dir, f"{filename}_{method}.save")
        
        try:
            results, par_names = fit_func(wave, flux, err, redshift=p['z'], checkpoint_file=checkpoint_file)
            weights = np.exp(results.logwt - results.logz[-1])
            samples = dynesty.utils.resample_equal(results.samples, weights)
            
            # Store log Evidence and basic diagnostics
            res[f'{method}_logz'] = results.logz[-1]
            res[f'{method}_logzerr'] = results.logzerr[-1]
            res[f'{method}_ncall'] = results.ncall
            res[f'{method}_eff'] = results.eff
            
            # Compute best-fit Chi2 (at the mean parameter values)
            mean_params = np.mean(samples, axis=0)
            p_dict_best = dict(zip(par_names, mean_params))
            
            model = sncosmo.Model(source=MODEL_NAME)
            model.set(z=p['z'])
            
            if method == 'full':
                if 'log10_x0' in p_dict_best: p_dict_best['x0'] = 10**p_dict_best.pop('log10_x0')
                model.set(**p_dict_best)
                m_flux = model.flux(0.0, wave)
                best_chisq = np.sum(((flux - m_flux) / err)**2)
            else:
                model.set(**p_dict_best)
                model.set(x0=1.0)
                m_flux_unit = model.flux(0.0, wave)
                w = 1.0 / err**2
                x0_best = np.sum(flux * m_flux_unit * w) / np.sum(m_flux_unit**2 * w)
                best_chisq = np.sum(((flux - x0_best * m_flux_unit) / err)**2)
            
            res[f'{method}_chi2'] = best_chisq
            res[f'{method}_ndof'] = len(wave) - len(par_names)

            for i, name in enumerate(par_names):
                s = samples[:, i]
                # Common stats
                mean, std = np.mean(s), np.std(s)
                median = np.median(s)
                q16, q84 = np.percentile(s, [16, 84])
                
                if name == 't0':
                    res[f'{method}_age'] = -mean
                    res[f'{method}_age_err'] = std
                    res[f'{method}_t0_mean'] = mean
                    res[f'{method}_t0_std'] = std
                    res[f'{method}_t0_median'] = median
                elif name == 'log10_x0':
                    x0_samples = 10**s
                    res[f'{method}_x0_mean'] = np.mean(x0_samples)
                    res[f'{method}_x0_std'] = np.std(x0_samples)
                    res[f'{method}_x0_median'] = np.median(x0_samples)
                    res[f'{method}_log10_x0_mean'] = mean
                    res[f'{method}_log10_x0_std'] = std
                else:
                    res[f'{method}_{name}_mean'] = mean
                    res[f'{method}_{name}_std'] = std
                    res[f'{method}_{name}_median'] = median
                    res[f'{method}_{name}_q16'] = q16
                    res[f'{method}_{name}_q84'] = q84
            
            # Special case for nuisance x0 (calculated for every sample)
            if method == 'nuis':
                x0_list = []
                for sample in samples:
                    p_dict = dict(zip(par_names, sample))
                    model.set(**p_dict)
                    model.set(x0=1.0)
                    m_flux_unit = model.flux(0.0, wave)
                    w = 1.0 / err**2
                    num = np.sum(flux * m_flux_unit * w)
                    den = np.sum(m_flux_unit**2 * w)
                    x0_val = num / den if den > 0 else np.nan
                    x0_list.append(x0_val)
                
                x0_arr = np.array(x0_list)
                res[f'nuis_x0_mean'] = np.nanmean(x0_arr)
                res[f'nuis_x0_std'] = np.nanstd(x0_arr)
                res[f'nuis_x0_median'] = np.nanmedian(x0_arr)

            # If fit succeeded, remove the checkpoint file
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                    
        except Exception as e:
            print(f"Error fitting {filename} with {method}: {e}")
            res[f'{method}_age'] = np.nan
            
    return res

def run_test():
    """Main pipeline execution with checkpointing."""
    os.makedirs("outputs/csvs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    output_csv = "outputs/csvs/allcfa_results.csv"

    sn_params = parse_param_file(PARAM_FILE)
    flm_files = sorted([f for f in os.listdir(SPECTRA_DIR) if f.endswith(('.flm', '.dat'))])
    
    # --- CHECKPOINTING ---
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        processed_files = set(existing_df['filename'].unique())
        print(f"Found existing results. {len(processed_files)} spectra already processed.")
        files_to_run = [f for f in flm_files if f not in processed_files]
    else:
        existing_df = pd.DataFrame()
        files_to_run = flm_files

    if not files_to_run:
        print("All candidate files already processed. Use a clean file to re-run.")
        return

    print(f"Processing {len(files_to_run)} files (out of {len(flm_files)}) on {N_CORES} cores...")

    # Process in chunks to allow periodic saving (checkpointing)
    chunk_size = 50 
    for i in range(0, len(files_to_run), chunk_size):
        chunk = files_to_run[i:i + chunk_size]
        print(f"\n--- Processing Chunk {i//chunk_size + 1} ({i} to {i+len(chunk)}) ---")
        
        results_raw = Parallel(n_jobs=N_CORES)(
            delayed(process_single_file)(f, sn_params) for f in chunk
        )
        
        new_results = [r for r in results_raw if r is not None]
        if new_results:
            new_df = pd.DataFrame(new_results)
            if not existing_df.empty:
                existing_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                existing_df = new_df
            
            # Save checkpoint
            existing_df.to_csv(output_csv, index=False)
            print(f"Checkpoint saved: {len(existing_df)} total spectra in {output_csv}")

    print(f"\nAll processing complete. Results: {output_csv}")

if __name__ == "__main__":
    run_test()
