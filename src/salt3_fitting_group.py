import os
import numpy as np
import pandas as pd
import sncosmo
import dynesty
from joblib import Parallel, delayed
from functools import partial

# --- CONFIGURATION ---
SPECTRA_DIR = "data/all_spectra_dereddened"
PARAM_FILE = os.path.join("data", "cfasnIa_param.dat")
DATA_FILE = os.path.join("data", "cfa_SNID_results.csv")
NLIVE = 200
MODEL_NAME = 'salt3'
N_CORES = 16
FORCE_RERUN = True 
STD_THRESHOLD = 1e-4  # Flag fits as failed if std dev falls below this

def load_flm_spectrum(file_path):
    """Loading an FLM spectrum from datafiles."""
    try:
        data = np.genfromtxt(file_path, invalid_raise=False)
        if len(data.shape) < 2:
            return np.array([]), np.array([]), np.array([])

        wavelength = data[:, 0]
        flux = data[:, 1]

        if data.shape[1] >= 3:
            flux_err = data[:, 2]
            zero_mask = (flux_err <= 0) | (~np.isfinite(flux_err))
            if np.any(zero_mask):
                fallback_err = 0.1 * np.abs(flux)
                min_err = np.percentile(np.abs(flux[flux > 0]), 5) * 0.1 if np.any(flux > 0) else 1e-18
                fallback_err = np.maximum(fallback_err, min_err)
                flux_err[zero_mask] = fallback_err[zero_mask]
        else:
            flux_err = 0.1 * np.abs(flux)

        mask = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
        return wavelength[mask], flux[mask], flux_err[mask]
    except Exception:
        return np.array([]), np.array([]), np.array([])

def parse_param_csv(file_path):
    """
    Loads the SN metadata CSV and returns a dictionary for lookup.
    """

    params = {}
    if not os.path.exists(file_path):
        print(f"Error: CSV file {file_path} not found.")
        return params

    # Load the CSV
    df = pd.read_csv(file_path)

    for _, row in df.iterrows():
        # Ensure we have a valid SN Name to use as a key
        file_name = str(row['Filename'])

        try:
            # In your old code, mjd_max was the date of peak brightness.
            # In your CSV, we have MJD (obs date) and Age.
            # Peak Date (mjd_max) = MJD_obs - Age
            mjd = row['MJD']

            params[file_name] = {
                'z': float(row['redshift']),
                'mjd': float(mjd),
                'true_age': float(row['Age_(days)']), # Optional: keep for verification
                'redshift': float(row['redshift'])
            }
        except (ValueError, KeyError):
            # Skip rows with missing data (NaN) or formatting errors
            continue

    return params

def prior_transform(u, params, priors):
    """Transforming the uniform prior to the parameter space."""
    t = np.zeros_like(u)
    for i, p in enumerate(params):
        low, high = priors[p]
        t[i] = u[i] * (high - low) + low
    return t

_MODEL_CACHE = {}

def get_model(model_name):
    """Global model cache to prevent repeated object creation in parallel workers."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = sncosmo.Model(source=model_name)
    return _MODEL_CACHE[model_name]

def log_likelihood_nuisance(t, params, wavelength, flux, flux_err, mjd_obs, model_name, redshift):
    model = get_model(model_name)
    if redshift is not None:
        model.set(z=redshift)

    p_dict = dict(zip(params, t))
    model.set(**p_dict)
    try:
        # setting up t0 as an offset
        t0_offset = p_dict.pop('t0')
        t0_mjd = mjd_obs - (t0_offset * (1 + redshift))

        # model setup
        model.set(t0 = t0_mjd)
        model.set(**p_dict)
        model.set(x0=1.0)

        # setup the flux with a x0 of 1
        m_flux_unit = model.flux(mjd_obs, wavelength)

        # setup the weights for the fluxes
        w = 1.0 / flux_err ** 2

        # setting up the numerator and denominator of the weighted least squares minimization
        num, den = np.sum(flux * m_flux_unit * w), np.sum(m_flux_unit ** 2 * w)

        # finding the best x0 scaling parameter
        if den <= 0: return -1e10
        x0_best = num / den
        if x0_best <= 0: return -1e10

        chisq = np.sum(((flux - x0_best * m_flux_unit) / flux_err) ** 2)

        return -0.5 * chisq if not np.isnan(chisq) else -np.inf
    except Exception:
        return -np.inf

def fit_nuisance(wavelength, flux, flux_err, mjd_obs, model_name='salt3', redshift=None):
    if redshift is not None:
        params = ['t0', 'x1', 'c']
        priors = {'t0': (-50, 50), 'x1': (-6, 6), 'c': (-1.5, 1.5)}
    else:
        params = ['z', 't0', 'x1', 'c']
        priors = {'z': (0.005, 0.1), 't0': (-50, 20), 'x1': (-6, 6), 'c': (-1.5, 1.5)}

    pt = partial(prior_transform, params=params, priors=priors)
    ll = partial(log_likelihood_nuisance, params=params, wavelength=wavelength, flux=flux, flux_err=flux_err,
                 mjd_obs=mjd_obs, model_name=model_name, redshift=redshift)

    sampler = dynesty.NestedSampler(ll, pt, len(params), nlive=NLIVE, sample='rslice')
    sampler.run_nested(print_progress=False)
    return sampler.results, params


def process_single_file(filename, sn_params):
    file_path = os.path.join(SPECTRA_DIR, filename)

    p = sn_params[filename]

    z = p['z']
    mjd_obs = p['mjd']
    true_age = p['true_age']

    wave, flux, err = load_flm_spectrum(file_path)
    if len(wave) == 0: return None
    mask = (wave > 3500) & (wave < 8000)
    wave, flux, err = wave[mask], flux[mask], err[mask]
    if len(wave) < 10: return None

    res = {'filename': filename, 'true_age': true_age, 'z_true': p['z']}

    try:
        results, par_names = fit_nuisance(wave, flux, err, mjd_obs, redshift=z)

        # Use math.exp or fallback safe operations for weights
        log_wts = results.logwt - results.logz[-1]
        weights = np.exp(np.clip(log_wts, -300, 300))
        samples = dynesty.utils.resample_equal(results.samples, weights)

        res[f'logz'] = results.logz[-1]
        res[f'logzerr'] = results.logzerr[-1]
        res[f'ncall'] = results.ncall
        res[f'eff'] = results.eff

        mean_params = np.mean(samples, axis=0)
        p_dict_best = dict(zip(par_names, mean_params))

        t0_offset = p_dict_best.pop('t0')
        t0_mjd = mjd_obs - (t0_offset * (1 + z))

        model = get_model(MODEL_NAME)
        model.set(z=z)
        model.set(t0 = t0_mjd)
        model.set(**p_dict_best)
        model.set(x0=1.0)

        m_flux_unit = model.flux(mjd_obs, wave)
        w = 1.0 / err ** 2
        x0_best = np.sum(flux * m_flux_unit * w) / np.sum(m_flux_unit ** 2 * w)

        res['chi2'] = np.sum(((flux - x0_best * m_flux_unit) / err) ** 2)
        res['ndof'] = len(wave) - len(par_names)

        failed = False
        for i, name in enumerate(par_names):
            s = samples[:, i]
            mean, std = np.mean(s), np.std(s)

            median = np.median(s)
            q16, q84 = np.percentile(s, [16, 84])

            # Check for failed convergence (std dev too low)
            if name in ['t0', 'x1', 'c'] and std < STD_THRESHOLD:
                failed = True

            if name == 't0':
                res[f'age_fit'] = mean
                res[f'age_err'] = std
                res[f'age_median'] = median
                res[f'age_q16'] = q16
                res[f'age_q84'] = q84
            else:
                res[f'{name}_mean'] = mean
                res[f'{name}_std'] = std
                res[f'{name}_median'] = median
                res[f'{name}_q16'] = q16
                res[f'{name}_q84'] = q84

        res[f'failed'] = failed


        x0_list = []
        subset_indices = np.random.choice(len(samples), min(500, len(samples)), replace=False)
        subset_samples = samples[subset_indices]

        for sample in subset_samples:
            p_dict = dict(zip(par_names, sample))

            # Re-calculate absolute MJD for t0
            t0_offset = p_dict.pop('t0')
            t0_mjd = mjd_obs - (t0_offset * (1 + z))

            model.set(t0=t0_mjd)
            model.set(**p_dict)
            model.set(x0=1.0)

            m_flux_unit = model.flux(mjd_obs, wave)

            den = np.sum(m_flux_unit ** 2 * w)
            if den > 0:
                x0_val = np.sum(flux * m_flux_unit * w) / den
                if x0_val > 0:
                    x0_list.append(x0_val)

        if x0_list:
            x0_arr = np.array(x0_list)
            res[f'x0_mean'] = np.nanmean(x0_arr)
            res[f'x0_std'] = np.nanstd(x0_arr)
            res[f'x0_median'] = np.nanmedian(x0_arr)
        else:
            res[f'x0_mean'] = np.nan

    except Exception as e:
        # print(f"Error fitting {filename} with {method}: {e}")
        res[f'age_fit'] = np.nan
        res[f'failed'] = True

    return res


def run_test():
    os.makedirs("outputs/csvs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    output_csv = "outputs/csvs/allcfa_results.csv"

    sn_params = parse_param_csv(DATA_FILE)

    if not sn_params:
        print("No metadata loaded. Check your CSV path and column names.")
        return

    all_disk_files = [f for f in os.listdir(SPECTRA_DIR) if f.endswith(('.flm', '.dat'))]
    flm_files = [f for f in all_disk_files if f in sn_params]

    print(f"Found {len(flm_files)} spectra matching the metadata CSV.")

    # 4. Handle Incremental Processing (Skip already finished files)
    if os.path.exists(output_csv) and not FORCE_RERUN:
        existing_df = pd.read_csv(output_csv)
        processed_files = set(existing_df['filename'].unique())
        print(f"Incremental mode: {len(processed_files)} spectra already processed.")
        files_to_run = [f for f in flm_files if f not in processed_files]
    else:
        if FORCE_RERUN:
            print("FORCE_RERUN is True. Starting a fresh run.")
        existing_df = pd.DataFrame()
        files_to_run = flm_files

    if not files_to_run:
        print("No new files to process.")
        return

    # 5. Pre-load model to safely seed Astropy/Cache
    print(f"Pre-loading {MODEL_NAME} model...")
    _ = get_model(MODEL_NAME)

    # 6. Parallel Processing in Chunks
    # Chunking is a safety measure so you don't lose all data if the power goes out
    print(f"Processing {len(files_to_run)} files on {N_CORES} cores...")

    chunk_size = 50
    for i in range(0, len(files_to_run), chunk_size):
        chunk = files_to_run[i:i + chunk_size]
        print(f"\n--- Processing Chunk {i // chunk_size + 1} ({i} to {i + len(chunk)}) ---")

        # The Parallel call: passes the filename and the master dictionary
        results_raw = Parallel(n_jobs=N_CORES)(
            delayed(process_single_file)(f, sn_params) for f in chunk
        )

        # 7. Collect and Save Results
        new_results = [r for r in results_raw if r is not None]
        if new_results:
            new_df = pd.DataFrame(new_results)
            if not existing_df.empty:
                existing_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                existing_df = new_df

            # Save after every chunk
            existing_df.to_csv(output_csv, index=False)
            print(f"Checkpoint saved: {len(existing_df)} total spectra in {output_csv}")

    print(f"\nAll processing complete. Final results: {output_csv}")


if __name__ == "__main__":
    run_test()
