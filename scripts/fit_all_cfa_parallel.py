import os
import sys
import numpy as np
import pandas as pd
import sncosmo
from scipy.optimize import minimize
import dynesty
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

SPECTRA_DIR = '/Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/data/all_spectra'
OUTPUT_CSV = '/Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/data/cfa_fits_results.csv'

def fit_single_spectrum(row):
    filename = row['Filename']
    t_spec = row['MJD']
    z = row['redshift']
    sn_name = row['SN_Name']
    true_age = row['Age_(days)']
    
    try:
        path = os.path.join(SPECTRA_DIR, filename)
        if not os.path.exists(path):
            return {
                'Filename': filename, 'SN_Name': sn_name, 'redshift': z, 'Age_(days)': true_age, 'MJD': t_spec,
                'Status': 'Failed', 'Error': f'File not found: {filename}',
                't0': np.nan, 't0_err': np.nan, 'x1': np.nan, 'x1_err': np.nan, 'c': np.nan, 'c_err': np.nan, 'chi2_red': np.nan
            }
            
        data = np.genfromtxt(path)
        wave_obs, flux = data[:, 0], data[:, 1]
        
        # Filter observed range (rest-frame 2000 to 9200 AA)
        mask = (wave_obs >= 2000.0 * (1.0 + z)) & (wave_obs <= 9200.0 * (1.0 + z))
        if np.sum(mask) < 10:
            return {
                'Filename': filename, 'SN_Name': sn_name, 'redshift': z, 'Age_(days)': true_age, 'MJD': t_spec,
                'Status': 'Failed', 'Error': 'Too few wavelength points in range',
                't0': np.nan, 't0_err': np.nan, 'x1': np.nan, 'x1_err': np.nan, 'c': np.nan, 'c_err': np.nan, 'chi2_red': np.nan
            }
        wave_f, flux_f = wave_obs[mask], flux[mask]
        
        # Estimate noise floor using MAD of adjacent differences
        diffs = np.diff(flux_f)
        mad = np.median(np.abs(diffs - np.median(diffs)))
        noise_floor = 1.4826 * mad / np.sqrt(2)
        if noise_floor <= 0:
            noise_floor = 0.1 * np.mean(np.abs(flux_f))
        
        # Quadrature sum of background noise and 5% shot/systematic noise
        err_f = np.sqrt(noise_floor**2 + (0.05 * np.maximum(0, flux_f))**2)
        
        # Initialize SALT3 model inside the worker to prevent pickling errors
        model = sncosmo.Model(source='salt3')
        model.set(z=z)
        
        # Optimization helper to find minimum chisq and best-fit scale factor
        def loss(p):
            t0, x1, c = p
            model.set(t0=t0, x1=x1, c=c)
            try:
                m_flux_unit = model.flux(t_spec, wave_f)
            except:
                return 1e10
            w = 1.0 / err_f**2
            num = np.sum(flux_f * m_flux_unit * w)
            den = np.sum(m_flux_unit**2 * w)
            if den <= 0 or num <= 0:
                return 1e10
            x0_best = num / den
            chisq = np.sum(((flux_f - x0_best * m_flux_unit) / err_f)**2)
            return chisq
            
        bounds = [
            (t_spec - 50.0 * (1.0 + z), t_spec + 20.0 * (1.0 + z)),
            (-3.0, 3.0),
            (-0.3, 1.0)
        ]
        
        # Perform optimization to scale the log-likelihood
        res = minimize(loss, [t_spec, 0.0, 0.0], bounds=bounds, method='L-BFGS-B')
        best_chisq = res.fun
        ndof = len(wave_f) - 3 - 1  # 4 free parameters (x0, t0, x1, c)
        chi2_red = best_chisq / ndof if ndof > 0 else 1.0
        if chi2_red <= 0 or np.isnan(chi2_red):
            chi2_red = 1.0
            
        params = ['t0', 'x1', 'c']
        priors = {
            't0': (t_spec - 50.0 * (1.0 + z), t_spec + 20.0 * (1.0 + z)),
            'x1': (-3.0, 3.0),
            'c': (-0.3, 1.0)
        }
        
        def pt(u):
            t = np.zeros_like(u)
            for i, p in enumerate(params):
                low, high = priors[p]
                t[i] = u[i] * (high - low) + low
            return t
            
        def ll(t):
            t0, x1, c = t
            model.set(t0=t0, x1=x1, c=c)
            try:
                m_flux_unit = model.flux(t_spec, wave_f)
            except:
                return -1e10
            w = 1.0 / err_f**2
            num = np.sum(flux_f * m_flux_unit * w)
            den = np.sum(m_flux_unit**2 * w)
            if den <= 0 or num <= 0:
                return -1e10
            x0_best = num / den
            chisq = np.sum(((flux_f - x0_best * m_flux_unit) / err_f)**2)
            loglike = -0.5 * chisq / chi2_red
            return loglike if not np.isnan(loglike) else -1e10

        sampler = dynesty.NestedSampler(ll, pt, len(params), nlive=80, sample='rwalk')
        sampler.run_nested(print_progress=False)
        dyn_res = sampler.results
        w = np.exp(dyn_res.logwt - dyn_res.logz[-1])
        samples = dynesty.utils.resample_equal(dyn_res.samples, w)
        
        t0_post = np.percentile(samples[:, 0], [16, 50, 84])
        x1_post = np.percentile(samples[:, 1], [16, 50, 84])
        c_post = np.percentile(samples[:, 2], [16, 50, 84])
        
        return {
            'Filename': filename, 'SN_Name': sn_name, 'redshift': z, 'Age_(days)': true_age, 'MJD': t_spec,
            'Status': 'Success', 'Error': '',
            't0': t0_post[1],
            't0_err': 0.5 * (t0_post[2] - t0_post[0]),
            'x1': x1_post[1],
            'x1_err': 0.5 * (x1_post[2] - x1_post[0]),
            'c': c_post[1],
            'c_err': 0.5 * (c_post[2] - c_post[0]),
            'chi2_red': chi2_red
        }
    except Exception as e:
        return {
            'Filename': filename, 'SN_Name': sn_name, 'redshift': z, 'Age_(days)': true_age, 'MJD': t_spec,
            'Status': 'Failed', 'Error': str(e),
            't0': np.nan, 't0_err': np.nan, 'x1': np.nan, 'x1_err': np.nan, 'c': np.nan, 'c_err': np.nan, 'chi2_red': np.nan
        }

if __name__ == '__main__':
    # Load metadata
    df_prop = pd.read_csv('/Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/data/spectra_properties.csv')
    df_mjd = pd.read_csv('/Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/data/cfasnIa_mjdspec.dat', sep='\\s+', comment='#', names=['Filename', 'MJD'])
    df_joined = pd.merge(df_prop, df_mjd, on='Filename')
    
    # Filter rest-frame phase range [-12, 35] days
    mask = (df_joined['Age_(days)'] >= -12.0) & (df_joined['Age_(days)'] <= 35.0)
    df_target = df_joined[mask]
    
    # Filter completed ones if they already exist
    completed_filenames = set()
    write_header = True
    if os.path.exists(OUTPUT_CSV):
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            completed_filenames = set(df_existing['Filename'].tolist())
            write_header = len(completed_filenames) == 0
            print(f"Found existing output file with {len(completed_filenames)} entries.")
        except Exception as e:
            print(f"Error reading existing CSV: {e}. Starting fresh.")
            
    df_todo = df_target[~df_target['Filename'].isin(completed_filenames)]
    total_todo = len(df_todo)
    print(f"Total spectra matching filter (-12 <= Age <= 35): {len(df_target)}")
    print(f"Already completed: {len(completed_filenames)}")
    print(f"Remaining to fit: {total_todo}")
    
    if total_todo == 0:
        print("All matching spectra have already been fitted!")
        sys.exit(0)
        
    rows_list = df_todo.to_dict('records')
    
    # Determine number of workers
    n_workers = max(1, os.cpu_count() - 1)
    print(f"Starting fitting with {n_workers} processes...")
    
    start_time = time.time()
    count = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(fit_single_spectrum, r): r for r in rows_list}
        
        for future in as_completed(futures):
            res = future.result()
            count += 1
            
            # Save incrementally
            df_res = pd.DataFrame([res])
            df_res.to_csv(OUTPUT_CSV, mode='a', index=False, header=write_header)
            write_header = False  # Only write header once
            
            # Progress update
            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 else 0
            eta = (total_todo - count) / rate if rate > 0 else 0
            
            if count % 10 == 0 or count == total_todo:
                print(f"Progress: {count}/{total_todo} | Rate: {rate:.2f} spec/s | Elapsed: {elapsed/60:.2f} min | ETA: {eta/60:.2f} min | Last: {res['Filename']} ({res['Status']})")
                sys.stdout.flush()
                
    print(f"Finished fitting all spectra in {(time.time() - start_time)/60:.2f} minutes.")
