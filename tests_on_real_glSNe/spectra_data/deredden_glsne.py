import pandas as pd
import numpy as np
np.int = int
import sfdmap
import extinction
import os
import glob
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
import time
import shutil

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFD_DATA_DIR = os.path.join(SCRIPT_DIR, '../../sfddata')
BASE_DIR = SCRIPT_DIR

# Mapping SN Names to their raw data directories
SN_DIRECTORIES = {
    'sn2016geu': '16geu_raw_data',
    'sn h0pe': 'H0pe_raw_data',
    'SN Encore': 'encore_raw_data'
}

# Ensure output directories exist
for sn_name, raw_dir in SN_DIRECTORIES.items():
    output_dir = os.path.join(BASE_DIR, raw_dir.replace('raw_data', 'dereddened'))
    os.makedirs(output_dir, exist_ok=True)

# Initialize SFD Map
sfd = sfdmap.SFDMap(SFD_DATA_DIR)
Simbad.add_votable_fields('ra', 'dec')

def query_extinction(sn_name):
    """Query SIMBAD for coords and get EBV from SFD."""
    try:
        result_table = Simbad.query_object(sn_name)
        if result_table is not None and len(result_table) > 0:
            ra_deg = result_table['ra'][0]
            dec_deg = result_table['dec'][0]
            # Coordinates might be in deg or HMS/DMS, astroquery handles both usually
            coords = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg) if isinstance(ra_deg, (float, np.float64)) else (u.hourangle, u.deg), frame='icrs')
            ebv = sfd.ebv(coords.ra.deg, coords.dec.deg, scale_by_r_v=False)
            return ebv
    except Exception as e:
        print(f"Error querying {sn_name}: {e}")
    return np.nan

def deredden_spectrum(file_path, ebv, output_path):
    if np.isnan(ebv):
        print(f"Warning: No E(B-V) for {file_path}. Copying as is.")
        shutil.copy2(file_path, output_path)
        return

    try:
        # Determine file format and separator
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
            # Find which column is WAVE and which is FLUX (case insensitive)
            cols = [c.upper() for c in data.columns]
            wave_idx = cols.index('WAVE') if 'WAVE' in cols else 0
            flux_idx = cols.index('FLUX') if 'FLUX' in cols else 1
            err_idx = cols.index('ERR') if 'ERR' in cols else (cols.index('ERROR') if 'ERROR' in cols else None)
            
            wave = data.iloc[:, wave_idx].values.astype(float)
            flux = data.iloc[:, flux_idx].values.astype(float)
        else:
            # Assume whitespace separated, no header
            data = pd.read_csv(file_path, sep=r"\s+", header=None, comment='#')
            wave = data.iloc[:, 0].values.astype(float)
            flux = data.iloc[:, 1].values.astype(float)
            err_idx = 2 if data.shape[1] > 2 else None

        # Dereddening calculation (Rv = 3.1)
        av = 3.1 * ebv
        a_lambda = extinction.fitzpatrick99(wave, av, 3.1)
        deredden_factor = 10**(0.4 * a_lambda)
        
        # Apply factor
        if file_path.endswith('.csv'):
            data.iloc[:, flux_idx] = flux * deredden_factor
            if err_idx is not None:
                data.iloc[:, err_idx] = data.iloc[:, err_idx].values.astype(float) * deredden_factor
        else:
            data.iloc[:, 1] = flux * deredden_factor
            if err_idx is not None:
                data.iloc[:, err_idx] = data.iloc[:, err_idx].values.astype(float) * deredden_factor
        
        # Save to output directory
        if file_path.endswith('.csv'):
            data.to_csv(output_path, index=False)
        else:
            data.to_csv(output_path, sep=' ', header=False, index=False, float_format='%.6e')
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    print("Obtaining E(B-V) for glSNe from SIMBAD + SFD Map...")
    ext_map = {}
    for sn_name in SN_DIRECTORIES.keys():
        ebv = query_extinction(sn_name)
        ext_map[sn_name] = ebv
        print(f"  {sn_name}: E(B-V) = {ebv:.4f}")
        time.sleep(0.5)

    for sn_name, raw_dir_name in SN_DIRECTORIES.items():
        raw_dir = os.path.join(BASE_DIR, raw_dir_name)
        output_dir = os.path.join(BASE_DIR, raw_dir_name.replace('raw_data', 'dereddened'))
        
        ebv = ext_map[sn_name]
        print(f"\nProcessing {sn_name} spectra from {raw_dir}...")
        
        # Get all relevant files
        spectrum_files = []
        for ext in ['*.flm', '*.dat', '*.txt', '*.ascii', '*.csv']:
            spectrum_files.extend(glob.glob(os.path.join(raw_dir, ext)))
        
        # Exclude metadata files if any
        spectrum_files = [f for f in spectrum_files if 'wiserep_spectra.csv' not in f]

        for i, f in enumerate(spectrum_files):
            basename = os.path.basename(f)
            output_path = os.path.join(output_dir, basename)
            deredden_spectrum(f, ebv, output_path)
            if (i+1) % 10 == 0 or (i+1) == len(spectrum_files):
                print(f"  [{i+1}/{len(spectrum_files)}] {basename}")

    print("\nDone! Dereddened glSNe spectra are in their respective '_dereddened' folders.")

if __name__ == '__main__':
    main()
