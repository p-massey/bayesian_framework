import pandas as pd
import numpy as np
np.int=int
import sfdmap
import extinction
import os
import glob
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
import time

# --- Configuration ---
INPUT_DIR = 'all_spectra'
OUTPUT_DIR = 'all_spectra_dereddened'
PARAM_FILE = 'cfasnIa_param.dat'
EXTINCTION_CACHE_FILE = 'supernova_extinction_data.csv'
SFD_DATA_DIR = 'sfddata'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize SFD Map
sfd = sfdmap.SFDMap(SFD_DATA_DIR)
Simbad.add_votable_fields('ra', 'dec')

def get_sn_list():
    """Extract SN names from filenames in INPUT_DIR."""
    files = glob.glob(os.path.join(INPUT_DIR, '*.flm'))
    sn_names = set()
    for f in files:
        basename = os.path.basename(f)
        if basename.startswith('snf'):
            parts = basename.split('-')
            sn_name = "-".join(parts[:2])
        else:
            sn_name = basename.split('-')[0]
        sn_names.add(sn_name)
    return sorted(list(sn_names))

def query_extinction(sn_name):
    """Query SIMBAD for coords and get EBV from SFD."""
    # Try with 'sn' prefix first as it's common in SIMBAD
    simbad_id = sn_name
    try:
        result_table = Simbad.query_object(simbad_id)
        if result_table is not None and len(result_table) > 0:
            ra_deg = result_table['ra'][0]
            dec_deg = result_table['dec'][0]
            coords = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
            ebv = sfd.ebv(coords.ra.deg, coords.dec.deg, scale_by_r_v=False)
            return ebv
    except Exception as e:
        print(f"Error querying {sn_name}: {e}")
    return np.nan

def load_or_create_extinction_map(sn_list):
    if os.path.exists(EXTINCTION_CACHE_FILE):
        print(f"Loading extinction cache from {EXTINCTION_CACHE_FILE}...")
        df = pd.read_csv(EXTINCTION_CACHE_FILE)
        # Create a dictionary for quick lookup
        return df.set_index('SN_Name')['E_B_V_SFD'].to_dict()
    
    print("Creating extinction mapping (this may take a while due to SIMBAD queries)...")
    mapping = {}
    for i, sn in enumerate(sn_list):
        print(f"[{i+1}/{len(sn_list)}] Querying {sn}...")
        ebv = query_extinction(sn)
        mapping[sn] = ebv
        time.sleep(0.5) # Be kind to SIMBAD
    
    # Save cache
    df = pd.DataFrame(list(mapping.items()), columns=['SN_Name', 'E_B_V_SFD'])
    df.to_csv(EXTINCTION_CACHE_FILE, index=False)
    return mapping

def deredden_spectrum(file_path, ebv, output_path):
    if np.isnan(ebv):
        # If no extinction data, just copy the file or skip? 
        # For now, let's copy it so the output directory is complete, but print a warning.
        print(f"Warning: No E(B-V) for {file_path}. Copying as is.")
        import shutil
        shutil.copy2(file_path, output_path)
        return

    try:
        # Load spectrum
        data = pd.read_csv(file_path, sep=r"\s+", header=None, comment='#')
        wave = data.iloc[:, 0].values.astype(float)
        flux = data.iloc[:, 1].values.astype(float)
        
        # Dereddening calculation (Rv = 3.1)
        av = 3.1 * ebv
        a_lambda = extinction.fitzpatrick99(wave, av, 3.1)
        deredden_factor = 10**(0.4 * a_lambda)
        
        # Apply factor
        data.iloc[:, 1] = flux * deredden_factor
        
        # Apply to error column if it exists
        if data.shape[1] > 2:
            errors = pd.to_numeric(data.iloc[:, 2], errors='coerce')
            if not errors.isna().all():
                data.iloc[:, 2] = errors.values * deredden_factor
        
        # Save to output directory
        data.to_csv(output_path, sep=' ', header=False, index=False, float_format='%.6e')
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    sn_list = get_sn_list()
    print(f"Found {len(sn_list)} unique Supernovae in {INPUT_DIR}.")
    
    ext_map = load_or_create_extinction_map(sn_list)
    
    spectrum_files = glob.glob(os.path.join(INPUT_DIR, '*.flm'))
    print(f"Processing {len(spectrum_files)} spectra...")
    
    for i, f in enumerate(spectrum_files):
        basename = os.path.basename(f)
        if basename.startswith('snf'):
            parts = basename.split('-')
            sn_name = "-".join(parts[:2])
        else:
            sn_name = basename.split('-')[0]
            
        ebv = ext_map.get(sn_name, np.nan)
        output_path = os.path.join(OUTPUT_DIR, basename)
        
        if i % 100 == 0:
            print(f"Progress: {i}/{len(spectrum_files)}...")
            
        deredden_spectrum(f, ebv, output_path)

    print(f"Done! Dereddened spectra are in {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
