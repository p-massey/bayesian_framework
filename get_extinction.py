import pandas as pd
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
import sfdmap
import numpy as np
import os
import time

# --- Setup ---
# Ensure necessary libraries are installed:
# pip install astroquery pandas astropy sfdmap numpy

# Path to the cfasnIa_param.dat file
CFA_PARAM_FILE = 'cfasnIa_param.dat'
OUTPUT_FILE = 'supernova_extinction_data.csv'

# Initialize SIMBAD and sfdmap
Simbad.add_votable_fields('ra', 'dec') # Request RA and Dec in degrees
sfd = sfdmap.SFDMap()

def get_sn_names(file_path):
    """
    Parses the cfasnIa_param.dat file to extract unique supernova names.
    """
    sn_names = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # SN name is the first column
            parts = line.split()
            if parts:
                sn_names.add(parts[0])
    return sorted(list(sn_names))

def query_simbad_and_sfd(sn_name):
    """
    Queries SIMBAD for supernova coordinates and then uses sfdmap for galactic extinction.
    """
    ra, dec, ebv = np.nan, np.nan, np.nan
    candidate_simbad_ids = []
    if sn_name.startswith('SNF'):
        candidate_simbad_ids.append(sn_name)
    else:
        candidate_simbad_ids.append('sn'+ sn_name)

    for simbad_id_to_try in candidate_simbad_ids:
        try:
            result_table = Simbad.query_object(simbad_id_to_try)
            if result_table is not None and len(result_table) > 0:
                ra_deg = result_table['ra'][0]
                dec_deg = result_table['dec'][0]
                
                coords = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
                ra, dec = coords.ra.deg, coords.dec.deg

                ebv = sfd.ebv(ra, dec, scale_by_r_v=False)
                
                print(f"Found {sn_name} using SIMBAD ID '{simbad_id_to_try}': RA={ra:.4f}, Dec={dec:.4f}, E(B-V)={ebv:.4f}")
                return ra, dec, ebv # Success!
            else:
                print(f"Could not find {sn_name} using SIMBAD ID '{simbad_id_to_try}'.")
        except Exception as e:
            print(f"Error querying SIMBAD for {sn_name} using SIMBAD ID '{simbad_id_to_try}': {e}")
        
    return ra, dec, ebv # If nothing found after all attempts

def main():
    print("Parsing supernova names...")
    sn_names = get_sn_names(CFA_PARAM_FILE)
    print(f"Found {len(sn_names)} unique supernova names.")

    results = []
    # Add a small delay between queries to be respectful of public services
    query_delay = 1 # seconds

    for i, sn_name in enumerate(sn_names):
        print(f"Querying {i+1}/{len(sn_names)}: {sn_name}...")
        ra, dec, ebv = query_simbad_and_sfd(sn_name)
        results.append({'SN_Name': sn_name, 'RA': ra, 'DEC': dec, 'E_B_V_SFD': ebv})
        time.sleep(query_delay) # Wait before next query

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")

    # Display a summary of the results
    print("--- Summary ---")
    print(f"Total SNe processed: {len(df)}")
    print(f"SNe with coordinates and E(B-V) data: {df['E_B_V_SFD'].notna().sum()}")
    print(f"SNe without coordinates or E(B-V) data: {df['E_B_V_SFD'].isna().sum()}")

if __name__ == '__main__':
    main()

