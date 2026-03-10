import pandas as pd
import numpy as np
import os

def load_dm15_data(dat_file):
    dm15_map = {}
    with open(dat_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6:
                sn_name = parts[0].strip().lower()
                try:
                    dm15 = float(parts[5])
                    # 9.99 often means no estimate in these datasets
                    if dm15 == 9.99:
                        dm15 = np.nan
                    dm15_map[sn_name] = dm15
                except ValueError:
                    continue
    return dm15_map

def update_csv(csv_file, dm15_map):
    df = pd.read_csv(csv_file)
    
    # Ensure SN_Name is string
    df['SN_Name'] = df['SN_Name'].astype(str)
    
    # Map dm15 values
    # Some SN names might have 'sn' prefix in the CSV but not in the DAT, or vice versa.
    # Let's handle both.
    def get_dm15(sn):
        sn_lower = sn.lower()
        if sn_lower in dm15_map:
            return dm15_map[sn_lower]
        if sn_lower.startswith('sn') and sn_lower[2:] in dm15_map:
            return dm15_map[sn_lower[2:]]
        if f"sn{sn_lower}" in dm15_map:
            return dm15_map[f"sn{sn_lower}"]
        return np.nan

    df['dm15'] = df['SN_Name'].apply(get_dm15)
    
    # Count how many were found
    found_count = df['dm15'].notna().sum()
    print(f"Added dm15 values for {found_count} out of {len(df)} rows.")
    
    output_file = csv_file # Overwrite or new file? Usually user wants to update.
    df.to_csv(output_file, index=False)
    print(f"Updated {output_file}")

if __name__ == "__main__":
    dat_path = 'cfasnIa_param.dat'
    csv_path = 'spectra_properties.csv'
    
    if os.path.exists(dat_path) and os.path.exists(csv_path):
        dm15_values = load_dm15_data(dat_path)
        update_csv(csv_path, dm15_values)
    else:
        print("Missing required files.")
