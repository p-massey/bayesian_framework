import pandas as pd
import numpy as np

def calculate_residuals(file_path):
    df = pd.read_csv(file_path)
    
    # Calculate delta (residual) = bootstrap_age - Age_(days)
    df['delta_age'] = df['bootstrap_age'] - df['Age_(days)']
    
    # Calculate uncertainty = sqrt(snid_std_dev^2 + Age_Unc_(days)^2)
    df['delta_age_unc'] = np.sqrt(df['snid_std_dev']**2 + df['Age_Unc_(days)']**2)
    
    # Save back to the same file
    df.to_csv(file_path, index=False)
    print(f"Calculated residuals and uncertainties. Saved to {file_path}")
    print(df[['SN_Name', 'Age_(days)', 'bootstrap_age', 'delta_age', 'delta_age_unc']].head())

if __name__ == "__main__":
    target_file = 'all_spectra_dereddened_analysis/all_spectra_found_dataset.csv'
    calculate_residuals(target_file)
