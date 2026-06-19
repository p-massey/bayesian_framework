import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = '/Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/data/cfa_fits_results.csv'
STYLE_PATH = '/Users/pxm588@student.bham.ac.uk/PhD/cfa_spectra_pipeline/src/GausSN.mplstyle'
OUTPUT_PNG = '/Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/outputs/cfa_residuals_vs_estimated_phase.png'

# Set style
if os.path.exists(STYLE_PATH):
    plt.style.use(STYLE_PATH)
    print(f"Loaded GausSN style from {STYLE_PATH}")
else:
    plt.style.use('default')
    print("Using default style")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Results CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    df_success = df[df['Status'] == 'Success'].copy()
    
    # Compute estimated phase and difference
    df_success['phase_est'] = (df_success['MJD'] - df_success['t0']) / (1.0 + df_success['redshift'])
    df_success['phase_diff'] = df_success['phase_est'] - df_success['Age_(days)']
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of individual epochs
    plt.scatter(df_success['phase_est'], df_success['phase_diff'], 
                color='#2b5c8f', alpha=0.4, s=20, label='Individual Epochs')
    
    plt.axhline(0, color='black', linestyle='-', lw=1.5, alpha=0.5)
    plt.axhline(1, color='gray', linestyle='--', lw=1, alpha=0.5)
    plt.axhline(-1, color='gray', linestyle='--', lw=1, alpha=0.5)
    
    # Compute binned running median/mean using estimated phase
    bins = np.arange(-12, 36, 4)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    binned_mean = []
    binned_std = []
    
    for i in range(len(bins)-1):
        mask_bin = (df_success['phase_est'] >= bins[i]) & (df_success['phase_est'] < bins[i+1])
        if mask_bin.sum() > 2:
            median_val = df_success.loc[mask_bin, 'phase_diff'].median()
            binned_mean.append(median_val)
            # robust scatter
            mad_bin = np.median(np.abs(df_success.loc[mask_bin, 'phase_diff'] - median_val))
            binned_std.append(1.4826 * mad_bin)
        else:
            binned_mean.append(np.nan)
            binned_std.append(np.nan)
            
    plt.errorbar(bin_centers, binned_mean, yerr=binned_std, fmt='s-', color='#e05a47', 
                 capsize=4, elinewidth=2, capthick=2, label='Running Binned Median (robust $\\sigma$)')
    
    plt.xlabel('Derived Spectroscopic Restframe Phase (days)', fontsize=12)
    plt.ylabel('Residual: Spectroscopic - Photometric Phase (days)', fontsize=12)
    plt.title('Phase residual trends as a function of derived phase', fontsize=14, pad=15)
    plt.ylim(-10, 10)
    plt.xlim(-15, 38)
    plt.legend(loc='lower left', fontsize=11)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()
    print(f"Saved residuals vs. estimated phase plot to {OUTPUT_PNG}")

if __name__ == '__main__':
    main()
