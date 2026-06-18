import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = '/Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/data/cfa_fits_results.csv'
STYLE_PATH = '/Users/pxm588@student.bham.ac.uk/PhD/cfa_spectra_pipeline/src/GausSN.mplstyle'
OUTPUT_DIR = '/Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/outputs'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
if os.path.exists(STYLE_PATH):
    plt.style.use(STYLE_PATH)
    print(f"Loaded GausSN style from {STYLE_PATH}")
else:
    plt.style.use('default')
    print("Using default matplotlib style")

def make_plot(df_plot, out_name, title_suffix=""):
    bin_width = 3.0
    bins = np.arange(-15, 30, bin_width)
    bin_centers = bins[:-1] + bin_width / 2
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))
    
    color = '#2563eb' # Use blue to match specsim's visual style or a nice custom theme
    
    # 1. Left Panel: Estimated vs True Age
    ax_left.errorbar(df_plot['Age_(days)'], df_plot['phase_est'], yerr=df_plot['phase_est_err'], 
                     fmt='o', color=color, ecolor='grey', elinewidth=0.4, capsize=0, 
                     markersize=3, alpha=0.3, label='Nested Sampling')
    
    # 1:1 line
    x_range = np.linspace(-15, 25, 100)
    ax_left.plot(x_range, x_range, color='#E25A4A', linestyle='--', lw=1.5, label='1:1 Line')
    
    ax_left.set_xlim(-15, 25)
    ax_left.set_ylim(-20, 30)
    ax_left.grid(True, linestyle='--', alpha=0.4)
    ax_left.set_xlabel(r'$t_{\mathrm{LC}}$ [days]', fontsize=16)
    ax_left.set_ylabel(r'$t_{\mathrm{est}}$ [days]', fontsize=16)
    ax_left.tick_params(axis='both', which='major', labelsize=12)
    ax_left.legend(loc='upper left', fontsize=12)
    ax_left.set_title(f"Nested Sampling Phase Determination{title_suffix}", fontsize=14, pad=10)
    
    # 2. Right Panel: Residuals vs True Age (with error bars)
    ax_right.errorbar(df_plot['Age_(days)'], df_plot['Residual'], yerr=df_plot['phase_est_err'], 
                      fmt='o', color=color, ecolor='grey', elinewidth=0.4, capsize=0, 
                      markersize=3, alpha=0.25)
    ax_right.axhline(0, color='black', linestyle='--', alpha=0.5, lw=1.5)
    
    # Running median and MAD
    running_med = []
    running_mad = []
    for b_start, b_end in zip(bins[:-1], bins[1:]):
        mask = (df_plot['Age_(days)'] >= b_start) & (df_plot['Age_(days)'] < b_end)
        bin_data = df_plot.loc[mask, 'Residual']
        if len(bin_data) >= 5:
            running_med.append(np.median(bin_data))
            running_mad.append(np.median(np.abs(bin_data - np.median(bin_data))))
        else:
            running_med.append(np.nan)
            running_mad.append(np.nan)
            
    ax_right.plot(bin_centers, running_med, color='#E25A4A', lw=2.5, label='Running Median')
    ax_right.fill_between(bin_centers, np.array(running_med) - np.array(running_mad), 
                         np.array(running_med) + np.array(running_mad), 
                         color='#E25A4A', alpha=0.15, label='Running MAD')
    
    ax_right.set_xlim(-15, 25)
    ax_right.set_ylim(-15, 15)
    ax_right.grid(True, linestyle='--', alpha=0.4)
    ax_right.set_xlabel(r'$t_{\mathrm{LC}}$ [days]', fontsize=16)
    ax_right.set_ylabel(r'Residual [days]', fontsize=16)
    ax_right.tick_params(axis='both', which='major', labelsize=12)
    ax_right.legend(loc='lower left', fontsize=12)
    ax_right.set_title("Phase Residuals vs. True Age", fontsize=14, pad=10)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot to {plot_path}")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Results CSV not found at {CSV_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(CSV_PATH)
    df_success = df[df['Status'] == 'Success'].copy()
    
    # Compute estimated phase and residuals
    df_success['phase_est'] = (df_success['MJD'] - df_success['t0']) / (1.0 + df_success['redshift'])
    df_success['phase_est_err'] = df_success['t0_err'] / (1.0 + df_success['redshift'])
    df_success['Residual'] = df_success['phase_est'] - df_success['Age_(days)']
    
    print(f"Total successful fits: {len(df_success)}")
    
    # Plot 1: All results (No quality cuts)
    make_plot(df_success, 'nested_sampling_comparison_raw.png', title_suffix=" (Raw)")
    
    # Plot 2: Results with quality cuts
    df_clean = df_success[(df_success['chi2_red'] <= 10.0) & (df_success['t0_err'] <= 1.0)].copy()
    print(f"Successful fits after quality cuts (chi2_red <= 10, t0_err <= 1.0): {len(df_clean)}")
    make_plot(df_clean, 'nested_sampling_comparison_clean.png', title_suffix=" (Quality Cuts)")

if __name__ == '__main__':
    main()
