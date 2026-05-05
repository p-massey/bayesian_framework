import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Use the filtered results from the main comparison script
MERGED_FILE = 'outputs/method_comparison/filtered_comparison_results.csv'
OUTPUT_DIR = 'outputs/method_comparison'
STYLE_FILE = 'assets/plotting_style.mplstyle'

# Colorblind-friendly colors (Okabe-Ito palette)
CB_VERMILION = '#D55E00'
CB_BLUE = '#0072B2'

def run_binned_comparison():
    # 1. Load Data
    if not os.path.exists(MERGED_FILE):
        print(f"Error: {MERGED_FILE} not found. Run compare_methods.py first.")
        return
        
    df = pd.read_csv(MERGED_FILE)
    
    # 2. Define Bins
    # Covering -15 to 25 days (to match current cuts)
    bin_edges = np.arange(-15, 26, 5)
    df['age_bin'] = pd.cut(df['true_age'], bins=bin_edges)
    
    # 3. Calculate Binned Statistics
    df['bias_dyn'] = df['nuis_age'] - df['true_age']
    df['bias_snid'] = df['bootstrap_age'] - df['true_age']
    
    bin_stats = df.groupby('age_bin', observed=True).agg({
        'bias_dyn': ['mean', 'std', 'count'],
        'bias_snid': ['mean', 'std', 'count'],
        'true_age': 'mean'
    }).reset_index()
    
    # Flatten multi-index columns
    bin_stats.columns = ['age_bin', 'bias_dyn_mean', 'bias_dyn_std', 'count_dyn', 
                         'bias_snid_mean', 'bias_snid_std', 'count_snid', 'bin_center_mean']
    
    # Filter out bins with too few points
    bin_stats = bin_stats[bin_stats['count_dyn'] > 2].copy()

    # 4. Plotting
    if os.path.exists(STYLE_FILE):
        plt.style.use(STYLE_FILE)
    else:
        plt.style.use('seaborn-v0_8-whitegrid')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Subplot 1: Mean Bias
    ax1.errorbar(
        bin_stats['bin_center_mean'], bin_stats['bias_dyn_mean'], 
        yerr=bin_stats['bias_dyn_std'] / np.sqrt(bin_stats['count_dyn']), # SEM
        fmt='o-', color='red', label='Dynesty (Nuisance)', capsize=3, lw=2, markersize=6
    )
    ax1.errorbar(
        bin_stats['bin_center_mean'], bin_stats['bias_snid_mean'], 
        yerr=bin_stats['bias_snid_std'] / np.sqrt(bin_stats['count_snid']), 
        fmt='s-', color='blue', label='SNID (Bootstrap)', capsize=3, lw=2, markersize=6
    )
    
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Mean Bias (days)')
    ax1.set_title('Binned Performance: Dynesty vs SNID (5-day bins)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Dispersion (Sigma_t)
    ax2.plot(
        bin_stats['bin_center_mean'], bin_stats['bias_dyn_std'], 
        'o-', color='red', label='Dynesty $\sigma_t$', lw=2, markersize=6
    )
    ax2.plot(
        bin_stats['bin_center_mean'], bin_stats['bias_snid_std'], 
        's-', color='blue', label='SNID $\sigma_t$', lw=2, markersize=6
    )
    
    ax2.set_xlabel('True Age (days)')
    ax2.set_ylabel('Dispersion $\sigma_t$ (days)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 12) # Reasonable range for dispersion

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'binned_method_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Binned comparison plot saved to {plot_path}")

    # Also save the binned stats to CSV
    stats_path = os.path.join(OUTPUT_DIR, 'binned_comparison_stats.csv')
    bin_stats.to_csv(stats_path, index=False)
    print(f"Binned statistics saved to {stats_path}")

if __name__ == "__main__":
    run_binned_comparison()
