import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Configuration
DYNESTY_FILE = 'outputs/csvs/allcfa_results_filtered.csv'
SNID_FILE = 'outputs/method_comparison/cfa_SNID_results.csv'
OUTPUT_DIR = 'outputs/method_comparison'
STYLE_FILE = 'assets/plotting_style.mplstyle'

def run_comparison():
    # 1. Load Data
    if not os.path.exists(DYNESTY_FILE):
        # Fallback to unfiltered if filtered doesn't exist
        DYNESTY_FILE_ALT = 'outputs/csvs/allcfa_results.csv'
        if not os.path.exists(DYNESTY_FILE_ALT):
            print(f"Error: Neither {DYNESTY_FILE} nor {DYNESTY_FILE_ALT} found.")
            return
        df_dyn = pd.read_csv(DYNESTY_FILE_ALT)
    else:
        df_dyn = pd.read_csv(DYNESTY_FILE)

    if not os.path.exists(SNID_FILE):
        print(f"Error: {SNID_FILE} not found.")
        return
        
    df_snid = pd.read_csv(SNID_FILE)

    # Normalize filenames for merging
    df_dyn['filename_norm'] = df_dyn['filename'].str.lower()
    df_snid['filename_norm'] = df_snid['Filename'].str.lower()

    # 2. Merge Data
    df_merged = pd.merge(df_dyn, df_snid, on='filename_norm', suffixes=('_dyn', '_snid'))
    
    # Handle potential column name changes due to suffixes
    snr_col = 'SNR_snid' if 'SNR_snid' in df_merged.columns else 'SNR'
    subtype_col = 'Subtype_snid' if 'Subtype_snid' in df_merged.columns else 'Subtype'
    true_age_col = 'true_age_dyn' if 'true_age_dyn' in df_merged.columns else 'true_age'

    # Clean up
    df_merged = df_merged.dropna(subset=['nuis_age', 'bootstrap_age'])
    
    # 3. Filtering
    # Use the same filters as compare_methods.py to be consistent
    mask = (
        (df_merged[true_age_col] >= -15) & (df_merged[true_age_col] <= 50) &
        (df_merged[snr_col] >= 10) &
        (df_merged[subtype_col] != '91bg') &
        (df_merged[subtype_col] != 'pec')
    )
    df_filtered = df_merged[mask].copy()
    
    if df_filtered.empty:
        print("Error: No spectra left after filtering.")
        return

    # 4. Calculate Correlation
    corr, _ = pearsonr(df_filtered['nuis_age'], df_filtered['bootstrap_age'])
    rmse = np.sqrt(((df_filtered['nuis_age'] - df_filtered['bootstrap_age'])**2).mean())
    bias = (df_filtered['nuis_age'] - df_filtered['bootstrap_age']).mean()

    print(f"--- SNID vs Dynesty Comparison (N={len(df_filtered)}) ---")
    print(f"Pearson Correlation: {corr:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Bias (Dynesty - SNID): {bias:.3f}")

    # 5. Plotting
    if os.path.exists(STYLE_FILE):
        plt.style.use(STYLE_FILE)
    
    fig, ax = plt.subplots(figsize=(8, 8))

    # Determine limits
    min_val = min(df_filtered['nuis_age'].min(), df_filtered['bootstrap_age'].min()) - 2
    max_val = max(df_filtered['nuis_age'].max(), df_filtered['bootstrap_age'].max()) + 2
    
    # Scatter plot
    scatter = ax.scatter(
        df_filtered['nuis_age'], df_filtered['bootstrap_age'],
        c=df_filtered[true_age_col], cmap='viridis',
        alpha=0.6, s=30, label='Spectra'
    )
    
    # Add error bars if available
    if 'nuis_age_err' in df_filtered.columns and 'snid_std_dev' in df_filtered.columns:
        ax.errorbar(
            df_filtered['nuis_age'], df_filtered['bootstrap_age'],
            xerr=df_filtered['nuis_age_err'], yerr=df_filtered['snid_std_dev'],
            fmt='none', color='gray', alpha=0.2, zorder=0
        )

    # 1:1 Line
    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', alpha=0.7, label='1:1 Line')
    
    ax.set_xlabel(r"Nested Sampling Age ($t_{Dynesty}$ [days])")
    ax.set_ylabel(r"SNID Age ($t_{SNID}$ [days])")
    ax.set_title(f"SNID vs Nested Sampling Fit (N={len(df_filtered)})")

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('True Age (Light Curve) [days]')

    # Metrics Text Box
    metrics_text = (f"Correlation: {corr:.3f}\n"
                    f"RMSE: {rmse:.2f} d\n"
                    f"Bias: {bias:.2f} d")
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
             va='top', ha='left', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'snid_vs_dynesty_one_to_one.png')
    plt.savefig(plot_path, dpi=300)
    print(f"One-to-one plot saved to {plot_path}")

if __name__ == "__main__":
    run_comparison()
