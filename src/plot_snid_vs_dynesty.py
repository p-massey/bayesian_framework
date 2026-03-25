import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.gridspec import GridSpec

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
    mask = (
        (df_merged[true_age_col] >= -15) & (df_merged[true_age_col] <= 50) &
        (df_merged['nuis_age'] <= 35) &
        (df_merged[snr_col] >= 10) &
        (df_merged[subtype_col] != '91bg') &
        (df_merged[subtype_col] != 'pec')
    )
    df_filtered = df_merged[mask].copy()
    
    if df_filtered.empty:
        print("Error: No spectra left after filtering.")
        return

    # 4. Calculate Stats (Done AFTER filtering for nuis_age <= 35)
    residuals = df_filtered['bootstrap_age'] - df_filtered['nuis_age']
    corr, _ = pearsonr(df_filtered['nuis_age'], df_filtered['bootstrap_age'])
    rmse = np.sqrt((residuals**2).mean())
    bias = residuals.mean()
    std_resid = residuals.std()

    print(f"--- SNID vs Dynesty Comparison (N={len(df_filtered)}, nuis_age <= 35) ---")
    print(f"Pearson Correlation: {corr:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Bias (SNID - Dynesty): {bias:.3f}")
    print(f"Std of Residuals: {std_resid:.3f}")

    # 5. Plotting
    if os.path.exists(STYLE_FILE):
        plt.style.use(STYLE_FILE)
    
    # Use GridSpec for better control over subplot ratios
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    
    ax_main = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1], sharex=ax_main)

    # Determine limits
    min_val = -5
    max_val = 20

    # --- Main Plot ---
    scatter = ax_main.scatter(
        df_filtered['nuis_age'], df_filtered['bootstrap_age'],
        c=df_filtered[true_age_col], cmap='viridis',
        alpha=0.6, s=30, label='Spectra'
    )
    
    # Add error bars
    if 'nuis_age_err' in df_filtered.columns and 'snid_std_dev' in df_filtered.columns:
        ax_main.errorbar(
            df_filtered['nuis_age'], df_filtered['bootstrap_age'],
            xerr=df_filtered['nuis_age_err'], yerr=df_filtered['snid_std_dev'],
            fmt='none', color='gray', alpha=0.1, zorder=0
        )

    ax_main.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', alpha=0.7, label='1:1 Line')
    
    ax_main.set_ylabel(r"SNID Age ($t_{SNID}$ [days])")
    ax_main.set_title(fr"SNID vs Nested Sampling Fit (N={len(df_filtered)}, $t_{{Dynesty}} \leq 35d$)")
    ax_main.grid(True, alpha=0.3)
    
    # Metrics Text Box
    metrics_text = (f"Correlation: {corr:.3f}\n"
                    f"RMSE: {rmse:.2f} d\n"
                    f"Bias: {bias:.2f} d")
    ax_main.text(0.05, 0.95, metrics_text, transform=ax_main.transAxes,
                va='top', ha='left', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

    # Add colorbar (attach to BOTH axes to ensure they shrink together and align)
    cbar = fig.colorbar(scatter, ax=[ax_main, ax_resid], pad=0.02, aspect=30)
    cbar.set_label('True Age (Light Curve) [days]')

    # --- Residual Plot ---
    ax_resid.scatter(
        df_filtered['nuis_age'], residuals,
        c=df_filtered[true_age_col], cmap='viridis',
        alpha=0.6, s=30
    )
    
    # Add horizontal lines
    ax_resid.axhline(0, color='black', linestyle='-', alpha=0.8)
    ax_resid.axhline(bias, color='red', linestyle='--', alpha=0.6, label=f'Bias ({bias:.2f})')
    
    # Add error bars for residuals if available
    if 'nuis_age_err' in df_filtered.columns and 'snid_std_dev' in df_filtered.columns:
        resid_err = np.sqrt(df_filtered['nuis_age_err']**2 + df_filtered['snid_std_dev']**2)
        ax_resid.errorbar(
            df_filtered['nuis_age'], residuals,
            yerr=resid_err,
            fmt='none', color='gray', alpha=0.1, zorder=0
        )

    ax_resid.set_xlabel(r"Nested Sampling Age ($t_{Dynesty}$ [days])")
    ax_resid.set_ylabel(r"Residual (SNID-Dyn)")
    ax_resid.grid(True, alpha=0.3)
    
    # Remove x-axis tick labels for the top plot
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # Legend for bias
    ax_resid.legend(loc='upper right', fontsize=9)

    # Set limits
    ax_main.set_xlim(min_val, max_val)
    ax_main.set_ylim(min_val, max_val)
    
    # Residual y-limits (centered around bias or 0)
    res_max = max(abs(residuals.max()), abs(residuals.min())) * 1.1
    ax_resid.set_ylim(-res_max, res_max)

    plot_path = os.path.join(OUTPUT_DIR, 'snid_vs_dynesty_one_to_one_with_residuals.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"One-to-one plot with residuals saved to {plot_path}")

if __name__ == "__main__":
    run_comparison()
