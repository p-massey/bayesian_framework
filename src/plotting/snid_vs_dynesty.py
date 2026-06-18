import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.gridspec import GridSpec

# Configuration
DYNESTY_FILE = 'outputs/csvs/allcfa_results.csv'
SNID_FILE = 'outputs/method_comparison/cfa_SNID_results.csv'
OUTPUT_DIR = 'outputs/method_comparison'
STYLE_FILE = 'assets/plotting_style.mplstyle'

def run_comparison():
    # 1. Load Data
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

    # print(df_merged.head())

    # Clean up
    df_merged = df_merged.dropna(subset=['age_fit', 'bootstrap_age'])

    # 3. Filtering
    mask = (
        (df_merged[true_age_col] >= -15) & (df_merged[true_age_col] <= 25) &
        (df_merged['age_fit'] >= -15) & (df_merged['age_fit'] <= 25) &
        (df_merged[snr_col] >= 10) &
        (df_merged['failed'] == False) &
        (df_merged[subtype_col].isin(['N', 'HV']))
    )


    df_filtered = df_merged[mask].copy()
    
    if df_filtered.empty:
        print("Error: No spectra left after filtering.")
        return

    # 4. Calculate Stats (Done AFTER filtering for nuis_age <= 35)
    residuals = df_filtered['bootstrap_age'] - df_filtered['age_fit']
    corr, _ = pearsonr(df_filtered['age_fit'], df_filtered['bootstrap_age'])
    rmse = np.sqrt((residuals**2).mean())
    bias = residuals.mean()
    std_resid = residuals.std()

    print(f"--- SNID vs Dynesty Comparison (N={len(df_filtered)}, phase range -15 to 25) ---")
    print(f"Pearson Correlation: {corr:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Bias (SNID - Dynesty): {bias:.3f}")
    print(f"Std of Residuals: {std_resid:.3f}")

    # 5. Plotting
    if os.path.exists(STYLE_FILE):
        plt.style.use(STYLE_FILE)
    
    # --- Plot 1: SNID vs Dynesty ---
    # Use GridSpec for better control over subplot ratios
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    
    ax_main = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1], sharex=ax_main)

    # Determine limits
    min_val = -15
    max_val = 25

    # --- Main Plot ---
    scatter = ax_main.scatter(
        df_filtered['age_fit'], df_filtered['bootstrap_age'],
        c=df_filtered[true_age_col], cmap='viridis',
        alpha=0.6, s=30, label='Spectra'
    )
    
    # Add error bars
    if 'age_err' in df_filtered.columns and 'snid_std_dev' in df_filtered.columns:
        ax_main.errorbar(
            df_filtered['age_fit'], df_filtered['bootstrap_age'],
            xerr=df_filtered['age_err'], yerr=df_filtered['snid_std_dev'],
            fmt='none', color='gray', alpha=0.2, zorder=0
        )
        
        # Combined error for residuals: sqrt(err1^2 + err2^2)
        combined_err = np.sqrt(df_filtered['age_err']**2 + df_filtered['snid_std_dev']**2)
        ax_resid.errorbar(
            df_filtered['age_fit'], residuals,
            yerr=combined_err,
            fmt='none', color='gray', alpha=0.2, zorder=0
        )

    ax_main.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', alpha=0.7, label='1:1 Line')
    
    ax_main.set_ylabel(r"SNID Age ($t_{SNID}$ [days])")
    ax_main.set_title(fr"SNID vs Nested Sampling Fit (N={len(df_filtered)}, Subtypes: N, HV)")
    ax_main.grid(True, alpha=0.3)
    
    # Metrics Text Box
    metrics_text = (f"Correlation: {corr:.3f}\n"
                    f"RMSE: {rmse:.2f} d\n"
                    f"Bias: {bias:.2f} d")
    ax_main.text(0.05, 0.95, metrics_text, transform=ax_main.transAxes,
                va='top', ha='left', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=[ax_main, ax_resid], pad=0.02, aspect=30)
    cbar.set_label('True Age (Light Curve) [days]')

    # --- Residual Plot ---
    ax_resid.scatter(
        df_filtered['age_fit'], residuals,
        c=df_filtered[true_age_col], cmap='viridis',
        alpha=0.6, s=30
    )
    
    # Add horizontal lines
    ax_resid.axhline(0, color='black', linestyle='-', alpha=0.8)
    ax_resid.axhline(bias, color='red', linestyle='--', alpha=0.6, label=f'Bias ({bias:.2f})')
    
    ax_resid.set_xlabel(r"Nested Sampling Age ($t_{Dynesty}$ [days])")
    ax_resid.set_ylabel(r"Residual (SNID-Dyn)")
    ax_resid.grid(True, alpha=0.3)

    plt.setp(ax_main.get_xticklabels(), visible=False)
    ax_resid.legend(loc='upper right', fontsize=9)

    ax_main.set_xlim(min_val, max_val)
    ax_main.set_ylim(min_val, max_val)
    ax_resid.set_ylim(-15, 15)

    plot_path = os.path.join(OUTPUT_DIR, 'snid_vs_dynesty_one_to_one_with_residuals.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"One-to-one plot with residuals saved to {plot_path}")

    # --- Plot 2: Found vs True Age (Both series) ---
    fig2 = plt.figure(figsize=(10, 10))
    gs2 = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    
    ax2_main = fig2.add_subplot(gs2[0])
    ax2_resid = fig2.add_subplot(gs2[1], sharex=ax2_main)

    true_age = df_filtered[true_age_col]
    
    # Dynesty series
    ax2_main.errorbar(
        true_age, df_filtered['age_fit'], 
        yerr=df_filtered['age_err'] if 'age_err' in df_filtered.columns else None,
        label='Dynesty', alpha=0.5, color='blue', fmt='o', markersize=4, capsize=0
    )
    # SNID series
    ax2_main.errorbar(
        true_age, df_filtered['bootstrap_age'], 
        yerr=df_filtered['snid_std_dev'] if 'snid_std_dev' in df_filtered.columns else None,
        label='SNID', alpha=0.5, color='red', fmt='o', markersize=4, capsize=0
    )
    
    min_t = -15
    max_t = 25
    ax2_main.plot([min_t, max_t], [min_t, max_t], 'k--', alpha=0.7, label='1:1 Line')
    
    ax2_main.set_ylabel("Inferred Age [days]")
    ax2_main.set_title(fr"Found vs True Age (N={len(df_filtered)}, Subtypes: N, HV)")
    ax2_main.legend()
    ax2_main.grid(True, alpha=0.3)
    ax2_main.set_xlim(min_t, max_t)
    ax2_main.set_ylim(min_t, max_t)

    # Residuals
    res_dyn = df_filtered['age_fit'] - true_age
    res_snid = df_filtered['bootstrap_age'] - true_age
    
    ax2_resid.errorbar(
        true_age, res_dyn, 
        yerr=df_filtered['age_err'] if 'age_err' in df_filtered.columns else None,
        alpha=0.4, color='blue', fmt='o', markersize=3, capsize=0
    )
    ax2_resid.errorbar(
        true_age, res_snid, 
        yerr=df_filtered['snid_std_dev'] if 'snid_std_dev' in df_filtered.columns else None,
        alpha=0.4, color='red', fmt='o', markersize=3, capsize=0
    )
    
    ax2_resid.axhline(0, color='black', linestyle='-')
    ax2_resid.set_xlabel("True Age (Light Curve) [days]")
    ax2_resid.set_ylabel("Residual (Found - True)")
    ax2_resid.grid(True, alpha=0.3)
    ax2_resid.set_ylim(-15, 15)

    plt.setp(ax2_main.get_xticklabels(), visible=False)
    
    plot_path2 = os.path.join(OUTPUT_DIR, 'found_vs_true_age_comparison.png')
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    print(f"Found vs True plot saved to {plot_path2}")

if __name__ == "__main__":
    run_comparison()
