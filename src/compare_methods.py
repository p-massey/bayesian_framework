import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
DYNESTY_FILE = 'outputs/csvs/allcfa_results.csv'
SNID_FILE = 'outputs/method_comparison/cfa_SNID_results.csv'
OUTPUT_DIR = 'outputs/method_comparison'
STYLE_FILE = 'assets/plotting_style.mplstyle'

def calculate_metrics(true, inferred):
    residuals = inferred - true
    sigma_t = residuals.std()
    rmse = np.sqrt((residuals**2).mean())
    bias = residuals.mean()
    return sigma_t, rmse, bias

def run_comparison():
    # 1. Load Data
    if not os.path.exists(DYNESTY_FILE):
        print(f"Error: {DYNESTY_FILE} not found.")
        return
    if not os.path.exists(SNID_FILE):
        print(f"Error: {SNID_FILE} not found.")
        return
        
    df_dyn = pd.read_csv(DYNESTY_FILE)
    df_snid = pd.read_csv(SNID_FILE)

    # Normalize filenames for merging
    df_dyn['filename_norm'] = df_dyn['filename'].str.lower()
    df_snid['filename_norm'] = df_snid['Filename'].str.lower()

    # 2. Merge Data
    # Inner join to compare spectra present in both sets
    df_merged = pd.merge(df_dyn, df_snid, on='filename_norm', suffixes=('_dyn', '_snid'))
    
    # Clean up
    df_merged = df_merged.dropna(subset=['nuis_age', 'bootstrap_age', 'true_age'])
    
    # 3. Apply Filtering (Same cuts as before)
    # Phase range -15 to 50
    # SNR >= 10
    # Exclude 91bg and pec
    mask = (
        (df_merged['true_age'] >= -15) & (df_merged['true_age'] <= 50) &
        (df_merged['SNR'] >= 10) &
        (df_merged['Subtype'] != '91bg') &
        (df_merged['Subtype'] != 'pec') &
        (df_merged['Subtype'].notna())
    )
    df_filtered = df_merged[mask].copy()
    
    if df_filtered.empty:
        print("Error: No spectra left after filtering.")
        return

    # 4. Calculate Metrics
    # Dynesty (Nuisance) vs SNID (Bootstrap)
    sigma_n, rmse_n, bias_n = calculate_metrics(df_filtered['true_age'], df_filtered['nuis_age'])
    sigma_s, rmse_s, bias_s = calculate_metrics(df_filtered['true_age'], df_filtered['bootstrap_age'])

    print(f"--- Comparison Statistics (N={len(df_filtered)}) ---")
    print(f"Dynesty (Nuisance): Sigma={sigma_n:.3f}, RMSE={rmse_n:.3f}, Bias={bias_n:.3f}")
    print(f"SNID (Bootstrap):  Sigma={sigma_s:.3f}, RMSE={rmse_s:.3f}, Bias={bias_s:.3f}")

    # Save filtered results
    filtered_path = os.path.join(OUTPUT_DIR, 'filtered_comparison_results.csv')
    df_filtered.to_csv(filtered_path, index=False)
    print(f"Filtered results saved to {filtered_path}")

    # 5. Plotting
    if os.path.exists(STYLE_FILE):
        plt.style.use(STYLE_FILE)
    else:
        plt.style.use('seaborn-v0_8-whitegrid')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})

    # Determine limits
    min_age, max_age = -15, 50

    # Top Panel: Inferred Age vs True Age
    ax1.errorbar(
        df_filtered['true_age'], df_filtered['nuis_age'], yerr=df_filtered['nuis_age_err'],
        fmt='o', color='red', alpha=0.4, label='Dynesty (Nuisance)', capsize=0, markersize=3
    )
    ax1.errorbar(
        df_filtered['true_age'], df_filtered['bootstrap_age'], yerr=df_filtered['snid_std_dev'],
        fmt='s', color='blue', alpha=0.3, label='SNID (Bootstrap)', capsize=0, markersize=3
    )
    
    ax1.plot([-20, 60], [-20, 60], color='black', linestyle='--', alpha=0.7, label='1:1 Line')
    ax1.set_ylabel(r"$t_{Inferred}$ (days)")
    ax1.set_title(f"Dynesty vs SNID on CfA Spectra (N={len(df_filtered)})")

    # Metrics Text Boxes
    metrics_text = (r"$\bf{Dynesty\ (Nuisance):}$" + "\n"
                    f"$\sigma$: {sigma_n:.2f} d\n"
                    f"Bias: {bias_n:.2f} d\n\n"
                    r"$\bf{SNID:}$" + "\n"
                    f"$\sigma$: {sigma_s:.2f} d\n"
                    f"Bias: {bias_s:.2f} d")
    
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
             va='top', ha='left', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min_age, max_age)
    ax1.set_ylim(-20, 55)

    # Bottom Panel: Residuals
    ax2.errorbar(
        df_filtered['true_age'], df_filtered['nuis_age'] - df_filtered['true_age'], yerr=df_filtered['nuis_age_err'],
        fmt='o', color='red', alpha=0.4, capsize=0, markersize=3
    )
    ax2.errorbar(
        df_filtered['true_age'], df_filtered['bootstrap_age'] - df_filtered['true_age'], yerr=df_filtered['snid_std_dev'],
        fmt='s', color='blue', alpha=0.3, capsize=0, markersize=3
    )
    
    ax2.axhline(0, color='black', linestyle='--', alpha=0.7)
    ax2.set_xlabel(r"$t_{True}$ (days)")
    ax2.set_ylabel(r"$t_{Inferred} - t_{True}$ (days)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min_age, max_age)
    ax2.set_ylim(-15, 15)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'dynesty_vs_snid_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    run_comparison()
