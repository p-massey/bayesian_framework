import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
DATA_FILE = 'outputs/csvs/cfa_test_results.csv'
STYLE_FILE = 'assets/plotting_style.mplstyle'
OUTPUT_PLOT = 'outputs/plots/cfa_comparison_enhanced.png'

def calculate_metrics(true, inferred):
    residuals = inferred - true
    sigma_t = residuals.std()
    rmse = np.sqrt((residuals**2).mean())
    bias = residuals.mean()
    return sigma_t, rmse, bias

def run_plotting():
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return
        
    df = pd.read_csv(DATA_FILE)
    df = df.dropna()
    if df.empty:
        print("Error: No valid data to plot.")
        return

    # 2. Styling
    if os.path.exists(STYLE_FILE):
        plt.style.use(STYLE_FILE)
    else:
        # Fallback to a standard style if plotting_style.mplstyle isn't perfect
        plt.style.use('seaborn-v0_8-whitegrid')

    # 3. Calculate Metrics
    sigma_f, rmse_f, bias_f = calculate_metrics(df['true_age'], df['full_age'])
    sigma_n, rmse_n, bias_n = calculate_metrics(df['true_age'], df['nuis_age'])

    print(f"--- Full Fit Statistics (N={len(df)}) ---")
    print(f"Sigma_t: {sigma_f:.3f}, RMSE: {rmse_f:.3f}, Bias: {bias_f:.3f}")
    print(f"--- Nuisance Fit Statistics ---")
    print(f"Sigma_t: {sigma_n:.3f}, RMSE: {rmse_n:.3f}, Bias: {bias_n:.3f}")

    # 4. Create Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Determine limits
    min_age = min(df['true_age'].min(), df['nuis_age'].min()) - 2
    max_age = max(df['true_age'].max(), df['nuis_age'].max()) + 2

    # Top Panel: Inferred Age vs True Age
    ax1.errorbar(
        df['true_age'], df['nuis_age'], yerr=df['nuis_age_err'],
        fmt='s', color='red', alpha=0.5, label='Nuisance ($x_0$ marginalized)', capsize=0
    )
    
    ax1.plot([min_age, max_age], [min_age, max_age], color='black', linestyle='--', alpha=0.7, label='1:1 Line')
    ax1.set_ylabel(r"$t_{Inferred}$ (days)")
    ax1.set_title(f"Method Comparison on CfA Spectra (N={len(df)})")

    # Metrics Text Boxes
    metrics_text_n = (r"$\bf{Nuisance\ Fit:}$" + "\n"
                      f"$\sigma_t$: {sigma_n:.2f} d\n"
                      f"Bias: {bias_n:.2f} d")
    
    ax1.text(0.05, 0.95, metrics_text_n, transform=ax1.transAxes,
             va='top', ha='left', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min_age, max_age)
    ax1.set_ylim(min_age, max_age)

    # Bottom Panel: Residuals
    ax2.errorbar(
        df['true_age'], df['nuis_age'] - df['true_age'], yerr=df['nuis_age_err'],
        fmt='s', color='red', alpha=0.5, capsize=0
    )
    
    ax2.axhline(0, color='black', linestyle='--', alpha=0.7)
    ax2.set_xlabel(r"$t_{True}$ (days)")
    ax2.set_ylabel(r"$t_{Inferred} - t_{True}$ (days)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min_age, max_age)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Enhanced comparison plot saved to {OUTPUT_PLOT}")
    # plt.show() # Uncomment if running interactively

if __name__ == "__main__":
    run_plotting()
