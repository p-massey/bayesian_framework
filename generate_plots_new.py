import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
BOOTSTRAP_FILE = 'cfa_spectra_data_corrected.csv'
OUTPUT_DIR = 'paper_plots_new'
STYLE_FILE = 'GausSN.mplstyle'

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading data from {BOOTSTRAP_FILE}...")
    if not os.path.exists(BOOTSTRAP_FILE):
        print(f"Error: {BOOTSTRAP_FILE} not found.")
        return
        
    df = pd.read_csv(BOOTSTRAP_FILE)
    
    if df.empty:
        print("Error: Dataset is empty.")
        return
        
    print(f"Found {len(df)} spectra.")
    
    # 2. Process and Filter
    print("Applying filters (SNR >= 5, No 91T/91bg/Unknown)...")
    
    # Subtype check: Exclude 91bg, 91T, and Unknown
    def is_valid_subtype(st):
        st = str(st).lower()
        if any(x in st for x in ["91bg", "91t", "unknown", "n/a", "nan"]):
            return False
        return True

    # Filter for SNR and Subtype
    df_filtered = df[
        (df['snr'] >= 5.0) & 
        (df['subtype'].apply(is_valid_subtype))
    ].copy()

    # Create plotting dataframe
    df_plot = pd.DataFrame({
        'true_age': df_filtered['true_age'],
        'true_unc': df_filtered['true_age_err'].fillna(0.0),
        'estimated_age': df_filtered['bootstrap_age'],
        'std_dev': df_filtered['std_dev'],
        'residual': df_filtered['delta']
    })
    
    # Filter out points where true_age > 100 days as requested
    initial_df_plot_len = len(df_plot)
    df_plot = df_plot[df_plot['true_age'] <= 100].copy()
    if len(df_plot) < initial_df_plot_len:
        print(f"Filtered out {initial_df_plot_len - len(df_plot)} points with true_age > 100 days.")
    
    # WHOLE RANGE: Removed the -12 to 36d filter
    print(f"Plotting data with true_age <= 100 days (N={len(df_plot)}).")

    if df_plot.empty:
        print("Error: No data remains after filters.")
        return
        
    # Save the cleaned data for analysis
    df_plot.to_csv(os.path.join(OUTPUT_DIR, 'top20_cleaned_data_full_range.csv'), index=False)
    
    # 3. Plotting
    if os.path.exists(STYLE_FILE):
        plt.style.use(STYLE_FILE)
    
    # Calculate metrics
    sigma_t = df_plot['residual'].std()
    rmse = np.sqrt((df_plot['residual']**2).mean())
    bias = df_plot['residual'].mean()
    
    print(f"\n--- Statistics for Full Range Dataset (N={len(df_plot)}) ---")
    print(f"Sigma_t (Dispersion): {sigma_t:.3f} days")
    print(f"RMSE: {rmse:.3f} days")
    print(f"Mean Bias: {bias:.3f} days")

    df_plot['total_error'] = np.sqrt(df_plot['std_dev']**2 + df_plot['true_unc']**2)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Determine limits
    all_ages = pd.concat([df_plot['true_age'], df_plot['estimated_age']])
    min_age, max_age = all_ages.min() - 2, 100

    # Top Panel: SNID Predicted Age vs True Age
    ax1.errorbar(
        df_plot['true_age'], df_plot['estimated_age'],
        yerr=df_plot['std_dev'], xerr=df_plot['true_unc'],
        fmt='o', alpha=0.6, ecolor='lightgray', capsize=0
    )
    ax1.plot([min_age, max_age], [min_age, max_age], color='red', linestyle='--')
    ax1.set_ylabel(r"$t_{SNID}$ (days)")
    
    metrics_text = f"$\sigma_t$: {sigma_t:.2f} days\nBias: {bias:.2f} days"
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
             va='top', ha='left', fontsize=18, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min_age, max_age)
    ax1.set_ylim(min_age, max_age)
    
    # Bottom Panel: Residuals
    ax2.errorbar(
        df_plot['true_age'], df_plot['residual'],
        yerr=df_plot['total_error'], xerr=df_plot['true_unc'],
        fmt='o', alpha=0.6, ecolor='lightgray', capsize=0
    )
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel(r"$t_{LC}$ (days)")
    ax2.set_ylabel(r"$t_{SNID} - t_{LC}$ (days)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min_age, max_age)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'snid_vs_true_correlation_full_range.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Correlation plot saved to {plot_path}")

if __name__ == "__main__":
    main()
