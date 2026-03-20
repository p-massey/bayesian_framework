import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def update_plot():
    csv_file = "outputs/csvs/random_test_results.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file).dropna()
    if df.empty:
        print("No valid results to plot.")
        return

    # Helper to clean SN names from filenames
    def clean_name(filename):
        if filename.startswith('snf') or filename.startswith('sne'):
            parts = filename.split('-')
            return f"{parts[0]}-{parts[1]}"
        else:
            return filename.split('-')[0]

    df['sn_name'] = df['filename'].apply(clean_name)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot Full Fit
    ax1.errorbar(df['true_age'], df['full_age'], yerr=df['full_age_err'], fmt='o', label='Full Fit', color='blue', alpha=0.4, capsize=3)
    # Plot Nuisance Fit
    ax1.errorbar(df['true_age'], df['nuis_age'], yerr=df['nuis_age_err'], fmt='s', label='Nuisance Fit', color='red', alpha=0.4, capsize=3)
    
    # Annotate names
    for i, row in df.iterrows():
        # Label the nuisance fit points (usually more stable)
        ax1.annotate(row['sn_name'], (row['true_age'], row['nuis_age']), 
                     xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
    min_val = min(df['true_age'].min(), df['full_age'].min(), df['nuis_age'].min()) - 5
    max_val = max(df['true_age'].max(), df['full_age'].max(), df['nuis_age'].max()) + 5
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='1:1 Line')
    
    ax1.set_ylabel('Inferred Age (days)')
    ax1.set_title(f'Method Comparison with Labels ({len(df)} spectra)')
    ax1.legend()
    ax1.grid(True, ls=':', alpha=0.6)
    ax1.set_xlim(-20, 50)
    ax1.set_ylim(-25, 50)
    
    # Residuals
    ax2.errorbar(df['true_age'], df['full_age'] - df['true_age'], yerr=df['full_age_err'], fmt='o', color='blue', alpha=0.4)
    ax2.errorbar(df['true_age'], df['nuis_age'] - df['true_age'], yerr=df['nuis_age_err'], fmt='s', color='red', alpha=0.4)
    ax2.axhline(0, color='k', ls='--')
    ax2.set_xlabel('True Age (days)')
    ax2.set_ylabel('Residual (days)')
    ax2.grid(True, ls=':', alpha=0.6)
    ax2.set_ylim(-20, 20)




    
    plt.tight_layout()
    output_file = "outputs/plots/random_age_comparison_500points.png"
    plt.savefig(output_file, dpi=200)
    print(f"Updated plot saved to {output_file}")

if __name__ == "__main__":
    update_plot()
