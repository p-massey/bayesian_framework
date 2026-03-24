import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
input_file = "outputs/csvs/allcfa_results.csv"
properties_file = "data/spectra_properties.csv"
output_dir = "outputs/analysis"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv(input_file)
props = pd.read_csv(properties_file)

# Merge with properties for SNR and Subtype
# Note: properties_file has 'Filename' while input_file has 'filename'
df = df.merge(props[['Filename', 'SNR', 'Subtype']], left_on='filename', right_on='Filename', how='left')

# Styling
style_file = 'assets/plotting_style.mplstyle'
if os.path.exists(style_file):
    plt.style.use(style_file)
else:
    print(f"Warning: {style_file} not found, using default style.")

# Filtering
# 1. Phase range -20 to 50
# 2. SNR >= 10
# 3. Exclude 91bg and pec (unknown/peculiar)
# 4. Handle NaN in Subtype if necessary (usually normal Ia if not specified, but let's be safe)
mask = (
    (df['true_age'] >= -20) & (df['true_age'] <= 50) &
    (df['SNR'] >= 10) &
    (df['Subtype'] != '91bg') &
    (df['Subtype'] != 'pec') &
    (df['Subtype'].notna())
)
df_filtered = df[mask].copy()

# Calculations
results = {}
for method in ['full', 'nuis']:
    age_col = f'{method}_age'
    
    # Residuals
    df_filtered[f'{method}_res'] = df_filtered[age_col] - df_filtered['true_age']
    
    # Statistics
    mean_res = df_filtered[f'{method}_res'].mean()
    std_res = df_filtered[f'{method}_res'].std()
    rmse = np.sqrt((df_filtered[f'{method}_res']**2).mean())
    
    results[method] = {
        'mean_offset': mean_res,
        'std_dev': std_res,
        'rmse': rmse
    }

print(f"Analysis for phases -20 to 50 days, SNR >= 10, excluding 91bg/pec (N={len(df_filtered)}):")
for method, stats in results.items():
    print(f"\nMethod: {method}")
    print(f"  Mean Offset: {stats['mean_offset']:.3f} days")
    print(f"  Standard Deviation: {stats['std_dev']:.3f} days")
    print(f"  RMSE: {stats['rmse']:.3f} days")

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})

# Main plot
ax1.errorbar(df_filtered['true_age'], df_filtered['full_age'], yerr=df_filtered['full_age_err'], 
             fmt='o', markersize=3, label=f"Full (RMSE={results['full']['rmse']:.2f})", alpha=0.4, color='blue', capsize=0)
ax1.errorbar(df_filtered['true_age'], df_filtered['nuis_age'], yerr=df_filtered['nuis_age_err'], 
             fmt='s', markersize=3, label=f"Nuisance (RMSE={results['nuis']['rmse']:.2f})", alpha=0.4, color='red', capsize=0)

# 1:1 Line
lims = [-25, 55]
ax1.plot(lims, lims, 'k--', alpha=0.7, label='1:1 Line')
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_ylabel('Inferred Age (days)')
ax1.set_title(f'CfA Spectra: -20 < Phase < 50, SNR >= 10, No 91bg/pec (N={len(df_filtered)})')
ax1.legend(loc='lower right')
ax1.grid(True, ls=':', alpha=0.6)

# Add statistics text boxes
stats_text_full = (r"$\bf{Full\ (x_0\ sampled):}$" + "\n"
                   f"Bias: {results['full']['mean_offset']:.2f} d\n"
                   f"$\sigma$: {results['full']['std_dev']:.2f} d\n"
                   f"RMSE: {results['full']['rmse']:.2f} d")

stats_text_nuis = (r"$\bf{Nuisance\ (x_0\ marginalized):}$" + "\n"
                   f"Bias: {results['nuis']['mean_offset']:.2f} d\n"
                   f"$\sigma$: {results['nuis']['std_dev']:.2f} d\n"
                   f"RMSE: {results['nuis']['rmse']:.2f} d")

ax1.text(0.05, 0.95, stats_text_full, transform=ax1.transAxes, va='top', ha='left', 
         fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="blue", alpha=0.7))

ax1.text(0.05, 0.75, stats_text_nuis, transform=ax1.transAxes, va='top', ha='left', 
         fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="red", alpha=0.7))

# Residuals plot
ax2.errorbar(df_filtered['true_age'], df_filtered['full_res'], yerr=df_filtered['full_age_err'], 
             fmt='o', markersize=3, alpha=0.4, color='blue', capsize=0)
ax2.errorbar(df_filtered['true_age'], df_filtered['nuis_res'], yerr=df_filtered['nuis_age_err'], 
             fmt='s', markersize=3, alpha=0.4, color='red', capsize=0)
ax2.axhline(0, color='k', ls='--')
ax2.set_xlabel('True Age (days)')
ax2.set_ylabel('Residual (days)')
ax2.grid(True, ls=':', alpha=0.6)

plt.tight_layout()
plot_path = os.path.join(output_dir, "filtered_cfa_comparison.png")
plt.savefig(plot_path, dpi=300)
print(f"\nPlot saved to: {plot_path}")
