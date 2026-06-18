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
df = df.merge(props[['Filename', 'SNR', 'Subtype']], left_on='filename', right_on='Filename', how='left')

# Styling
style_file = 'assets/plotting_style.mplstyle'
if os.path.exists(style_file):
    plt.style.use(style_file)
else:
    print(f"Warning: {style_file} not found, using default style.")

# Filtering
# 1. Phase range -15 to 50 (per user request to change x min to -15)
# 2. SNR >= 10
# 3. Exclude 91bg and pec
mask = (
    (df['true_age'] >= -15) & (df['true_age'] <= 50) &
    (df['SNR'] >= 10) &
    (df['Subtype'] != '91bg') &
    (df['Subtype'] != 'pec') &
    (df['Subtype'].notna())
)
df_filtered = df[mask].copy()

# Calculations for Nuisance only
method = 'nuis'
age_col = f'{method}_age'
err_col = f'{method}_age_err'

# Residuals
df_filtered['res'] = df_filtered[age_col] - df_filtered['true_age']

# Statistics
mean_res = df_filtered['res'].mean()
std_res = df_filtered['res'].std()
rmse = np.sqrt((df_filtered['res']**2).mean())

print(f"Analysis for Nuisance method (Phase -15 to 50, SNR >= 10, No 91bg/pec, N={len(df_filtered)}):")
print(f"  Mean Offset (Bias): {mean_res:.3f} days")
print(f"  Standard Deviation (Scatter): {std_res:.3f} days")
print(f"  RMSE: {rmse:.3f} days")

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})

# Main plot
ax1.errorbar(df_filtered['true_age'], df_filtered[age_col], yerr=df_filtered[err_col], 
             fmt='s', markersize=4, label='Nuisance ($x_0$ marginalized)', alpha=0.6, color='red', capsize=0)

# 1:1 Line
lims_x = [-15, 50]
lims_y = [-15, 55] # Slightly more headroom
ax1.plot([-20, 60], [-20, 60], 'k--', alpha=0.7, label='1:1 Line')
ax1.set_xlim(lims_x)
ax1.set_ylim([-20, 55])
ax1.set_ylabel('Inferred Age (days)')
ax1.set_title(f'Nuisance Method: -15 < Phase < 50, SNR >= 10 (N={len(df_filtered)})')
ax1.legend(loc='lower right')
ax1.grid(True, ls=':', alpha=0.6)

# Add statistics text box
stats_text = (r"$\bf{Nuisance\ Method:}$" + "\n"
              f"Bias: {mean_res:.2f} d\n"
              f"$\sigma$: {std_res:.2f} d\n"
              f"RMSE: {rmse:.2f} d")

ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, va='top', ha='left', 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="red", alpha=0.8))

# Residuals plot
ax2.errorbar(df_filtered['true_age'], df_filtered['res'], yerr=df_filtered[err_col], 
             fmt='s', markersize=4, alpha=0.6, color='red', capsize=0)
ax2.axhline(0, color='k', ls='--')
ax2.set_xlabel('True Age (days)')
ax2.set_ylabel('Residual (days)')
ax2.grid(True, ls=':', alpha=0.6)
ax2.set_ylim([-15, 15]) # Focused residual range

plt.tight_layout()
plot_path = os.path.join(output_dir, "nuisance_only_comparison.png")
plt.savefig(plot_path, dpi=300)
print(f"\nPlot saved to: {plot_path}")
