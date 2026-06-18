import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
input_file = "outputs/csvs/allcfa_results_filtered.csv"
output_dir = "outputs/analysis"
os.makedirs(output_dir, exist_ok=True)

# Load already filtered data
if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Please run the filtering step first.")
    exit()

df = pd.read_csv(input_file)

# Styling
style_file = 'assets/plotting_style.mplstyle'
if os.path.exists(style_file):
    plt.style.use(style_file)
else:
    print(f"Warning: {style_file} not found, using default style.")

# Calculations
df['res'] = df['nuis_age'] - df['true_age']
mean_res = df['res'].mean()
std_res = df['res'].std()
rmse = np.sqrt((df['res']**2).mean())

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 12), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})

# Main plot - Scatter colored by SNR
# We use scatter for the points to get the color mapping, then errorbars without points
SNR_MAX_COLOR = 150 # Cap for colorbar visualization
sc1 = ax1.scatter(df['true_age'], df['nuis_age'], c=df['SNR'], cmap='viridis', 
                  s=30, alpha=0.8, edgecolors='none', zorder=3, label='Nuisance Fit', 
                  vmax=SNR_MAX_COLOR)
ax1.errorbar(df['true_age'], df['nuis_age'], yerr=df['nuis_age_err'], 
             fmt='none', ecolor='gray', alpha=0.3, capsize=0, zorder=2)

# Colorbar for the top plot
cbar = plt.colorbar(sc1, ax=ax1, pad=0.02, extend='max')
cbar.set_label('Signal-to-Noise Ratio (SNR)')

# 1:1 Line
ax1.plot([-20, 60], [-20, 60], 'k--', alpha=0.7, label='1:1 Line')
ax1.set_xlim([-15, 50])
ax1.set_ylim([-20, 55])
ax1.set_ylabel('Inferred Age (days)')
ax1.set_title(f'Nuisance Method colored by SNR (N={len(df)})')
ax1.legend(loc='lower right')
ax1.grid(True, ls=':', alpha=0.4)

# Add statistics text box
stats_text = (r"$\bf{Nuisance\ Method:}$" + "\n"
              f"Bias: {mean_res:.2f} d\n"
              f"$\sigma$: {std_res:.2f} d\n"
              f"RMSE: {rmse:.2f} d")

ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, va='top', ha='left', 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

# Residuals plot - Scatter colored by SNR
sc2 = ax2.scatter(df['true_age'], df['res'], c=df['SNR'], cmap='viridis', 
                  s=30, alpha=0.8, edgecolors='none', zorder=3, vmax=SNR_MAX_COLOR)
ax2.errorbar(df['true_age'], df['res'], yerr=df['nuis_age_err'], 
             fmt='none', ecolor='gray', alpha=0.3, capsize=0, zorder=2)

# Colorbar for the bottom plot (optional, but keeps alignment)
cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.02, extend='max')
cbar2.set_label('SNR')

ax2.axhline(0, color='k', ls='--')
ax2.set_xlabel('True Age (days)')
ax2.set_ylabel('Residual (days)')
ax2.grid(True, ls=':', alpha=0.4)
ax2.set_ylim([-15, 15])

plt.tight_layout()
plot_path = os.path.join(output_dir, "nuisance_snr_comparison.png")
plt.savefig(plot_path, dpi=300)
print(f"\nSNR-colored plot saved to: {plot_path}")
