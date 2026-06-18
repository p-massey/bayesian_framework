import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LogNorm

# Configuration
input_file = "outputs/csvs/allcfa_results_filtered.csv"
output_dir = "outputs/analysis"
os.makedirs(output_dir, exist_ok=True)

# Load already filtered data (but we will apply the stricter cuts)
if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Please run src/analyze_cfa_results.py first.")
    exit()

df = pd.read_csv(input_file)

# Styling
style_file = 'assets/plotting_style.mplstyle'
if os.path.exists(style_file):
    plt.style.use(style_file)
else:
    print(f"Warning: {style_file} not found, using default style.")

# Filtering logic matching src/plot_snid_vs_dynesty.py
# (Note: true_age, SNR, and Subtype cuts are already partially in the filtered CSV, but we re-apply for precision)
mask = (
    (df['true_age'] >= -15) & (df['true_age'] <= 50) &
    (df['nuis_age'] <= 35) &
    (df['SNR'] >= 10) &
    (df['Subtype'] != '91bg') &
    (df['Subtype'] != 'pec')
)
df_plot = df[mask].copy()

# Calculations
df_plot['res'] = df_plot['nuis_age'] - df_plot['true_age']
mean_res = df_plot['res'].mean()
std_res = df_plot['res'].std()
rmse = np.sqrt((df_plot['res']**2).mean())

print(f"Plotting for Nuisance method with strict filtering (N={len(df_plot)}):")
print(f"  Mean Offset: {mean_res:.3f} days")
print(f"  Standard Deviation: {std_res:.3f} days")
print(f"  RMSE: {rmse:.3f} days")

def create_param_plot(df, param_col, label, filename, is_log=False):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 12), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})

    # Color mapping
    norm = LogNorm() if is_log else None
    
    sc1 = ax1.scatter(df['true_age'], df['nuis_age'], c=df[param_col], cmap='viridis', 
                      s=30, alpha=0.8, edgecolors='none', zorder=3, norm=norm)
    ax1.errorbar(df['true_age'], df['nuis_age'], yerr=df['nuis_age_err'], 
                 fmt='none', ecolor='gray', alpha=0.3, capsize=0, zorder=2)

    # Colorbar
    cbar = plt.colorbar(sc1, ax=ax1, pad=0.02)
    cbar.set_label(label)

    # 1:1 Line
    ax1.plot([-20, 60], [-20, 60], 'k--', alpha=0.7, label='1:1 Line')
    ax1.set_xlim([-15, 50])
    ax1.set_ylim([-20, 55])
    ax1.set_ylabel('Inferred Age (days)')
    ax1.set_title(f'Coloured by {label} (N={len(df)})')
    ax1.legend(loc='lower right')
    ax1.grid(True, ls=':', alpha=0.4)

    # Add statistics text box
    stats_text = (r"$\bf{Nuisance\ Method:}$" + "\n"
                  f"Bias: {mean_res:.2f} d\n"
                  f"$\sigma$: {std_res:.2f} d\n"
                  f"RMSE: {rmse:.2f} d")

    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, va='top', ha='left', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

    # Residuals plot
    sc2 = ax2.scatter(df['true_age'], df['res'], c=df[param_col], cmap='viridis', 
                      s=30, alpha=0.8, edgecolors='none', zorder=3, norm=norm)
    ax2.errorbar(df['true_age'], df['res'], yerr=df['nuis_age_err'], 
                 fmt='none', ecolor='gray', alpha=0.3, capsize=0, zorder=2)

    # Colorbar for the bottom plot to keep alignment
    cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.02)
    cbar2.set_label(label)

    ax2.axhline(0, color='k', ls='--')
    ax2.set_xlabel('True Age (days)')
    ax2.set_ylabel('Residual (days)')
    ax2.grid(True, ls=':', alpha=0.4)
    ax2.set_ylim([-15, 15])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")
    plt.close()

# Create the three plots
create_param_plot(df_plot, 'nuis_x1_mean', 'Salt3 $x_1$', 'nuis_x1_comparison.png')
create_param_plot(df_plot, 'nuis_c_mean', 'Salt3 $c$', 'nuis_c_comparison.png')
create_param_plot(df_plot, 'nuis_x0_mean', 'Salt3 $x_0$', 'nuis_x0_comparison.png', is_log=True)
