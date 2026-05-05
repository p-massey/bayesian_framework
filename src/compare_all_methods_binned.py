import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MERGED_FILE = 'outputs/method_comparison/filtered_comparison_results.csv'
OUTPUT_DIR = 'outputs/method_comparison'
STYLE_FILE = 'assets/plotting_style.mplstyle'

# Use same filtering as in the analysis scripts
PHASE_MIN = -15
PHASE_MAX = 25
BIN_SIZE = 5

def calculate_binned_metrics(df, age_col, true_age_col='true_age'):
    # Calculate residuals
    residuals = df[age_col] - df[true_age_col]
    
    # Group by bin
    grouped = residuals.groupby(df['age_bin'], observed=True)
    
    bias = grouped.mean()
    spread = grouped.std()
    rmse = np.sqrt(grouped.apply(lambda x: (x**2).mean()))
    count = grouped.count()
    
    return bias, spread, rmse, count

def run_comparison():
    # 1. Load Data
    if not os.path.exists(MERGED_FILE):
        print(f"Error: {MERGED_FILE} not found. Run src/compare_methods.py first.")
        return
        
    df = pd.read_csv(MERGED_FILE)
    
    # 2. Filter to range of interest
    df = df[(df['true_age'] >= PHASE_MIN) & (df['true_age'] <= PHASE_MAX)].copy()
    
    if df.empty:
        print("Error: No data in specified phase range.")
        return

    # 3. Define Bins
    bin_edges = np.arange(PHASE_MIN, PHASE_MAX + BIN_SIZE, BIN_SIZE)
    df['age_bin'] = pd.cut(df['true_age'], bins=bin_edges)
    bin_centers = bin_edges[:-1] + BIN_SIZE / 2
    
    # 4. Calculate Metrics for each method
    methods = {
        'Full Bayesian': 'full_age',
        'Nuisance Bayesian': 'nuis_age',
        'SNID (Bootstrap)': 'bootstrap_age'
    }
    
    results = {}
    for name, col in methods.items():
        bias, spread, rmse, count = calculate_binned_metrics(df, col)
        results[name] = {
            'bias': bias,
            'spread': spread,
            'rmse': rmse,
            'count': count
        }

    # 5. Plotting
    if os.path.exists(STYLE_FILE):
        plt.style.use(STYLE_FILE)
    else:
        plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    colors = {'Full Bayesian': '#D55E00', 'Nuisance Bayesian': '#0072B2', 'SNID (Bootstrap)': '#009E73'}
    markers = {'Full Bayesian': 'o', 'Nuisance Bayesian': 's', 'SNID (Bootstrap)': '^'}

    # Plot 1: Bias
    ax = axes[0]
    for name in methods:
        ax.plot(bin_centers, results[name]['bias'], marker=markers[name], color=colors[name], label=name, lw=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Bias (days)')
    ax.set_title('Binned Comparison: Bias (Mean Residual)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Spread (Sigma)
    ax = axes[1]
    for name in methods:
        ax.plot(bin_centers, results[name]['spread'], marker=markers[name], color=colors[name], label=name, lw=2)
    ax.set_ylabel('Spread (std of residuals) [days]')
    ax.set_title('Binned Comparison: Spread (Standard Deviation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 8)

    # Plot 3: RMSE
    ax = axes[2]
    for name in methods:
        ax.plot(bin_centers, results[name]['rmse'], marker=markers[name], color=colors[name], label=name, lw=2)
    ax.set_ylabel('RMSE (days)')
    ax.set_title('Binned Comparison: RMSE')
    ax.set_xlabel('True Age (days)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'comprehensive_method_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Comprehensive comparison plot saved to {plot_path}")

    # Save metrics to CSV
    metrics_summary = []
    for name in methods:
        for i, bin_val in enumerate(results[name]['bias'].index):
            metrics_summary.append({
                'Method': name,
                'Age Bin': bin_val,
                'Bin Center': bin_centers[i],
                'Bias': results[name]['bias'].iloc[i],
                'Spread': results[name]['spread'].iloc[i],
                'RMSE': results[name]['rmse'].iloc[i],
                'Count': results[name]['count'].iloc[i]
            })
    
    summary_df = pd.DataFrame(metrics_summary)
    summary_path = os.path.join(OUTPUT_DIR, 'comprehensive_metrics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Comprehensive metrics saved to {summary_path}")

if __name__ == "__main__":
    run_comparison()
