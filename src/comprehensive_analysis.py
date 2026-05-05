import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MERGED_FILE = 'outputs/method_comparison/filtered_comparison_results.csv'
OUTPUT_DIR = 'outputs/method_comparison'
STYLE_FILE = 'assets/plotting_style.mplstyle'

# Settings for the two binning versions
BIN_CONFIGS = [
    {'name': '5day', 'size': 5},
    {'name': '2day', 'size': 2}
]

PHASE_MIN = -15
PHASE_MAX = 25

def calculate_binned_metrics(df, age_col, err_col, true_age_col='true_age'):
    # Calculate residuals for the whole dataframe first
    df = df.copy()
    df['residual'] = df[age_col] - df[true_age_col]
    
    # Group by bin
    grouped = df.groupby('age_bin', observed=True)
    
    def get_group_stats(group):
        res = group['residual']
        err = group[err_col]
        
        # Coverage
        cov1 = (np.abs(res) <= err).mean()
        cov2 = (np.abs(res) <= 2 * err).mean()
        
        # Reduced Chi-Squared
        # Guard against zero or extremely small errors
        safe_err = np.where(err > 0, err, np.nan)
        chi2_red = np.nanmean((res / safe_err)**2)
        
        return pd.Series({
            'bias': res.mean(),
            'spread': res.std(),
            'rmse': np.sqrt((res**2).mean()),
            'mad': np.median(np.abs(res)),
            'coverage_1sigma': cov1,
            'coverage_2sigma': cov2,
            'chi2_red': chi2_red,
            'count': len(group)
        })

    return grouped.apply(get_group_stats)

def run_analysis(bin_size, suffix):
    print(f"Running analysis with {bin_size}-day bins...")
    
    # 1. Load Data
    if not os.path.exists(MERGED_FILE):
        print(f"Error: {MERGED_FILE} not found.")
        return

    df = pd.read_csv(MERGED_FILE)
    df = df[(df['true_age'] >= PHASE_MIN) & (df['true_age'] <= PHASE_MAX)].copy()
    
    if df.empty:
        print("Error: No data in specified phase range.")
        return

    # 2. Define Bins
    bin_edges = np.arange(PHASE_MIN, PHASE_MAX + bin_size, bin_size)
    df['age_bin'] = pd.cut(df['true_age'], bins=bin_edges)
    # Re-calculate centers based on actual edges to ensure alignment
    actual_bins = df['age_bin'].cat.categories
    bin_centers = np.array([b.mid for b in actual_bins])
    
    # 3. Methods Mapping
    # Column mapping: (age_col, err_col)
    methods = {
        'Full Bayesian': ('full_age', 'full_age_err'),
        'Nuisance Bayesian': ('nuis_age', 'nuis_age_err'),
        'SNID (Bootstrap)': ('bootstrap_age', 'snid_std_dev')
    }
    
    all_results = {}
    for name, cols in methods.items():
        all_results[name] = calculate_binned_metrics(df, cols[0], cols[1])

    # 4. Plotting
    if os.path.exists(STYLE_FILE):
        plt.style.use(STYLE_FILE)
    else:
        plt.style.use('seaborn-v0_8-whitegrid')

    # Define plot sets
    plot_sets = [
        {
            'title': 'Accuracy & Precision',
            'filename': f'comparison_accuracy_{suffix}.png',
            'metrics': ['bias', 'spread', 'rmse'],
            'labels': ['Bias (days)', 'Spread (days)', 'RMSE (days)']
        },
        {
            'title': 'Error Calibration',
            'filename': f'comparison_calibration_{suffix}.png',
            'metrics': ['coverage_1sigma', 'coverage_2sigma', 'chi2_red'],
            'labels': ['1-$\sigma$ Coverage', '2-$\sigma$ Coverage', 'Reduced $\chi^2$']
        }
    ]

    colors = {'Full Bayesian': '#D55E00', 'Nuisance Bayesian': '#0072B2', 'SNID (Bootstrap)': '#009E73'}
    markers = {'Full Bayesian': 'o', 'Nuisance Bayesian': 's', 'SNID (Bootstrap)': '^'}

    for pset in plot_sets:
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        for i, metric in enumerate(pset['metrics']):
            ax = axes[i]
            for name in methods:
                res = all_results[name]
                # Filter out bins with 0 count or all NaNs for this metric
                valid_mask = (res['count'] > 0) & (res[metric].notna())
                centers = bin_centers[valid_mask]
                values = res[metric][valid_mask]
                
                ax.plot(centers, values, marker=markers[name], color=colors[name], label=name, lw=2)
            
            if metric == 'bias':
                ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            elif metric == 'coverage_1sigma':
                ax.axhline(0.683, color='black', linestyle='--', alpha=0.5, label='Expected (68.3%)')
                ax.set_ylim(0, 1.05)
            elif metric == 'coverage_2sigma':
                ax.axhline(0.954, color='black', linestyle='--', alpha=0.5, label='Expected (95.4%)')
                ax.set_ylim(0, 1.05)
            elif metric == 'chi2_red':
                ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Expected (1.0)')
                ax.set_yscale('log')
            
            ax.set_ylabel(pset['labels'][i])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        axes[0].set_title(f"{pset['title']} ({bin_size}-day bins)")
        axes[2].set_xlabel('True Age (days)')
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, pset['filename'])
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved plot: {save_path}")

    # 5. Save Summary Table
    summary_data = []
    for name in methods:
        res = all_results[name].reset_index()
        res['Method'] = name
        # Centers need to match the index of the series
        res['bin_center'] = bin_centers
        summary_data.append(res)
    
    final_df = pd.concat(summary_data, ignore_index=True)
    csv_path = os.path.join(OUTPUT_DIR, f'detailed_metrics_{suffix}.csv')
    final_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_analysis(5, '5day')
    run_analysis(2, '2day')
