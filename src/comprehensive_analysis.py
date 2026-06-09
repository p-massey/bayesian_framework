import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# filtered_comparison_results.csv contains all three methods (Full, Nuisance, SNID)
DYNESTY_FILE = 'outputs/csvs/allcfa_results.csv'
SNID_FILE = 'outputs/method_comparison/cfa_SNID_results.csv'
OUTPUT_DIR = 'outputs/method_comparison/comprehensive_results'
STYLE_FILE = 'assets/plotting_style.mplstyle'

# Constants for filtering and binning
PHASE_MIN = -15
PHASE_MAX = 25

# Font size settings for better visibility
LABEL_SIZE = 14
TITLE_SIZE = 16
TICK_SIZE = 12
LEGEND_SIZE = 12

# Settings for the two binning versions
BIN_CONFIGS = [
    {'name': '5day', 'size': 5},
    {'name': '2day', 'size': 2}
]

def calculate_binned_metrics(df, snid_age, snid_age_err, dyn_age, dyn_age_err, true_age='true_age', true_age_err='Age_Unc_(days)'):
    """
    Calculates performance metrics binned by age for both SNID and Dynesty methods.
    Follows the requested logic of calculating residuals and errors for both upfront.
    """
    df = df.copy()
    
    # Calculate residuals for both methods by subtracting away the true age
    df['snid_residual'] = df[snid_age] - df[true_age]
    df['dyn_residual'] = df[dyn_age] - df[true_age]
    
    # Calculate errors combined in quadrature
    df['snid_residual_err'] = np.sqrt(df[snid_age_err]**2 + df[true_age_err]**2)
    df['dyn_residual_err'] = np.sqrt(df[dyn_age_err]**2 + df[true_age_err]**2)

    # Group by the pre-defined age_bin
    grouped = df.groupby('age_bin', observed=True)
    
    # Map for easy access to method-specific error columns
    err_col_map = {'snid': snid_age_err, 'dyn': dyn_age_err}
    
    def get_group_stats(group):
        if len(group) == 0:
            return pd.Series({
                'snid_bias': np.nan, 'snid_spread': np.nan, 'snid_rmse': np.nan, 'snid_mad': np.nan, 'snid_snid_err': np.nan,
                'dyn_bias': np.nan, 'dyn_spread': np.nan, 'dyn_rmse': np.nan, 'dyn_mad': np.nan,
                'count': 0
            })
            
        stats = {}
        for prefix in ['snid', 'dyn']:
            res = group[f'{prefix}_residual']
            err = group[f'{prefix}_residual_err']
            
            # Coverage
            cov1 = (np.abs(res) <= err).mean()
            cov2 = (np.abs(res) <= 2 * err).mean()
            
            # Reduced Chi-Squared
            safe_err = np.where(err > 0, err, np.nan)
            chi2_red = np.nanmean((res / safe_err)**2)
            
            stats.update({
                f'{prefix}_bias': res.mean(),
                f'{prefix}_spread': res.std(),
                f'{prefix}_rmse': np.sqrt((res**2).mean()),
                f'{prefix}_mad': np.median(np.abs(res)),
                f'{prefix}_coverage_1sigma': cov1,
                f'{prefix}_coverage_2sigma': cov2,
                f'{prefix}_chi2_red': chi2_red,
            })
            # Specifically add the requested error metric
            if prefix == 'snid':
                stats['snid_snid_err'] = group[err_col_map[prefix]].mean()

        stats['count'] = len(group)
        return pd.Series(stats)

    return grouped.apply(get_group_stats)

def run_analysis(bin_size, suffix):
    print(f"\nRunning analysis with {bin_size}-day bins...")
    
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

    # Merge Data
    df = pd.merge(df_dyn, df_snid, on='filename_norm', suffixes=('_dyn', '_snid'))
    
    # Apply standard filters
    df = df[df['SNR'] >= 10].copy()
    df = df[df['Subtype'].isin(['N', 'HV'])].copy()
    
    # Filter to specified phase range
    df = df[(df['true_age'] >= PHASE_MIN) & (df['true_age'] <= PHASE_MAX)].copy()
    
    if df.empty:
        print("Error: No data in specified phase range after filtering.")
        return

    # 2. Define Bins
    bin_edges = np.arange(PHASE_MIN, PHASE_MAX + bin_size, bin_size)
    df['age_bin'] = pd.cut(df['true_age'], bins=bin_edges)
    
    # Calculate centers based on actual edges to ensure alignment
    actual_bins = df['age_bin'].cat.categories
    bin_centers = np.array([b.mid for b in actual_bins])
    
    # 3. Methods Mapping
    # For now, we compare 'Dynesty (Nuisance)' as 'dyn' and 'SNID (Bootstrap)' as 'snid'
    results = calculate_binned_metrics(
        df, 
        snid_age='bootstrap_age', 
        snid_age_err='snid_std_dev', 
        dyn_age='age_median', 
        dyn_age_err='age_err',
        true_age='true_age',
        true_age_err='Age_Unc_(days)'
    )

    # 4. Plotting
    if os.path.exists(STYLE_FILE):
        plt.style.use(STYLE_FILE)
    else:
        plt.style.use('seaborn-v0_8-whitegrid')

    # Define individual plots
    plot_definitions = [
        {'id': 'bias', 'label': 'Bias (days)', 'title': 'Mean Residual (Bias)'},
        {'id': 'spread', 'label': 'Spread (days)', 'title': 'Standard Deviation of Residuals'},
        {'id': 'rmse', 'label': 'RMSE (days)', 'title': 'Root Mean Square Error'},
        {'id': 'mad', 'label': 'MAD (days)', 'title': 'Median Absolute Deviation'},
        {'id': 'snid_err', 'label': 'SNID Error (days)', 'title': 'Mean SNID Estimated Error', 'methods': ['snid']},
        {'id': 'coverage_1sigma', 'label': r'1-$\sigma$ Coverage', 'title': r'Error Calibration (1-$\sigma$)'},
        {'id': 'coverage_2sigma', 'label': r'2-$\sigma$ Coverage', 'title': r'Error Calibration (2-$\sigma$)'},
        {'id': 'chi2_red', 'label': r'Reduced $\chi^2$', 'title': r'Reduced $\chi^2$ Statistics'}
    ]

    colors = {'dyn': '#D55E00', 'snid': '#009E73'}
    labels = {'dyn': 'Full Bayesian', 'snid': 'SNID (Bootstrap)'}
    markers = {'dyn': 'o', 'snid': '^'}

    for pdef in plot_definitions:
        metric_base = pdef['id']
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determine which methods to plot (default to both)
        methods_to_plot = pdef.get('methods', ['dyn', 'snid'])
        
        for prefix in methods_to_plot:
            metric = f'{prefix}_{metric_base}'
            # Filter out bins with 0 count or NaNs
            valid_mask = (results['count'] > 0) & (results[metric].notna())
            centers = bin_centers[valid_mask]
            values = results[metric][valid_mask]
            
            ax.plot(centers, values, marker=markers[prefix], color=colors[prefix], label=labels[prefix], lw=2.5, markersize=8)
        
        # Reference lines and scales
        if metric_base == 'bias':
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        elif metric_base == 'coverage_1sigma':
            ax.axhline(0.683, color='black', linestyle='--', alpha=0.5, label='Expected (68.3%)')
            ax.set_ylim(0, 1.05)
        elif metric_base == 'coverage_2sigma':
            ax.axhline(0.954, color='black', linestyle='--', alpha=0.5, label='Expected (95.4%)')
            ax.set_ylim(0, 1.05)
        elif metric_base == 'chi2_red':
            ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Expected (1.0)')
            ax.set_yscale('log')
        
        # Labels and formatting
        ax.set_ylabel(pdef['label'], fontsize=LABEL_SIZE)
        ax.set_xlabel('t_LC (days)', fontsize=LABEL_SIZE)
        ax.set_title(f"{pdef['title']} ({bin_size}-day bins)", fontsize=TITLE_SIZE)
        
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        ax.legend(fontsize=LEGEND_SIZE)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"{suffix}_{metric_base}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved individual plot: {save_path}")

    # 5. Save Summary Table
    results['bin_center'] = bin_centers
    csv_path = os.path.join(OUTPUT_DIR, f'detailed_metrics_{suffix}.csv')
    results.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    for config in BIN_CONFIGS:
        run_analysis(config['size'], config['name'])
