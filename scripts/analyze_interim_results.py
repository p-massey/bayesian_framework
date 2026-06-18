import pandas as pd
import numpy as np
import os

CSV_PATH = '/Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/data/cfa_fits_results.csv'
PROP_PATH = '/Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/data/spectra_properties.csv'

def main():
    if not os.path.exists(CSV_PATH):
        print("Results file not found.")
        return
        
    df = pd.read_csv(CSV_PATH)
    df_success = df[df['Status'] == 'Success'].copy()
    
    if len(df_success) == 0:
        print("No successful fits yet.")
        return
        
    # Compute phases and differences
    df_success['phase_est'] = (df_success['MJD'] - df_success['t0']) / (1.0 + df_success['redshift'])
    df_success['phase_diff'] = df_success['phase_est'] - df_success['Age_(days)']
    
    # Merge with properties to get subtype info
    df_prop = pd.read_csv(PROP_PATH)
    df_joined = pd.merge(df_success, df_prop[['Filename', 'Subtype']], on='Filename', how='left')
    
    # Global metrics
    mean_bias = df_joined['phase_diff'].mean()
    median_bias = df_joined['phase_diff'].median()
    std_scatter = df_joined['phase_diff'].std()
    mad_scatter = 1.4826 * np.median(np.abs(df_joined['phase_diff'] - median_bias))
    
    print("==================================================")
    print(f"RESULTS BEFORE QUALITY CUTS (N = {len(df_joined)} successful fits)")
    print("==================================================")
    print(f"Global Mean Bias:      {mean_bias:+.3f} days")
    print(f"Global Median Bias:    {median_bias:+.3f} days")
    print(f"Global Std Scatter:    {std_scatter:.3f} days")
    print(f"Global Robust MAD:     {mad_scatter:.3f} days")
    print("--------------------------------------------------")
    
    # Subtype metrics
    print(f"{'Subtype':<8} | {'Count':<5} | {'Median Bias':<11} | {'Std':<6} | {'Robust MAD':<10}")
    print("-" * 50)
    for subtype, group in df_joined.groupby('Subtype', dropna=False):
        sub_name = str(subtype) if pd.notna(subtype) else 'Unknown'
        sub_count = len(group)
        sub_median = group['phase_diff'].median()
        sub_std = group['phase_diff'].std()
        sub_mad = 1.4826 * np.median(np.abs(group['phase_diff'] - sub_median)) if sub_count > 1 else 0.0
        
        std_str = f"{sub_std:.2f}d" if pd.notna(sub_std) else "N/A"
        print(f"{sub_name:<8} | {sub_count:<5} | {sub_median:>+10.2f}d | {std_str:<6} | {sub_mad:.2f}d")
    print("==================================================")
    print("\n")
    
    # Filter with quality cuts
    df_cut = df_joined[(df_joined['chi2_red'] <= 10.0) & (df_joined['t0_err'] <= 1.0)].copy()
    
    cut_mean = df_cut['phase_diff'].mean()
    cut_median = df_cut['phase_diff'].median()
    cut_std = df_cut['phase_diff'].std()
    cut_mad = 1.4826 * np.median(np.abs(df_cut['phase_diff'] - cut_median)) if len(df_cut) > 0 else 0.0
    
    print("==================================================")
    print(f"RESULTS AFTER QUALITY CUTS (chi2_red <= 10, t0_err <= 1.0) (N = {len(df_cut)} fits)")
    print("==================================================")
    print(f"Global Mean Bias:      {cut_mean:+.3f} days")
    print(f"Global Median Bias:    {cut_median:+.3f} days")
    print(f"Global Std Scatter:    {cut_std:.3f} days")
    print(f"Global Robust MAD:     {cut_mad:.3f} days")
    print("--------------------------------------------------")
    print(f"{'Subtype':<8} | {'Count':<5} | {'Median Bias':<11} | {'Std':<6} | {'Robust MAD':<10}")
    print("-" * 50)
    for subtype, group in df_cut.groupby('Subtype', dropna=False):
        sub_name = str(subtype) if pd.notna(subtype) else 'Unknown'
        sub_count = len(group)
        sub_median = group['phase_diff'].median()
        sub_std = group['phase_diff'].std()
        sub_mad = 1.4826 * np.median(np.abs(group['phase_diff'] - sub_median)) if sub_count > 1 else 0.0
        
        std_str = f"{sub_std:.2f}d" if pd.notna(sub_std) else "N/A"
        print(f"{sub_name:<8} | {sub_count:<5} | {sub_median:>+10.2f}d | {std_str:<6} | {sub_mad:.2f}d")
    print("==================================================")

if __name__ == '__main__':
    main()
