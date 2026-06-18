import pandas as pd
import numpy as np
import os

# Load the filtered results
filtered_results_path = "outputs/csvs/allcfa_results_filtered.csv"

if not os.path.exists(filtered_results_path):
    print(f"Error: {filtered_results_path} not found. Please run src/analyze_cfa_results.py first.")
    exit()

df = pd.read_csv(filtered_results_path)

# Calculate absolute residual for Nuisance method if not present
if 'nuis_res' not in df.columns:
    df['nuis_res'] = df['nuis_age'] - df['true_age']

df['abs_nuis_res'] = df['nuis_res'].abs()

# Sort by absolute residual descending
outliers = df.sort_values('abs_nuis_res', ascending=False).head(20)

# Select relevant columns for display
# These columns are already present in the filtered CSV (merged with SNR/Subtype)
cols = ['filename', 'true_age', 'nuis_age', 'nuis_res', 'SNR', 'Subtype']
print("Top 20 Outliers within the Filtered Range (-20 to 50 days, SNR >= 10, no 91bg/pec):")
print(outliers[cols].to_string(index=False))
