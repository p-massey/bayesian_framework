import pandas as pd
import numpy as np
import os

# Load the filtered results
filtered_results_path = "outputs/csvs/allcfa_results_filtered.csv"
output_path = "outputs/analysis/filtered_outliers.csv"

if not os.path.exists(filtered_results_path):
    print(f"Error: {filtered_results_path} not found. Please run src/analyze_cfa_results.py first.")
    exit()

df = pd.read_csv(filtered_results_path)

# Calculate absolute residual for Nuisance method
if 'nuis_res' not in df.columns:
    df['nuis_res'] = df['nuis_age'] - df['true_age']

df['abs_nuis_res'] = df['nuis_res'].abs()

# Sort by absolute residual descending and take top 50
outliers = df.sort_values('abs_nuis_res', ascending=False).head(50)

# Save to CSV
outliers.to_csv(output_path, index=False)
print(f"Successfully saved top 50 outliers to: {output_path}")

# Print a small subset of the columns for confirmation
cols = ['filename', 'true_age', 'nuis_age', 'nuis_res', 'SNR', 'Subtype']
print("\nTop 10 Outliers Preview:")
print(outliers[cols].head(10).to_string(index=False))
