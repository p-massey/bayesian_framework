import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import george
from george import kernels
from scipy.optimize import minimize

# --- 1. Define File Paths ---
CFA_AGES_FILE = '../cfa_all_spectra_ages_filtered.txt'
SNID_RESULTS_FILE = 'cfa_SNID_age_results_weighted_bootstrap.csv'
OUTPUT_PLOT_FILE = 'cfa_delta_vs_true_age_weighted_bootstrap_plot.png'
OUTPUT_NAMES_FILE = 'plotted_spectra_names_weighted_bootstrap.csv'

def load_and_prepare_data(cfa_file, snid_file):
    """
    Loads data from text and CSV files, merges them, and returns NumPy arrays.
    """
    cfa_data = {}

    # --- Read CFA data into a dictionary ---
    try:
        with open(cfa_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                if i <= 2: continue
                parts = line.strip().split()
                if not parts: continue
                try:
                    base_name = parts[0].removesuffix('.flm')
                    cfa_data[base_name] = {
                        'true_age': float(parts[2]),
                        'true_unc': float(parts[3])
                    }
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        print(f"Error: The file '{cfa_file}' was not found.")
        return None, None, None, None

    print(f"Number of items in CFA file: {len(cfa_data)}")

    # --- Read SNID data into a dictionary ---
    snid_data = {}
    try:
        with open(snid_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                base_name = row['Spectrum'].removesuffix('_snid')
                snid_data[base_name] = {
                    'bootstrap_age': float(row['Weighted_Mean_Age']),
                    'bootstrap_unc': float(row['Bootstrap_Uncertainty'])
                }
    except FileNotFoundError:
        print(f"\nError: The file '{snid_file}' was not found.")
        return None, None, None, None
    except (KeyError, ValueError):
        print(f"\nError: Could not parse required columns in '{snid_file}'.")
        return None, None, None, None

    print(f"\nNumber of items in SNID file: {len(snid_data)}")

    # --- Merge data and create lists for plotting ---
    x_vals, y_vals, y_errs, base_names = [], [], [], []
    for base_name, snid_info in snid_data.items():
        if base_name in cfa_data:
            cfa_info = cfa_data[base_name]
            delta = cfa_info['true_age'] - snid_info['bootstrap_age']
            delta_unc = np.sqrt(cfa_info['true_unc'] ** 2 + snid_info['bootstrap_unc'] ** 2)
            x_vals.append(cfa_info['true_age'])
            y_vals.append(delta)
            y_errs.append(delta_unc)
            base_names.append(base_name)

    if not x_vals:
        print("Found 0 matching spectra between the two files.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    print(f"\nFound {len(x_vals)} matching spectra between the two files.")

    sorted_indices = np.argsort(x_vals)
    return (np.array(x_vals)[sorted_indices],
            np.array(y_vals)[sorted_indices],
            np.array(y_errs)[sorted_indices],
            np.array(base_names)[sorted_indices])

def save_names_to_csv(names, filename):
    """
    Saves a list of spectrum names to a single-column CSV file.
    """
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['SpectrumName'])
            for name in names:
                writer.writerow([name])
        print(f"Saved {len(names)} spectrum names to '{filename}'")
    except IOError as e:
        print(f"\nError: Could not write to file '{filename}'. Reason: {e}")

def main():
    """
    Main function to perform segmented Gaussian Process regression and plot the results.
    """
    x, y, yerr, names = load_and_prepare_data(CFA_AGES_FILE, SNID_RESULTS_FILE)

    if x is None or x.size == 0:
        print("Data loading error or no matching data. Exiting.")
        return

    save_names_to_csv(names, OUTPUT_NAMES_FILE)
    
    # The rest of the main function remains unchanged...
    bin_width = 30
    boundaries = [x.min(), x.max()]
    print("\n--- Fitting Segmented Gaussian Process ---")
    all_x_pred, all_pred, all_pred_std = [], [], []

    for i in range(len(boundaries) - 1):
        lower_bound, upper_bound = boundaries[i], boundaries[i + 1]
        mask_seg = (x >= lower_bound) & (x < upper_bound)
        x_seg, y_seg, yerr_seg = x[mask_seg], y[mask_seg], yerr[mask_seg]

        if len(x_seg) < 2:
            print(f"Segment [{lower_bound:.1f}, {upper_bound:.1f}) days: Skipped (only {len(x_seg)} data point).")
            continue

        initial_ell = x_seg.max() - x_seg.min()
        initial_metric = max(initial_ell ** 2, 1.0)
        print(
            f"Segment [{lower_bound:.1f}, {upper_bound:.1f}) days: Fitting with {len(x_seg)} data points... Initial metric: {initial_metric:.2f}")

        kernel = np.var(y_seg) * kernels.ExpSquaredKernel(metric=initial_metric)
        gp = george.GP(kernel)
        gp.compute(x_seg, yerr_seg)

        def neg_ln_like(p, gp=gp, y_seg=y_seg):
            gp.set_parameter_vector(p)
            ll = gp.log_likelihood(y_seg, quiet=True)
            return -ll if np.isfinite(ll) else 1e25

        def grad_neg_ln_like(p, gp=gp, y_seg=y_seg):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(y_seg, quiet=True)

        result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
        gp.set_parameter_vector(result.x)
        final_metric = np.exp(gp.get_parameter_vector()[1])
        print(f"   -> Optimized metric: {final_metric:.2f}")

        x_pred_seg = np.linspace(lower_bound, upper_bound, 200)
        pred_seg, pred_var_seg = gp.predict(y_seg, x_pred_seg, return_var=True)
        pred_std_seg = np.sqrt(pred_var_seg)

        all_x_pred.append(x_pred_seg)
        all_pred.append(pred_seg)
        all_pred_std.append(pred_std_seg)

    if not all_x_pred:
        print("No segments had enough data to fit the GP. No plot will be generated.")
        return

    x_pred = np.concatenate(all_x_pred)
    pred = np.concatenate(all_pred)
    pred_std = np.concatenate(all_pred_std)

    # --- Create and Save the Plot ---
    plt.style.use('../GausSN.mplstyle')
    fig, ax = plt.subplots(figsize=(12, 8))
    three_sigma_mask = np.abs(y) > 3 * yerr
    ax.errorbar(x[~three_sigma_mask], y[~three_sigma_mask], yerr=yerr[~three_sigma_mask],
                fmt=".k", capsize=0, label='Data (within 3σ)')
    if np.any(three_sigma_mask):
        ax.errorbar(x[three_sigma_mask], y[three_sigma_mask], yerr=yerr[three_sigma_mask],
                    fmt=".", color='red', capsize=0, label='Data (>3σ from zero)')
    ax.plot(x_pred, pred, color='green', label='GP Mean Prediction')
    ax.fill_between(x_pred, pred - pred_std, pred + pred_std, color='green', alpha=0.2,
                    label='GP 1-sigma Confidence')
    ax.axhline(0, color='b', linestyle='--', alpha=0.75, zorder=0, label='Perfect Agreement')
    ax.set_xlabel('True Age (days)')
    ax.set_ylabel('Delta (True Age - SNID Age)')
    ax.set_title('Delta vs. True Age for Weighted Bootstrap Method')
    ax.legend(loc='best')
    ax.set_ylim(-30, 35)
    ax.set_xlim(-20, 70)

    output_dir = os.path.dirname(OUTPUT_PLOT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(OUTPUT_PLOT_FILE, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved to {OUTPUT_PLOT_FILE}")


if __name__ == "__main__":
    main()
