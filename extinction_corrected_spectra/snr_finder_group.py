import numpy as np
import matplotlib.pyplot as plt
import os
import re
import csv
import argparse


# --- Helper functions (Unchanged) ---
def extract_sn_name(spectrum_file):
    """
    Extracts the supernova name from a filename, handling multiple formats.
    """
    basename = os.path.basename(spectrum_file)
    match = re.search(r'(snf\d{8}-\d{3})|(\d{4}[a-zA-Z]{1,2})', basename, re.IGNORECASE)
    if match:
        return match.group(1) or match.group(2)
    return None


def get_redshift(sn_name, params_file):
    """
    Finds the redshift for a given SN name in the parameter file.

    Args:
        sn_name (str): The name of the supernova.
        params_file (str): The path to the supernova parameters file.

    Returns:
        float or None: The redshift value or None if not found.
    """
    try:
        with open(params_file, 'r') as f:
            for line in f:
                if line.strip().startswith('#') or not line.strip():
                    continue
                parts = line.split()
                # Check if the line has enough columns and the name matches (case-insensitively)
                # --- THIS IS THE CORRECTED LINE ---
                if len(parts) > 1 and parts[0].lower() == sn_name.lower():
                    return float(parts[1])
    except FileNotFoundError:
        print(f"Error: Parameter file '{params_file}' not found.")
        return None
    except Exception as e:
        print(f"Error reading parameter file: {e}")
        return None
    return None


# --- Refactored Core Calculation and Plotting Function ---
def analyze_spectrum(spectrum_file, params_file, make_plot=False, plot_dir="snr_plots"):
    """
    Performs the fully adaptive S/N calculation for a single spectrum.

    This function is designed to be called in a loop. It returns its results
    as a dictionary for easy collection and CSV writing.

    Args:
        spectrum_file (str): Path to the spectrum data file.
        params_file (str): Path to the supernova parameters file.
        make_plot (bool): If True, generate and save a diagnostic plot.
        plot_dir (str): Directory where plots will be saved.

    Returns:
        dict or None: A dictionary with the results, or None if analysis fails.
    """
    # --- Adaptive Parameters (Rest Frame Angstroms) ---
    SEARCH_WINDOW_REST = (5800, 6400)
    CONTINUUM_WIDTH = 50
    CONTINUUM_PADDING = 50

    # --- Load Data and Get Redshift ---
    try:
        wavelength, flux, uncertainty = np.loadtxt(spectrum_file, unpack=True)
    except Exception as e:
        print(f"  > Error loading spectrum file: {e}")
        return None

    sn_name = extract_sn_name(spectrum_file)
    if not sn_name:
        print(f"  > Could not extract a valid SN name from '{os.path.basename(spectrum_file)}'. Skipping.")
        return None

    redshift = get_redshift(sn_name, params_file)
    if redshift is None:
        print(f"  > Could not find redshift for SN {sn_name}. Skipping.")
        return None

    # --- Step 1: Broad Search for the Feature Minimum ---
    observed_search_window = tuple(wl * (1 + redshift) for wl in SEARCH_WINDOW_REST)
    search_mask = (wavelength >= observed_search_window[0]) & (wavelength <= observed_search_window[1])
    if not np.any(search_mask):
        print(f"  > Error: No data in broad search window for {sn_name}. Skipping.")
        return None

    min_flux_idx_global = np.where(wavelength == wavelength[search_mask][np.argmin(flux[search_mask])])[0][0]
    found_min_wavelength = wavelength[min_flux_idx_global]
    min_flux = flux[min_flux_idx_global]

    # --- Step 2: First-Pass Continuum Fit to Find Feature Edges ---

    # Make a first guess at the feature width
    temp_feature_width_obs = 150 * (1 + redshift)

    # calculate the edges of the blue and red regions
    blue_start = found_min_wavelength - (temp_feature_width_obs / 2) - (CONTINUUM_PADDING * (1 + redshift)) - (
                CONTINUUM_WIDTH * (1 + redshift))
    blue_end = blue_start + (CONTINUUM_WIDTH * (1 + redshift))
    red_start = found_min_wavelength + (temp_feature_width_obs / 2) + (CONTINUUM_PADDING * (1 + redshift))
    red_end = red_start + (CONTINUUM_WIDTH * (1 + redshift))

    # make a mask of the data within the blue and red regions
    continuum_mask_pass1 = ((wavelength >= blue_start) & (wavelength <= blue_end)) | \
                           ((wavelength >= red_start) & (wavelength <= red_end))

    if np.sum(continuum_mask_pass1) < 3:
        print(f"  > Error: Not enough points for first-pass continuum fit for {sn_name}. Skipping.")
        return None

    # make a first fit for the continuum
    coeffs_pass1 = np.polyfit(wavelength[continuum_mask_pass1], flux[continuum_mask_pass1], 2)
    continuum_fit_pass1 = np.poly1d(coeffs_pass1)

    # --- Step 3: Find where the spectrum intersects the first-pass continuum ---

    # find the intersection of the first pass fit with the spectrum
    blue_edge_idx = min_flux_idx_global
    while blue_edge_idx > 0 and flux[blue_edge_idx] < continuum_fit_pass1(wavelength[blue_edge_idx]):
        blue_edge_idx -= 1
    red_edge_idx = min_flux_idx_global
    while red_edge_idx < len(wavelength) - 1 and flux[red_edge_idx] < continuum_fit_pass1(wavelength[red_edge_idx]):
        red_edge_idx += 1

    # find the region of the feature
    observed_feature_region = (wavelength[blue_edge_idx], wavelength[red_edge_idx])
    # find the width of the line
    measured_width = observed_feature_region[1] - observed_feature_region[0]

    # --- Step 4: Define Final Continuum Regions and Perform Second, Accurate Fit ---
    continuum_padding_obs = CONTINUUM_PADDING * (1 + redshift)
    continuum_width_obs = CONTINUUM_WIDTH * (1 + redshift)

    blue_continuum_end = observed_feature_region[0] - continuum_padding_obs
    blue_continuum_start = blue_continuum_end - continuum_width_obs
    red_continuum_start = observed_feature_region[1] + continuum_padding_obs
    red_continuum_end = red_continuum_start + continuum_width_obs
    final_continuum_regions = [(blue_continuum_start, blue_continuum_end), (red_continuum_start, red_continuum_end)]

    continuum_mask_pass2 = ((wavelength >= blue_continuum_start) & (wavelength <= blue_continuum_end)) | \
                           ((wavelength >= red_continuum_start) & (wavelength <= red_continuum_end))

    if np.sum(continuum_mask_pass2) < 3:
        print(f"  > Error: Not enough points for final continuum fit for {sn_name}. Skipping.")
        return None

    # --- Step 5: Calculate Final S/N ---
    coeffs_final = np.polyfit(wavelength[continuum_mask_pass2], flux[continuum_mask_pass2], 2)
    continuum_fit_final = np.poly1d(coeffs_final)

    continuum_level_at_min = continuum_fit_final(found_min_wavelength)
    feature_depth = continuum_level_at_min - min_flux
    noise = None
    # First, try to use the uncertainty column if it exists and has meaningful data
    if uncertainty is not None:
        # Check if any uncertainty values in the continuum are actually greater than zero
        continuum_uncertainties = uncertainty[continuum_mask_pass2]
        if np.any(continuum_uncertainties > 0):
            noise = np.mean(continuum_uncertainties)
        else:
            print(f"  > Note: Uncertainty column for {sn_name} contains only zero or negative values. Falling back to continuum std dev.")

    # If noise couldn't be determined from the uncertainty column, calculate it from flux std dev
    if noise is None:
        (blue_start, blue_end), (red_start, red_end) = final_continuum_regions
        blue_mask = (wavelength >= blue_start) & (wavelength <= blue_end)
        red_mask = (wavelength >= red_start) & (wavelength <= red_end)

        std_devs = []
        if np.sum(blue_mask) > 1:
            std_devs.append(np.std(flux[blue_mask]))
        if np.sum(red_mask) > 1:
            std_devs.append(np.std(flux[red_mask]))

        if not std_devs:
            print(f"  > Error: Not enough data points in any continuum region to calculate std dev for {sn_name}. Skipping.")
            return None

        noise = np.mean(std_devs)
    snr = feature_depth / noise if noise > 0 else float('inf')

    # --- Optional Plotting ---
    if make_plot:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(wavelength, flux, color='black', label='Spectrum Flux', lw=1.5)
        ax.axvspan(observed_feature_region[0], observed_feature_region[1], color='cyan', alpha=0.4,
                   label=f'Adaptive Feature Region (Width={measured_width:.0f} Å)')
        for region in final_continuum_regions:
            ax.axvspan(region[0], region[1], color='red', alpha=0.2)
        ax.plot([], [], color='red', alpha=0.4, lw=8, label='Final Continuum Regions')

        plot_wl_range = np.linspace(final_continuum_regions[0][0], final_continuum_regions[-1][1], 300)
        ax.plot(plot_wl_range, continuum_fit_final(plot_wl_range), 'r--', label='Final Continuum Fit', lw=2)
        ax.plot(plot_wl_range, continuum_fit_pass1(plot_wl_range), color='gray', ls=':', label='First-Pass Fit', lw=2)

        ax.plot(found_min_wavelength, min_flux, 'go', markersize=8, label='Feature Minimum')
        ax.arrow(found_min_wavelength, continuum_level_at_min, 0, -feature_depth, length_includes_head=True,
                 head_width=15, head_length=flux.max() * 0.02, color='green', lw=1.5)

        ax.set_title(f'Fully Adaptive S/N of Si II λ6355 for {sn_name} = {snr:.2f}', fontsize=16)
        ax.set_xlabel('Observed Wavelength (Å)', fontsize=12)
        ax.set_ylabel('Flux', fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        ax.set_xlim(final_continuum_regions[0][0] - 100, final_continuum_regions[-1][1] + 100)
        min_y = min(0, np.min(flux[search_mask]) - np.std(flux[search_mask]))
        max_y = np.max(flux[continuum_mask_pass2]) + np.std(flux[continuum_mask_pass2])
        ax.set_ylim(min_y, max_y * 1.1)

        plt.tight_layout()
        plot_filename = os.path.join(plot_dir, f"{sn_name}_snr_plot.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close(fig)  # Use plt.close() to prevent plots from displaying in batch mode

    # --- Return Results as a Dictionary ---
    return {
        "sn_name": sn_name,
        "snr": f"{snr:.2f}",
        "redshift": f"{redshift:.5f}",
        "feature_depth": f"{feature_depth:.4e}",
        "mean_continuum_noise": f"{noise:.4e}",
        "adaptive_feature_width_A": f"{measured_width:.1f}",
        "min_wavelength_A": f"{found_min_wavelength:.1f}",
        "spectrum_file": os.path.basename(spectrum_file),
    }


# --- Main Function to Process a Directory ---
def process_directory(spectra_dir, params_file, output_csv, plot_each):
    """
    Processes all spectrum files in a directory and saves results to a CSV.
    """
    results = []
    plot_dir = "snr_plots"
    if plot_each:
        os.makedirs(plot_dir, exist_ok=True)  # Create the plot directory if needed

    print(f"Starting analysis on directory: '{spectra_dir}'")
    print("-" * 50)

    # Use os.scandir for efficiency
    for entry in os.scandir(spectra_dir):
        if entry.is_file() and entry.name.endswith('.flm'):
            print(f"Processing {entry.name}...")
            spectrum_path = entry.path
            # Call the refactored analysis function
            result = analyze_spectrum(spectrum_path, params_file, make_plot=plot_each, plot_dir=plot_dir)
            if result:
                results.append(result)
                print(f"  > Success! S/N = {result['snr']}")

    if not results:
        print("\nNo spectra were successfully processed.")
        return

    # Write all collected results to the CSV file
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Successfully processed {len(results)} spectra.")
        print(f"Results have been saved to '{output_csv}'")
        if plot_each:
            print(f"Plots have been saved to the '{plot_dir}' directory.")
    except Exception as e:
        print(f"\n❌ Error writing to CSV file: {e}")


if __name__ == '__main__':
    # --- Hardcoded Configuration ---
    SPECTRA_DIRECTORY = '/Users/pxm588@student.bham.ac.uk/Desktop/snid/cfaspec_snIa/full_cfa_spectra_extinction_corrected/corrected_spectra'
    PARAMS_FILE = '/Users/pxm588@student.bham.ac.uk/PhD/testsuite/cfasnIa_param.dat'
    OUTPUT_CSV_FILE = 'extinction_corrected_snr_results.csv'
    GENERATE_PLOTS = False # Set to True to generate plots

    # Check if the directories/files exist before starting
    if not os.path.isdir(SPECTRA_DIRECTORY):
        print(f"Error: Spectra directory not found at '{SPECTRA_DIRECTORY}'")
    elif not os.path.isfile(PARAMS_FILE):
        print(f"Error: Parameters file not found at '{PARAMS_FILE}'")
    else:
        process_directory(SPECTRA_DIRECTORY, PARAMS_FILE, OUTPUT_CSV_FILE, GENERATE_PLOTS)
