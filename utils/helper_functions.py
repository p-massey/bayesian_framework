import csv
import numpy as np
import pandas as pd
import os
import statistics
import random

def parse_snid_file(filename):
    """
    Parses a SNID output file to extract all ages, rlaps, and redshifts
    until the rlap cutoff is reached.

    Args:
        filename (str): The path to the SNID output file.

    Returns:
        tuple: (list of ages, list of rlaps, list of redshifts) or (None, None, None) if parsing fails.
    """
    ages = []
    rlaps = []
    redshifts = []
    in_rlap_section = False

    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('### rlap-ordered template listings ###'):
                    in_rlap_section = True
                    next(f)  # Skip the header line
                    continue

                if in_rlap_section:
                    if line.startswith('#--- rlap cutoff'):
                        break
                    if line.startswith('#') or not line:
                        continue

                    parts = line.split()
                    if len(parts) > 9 and parts[9] == 'good':
                        try:
                            age = float(parts[7])
                            rlap = float(parts[4])
                            z = float(parts[5])
                            ages.append(age)
                            rlaps.append(rlap)
                            redshifts.append(z)
                        except (IndexError, ValueError) as e:
                            print(f"Could not parse line in {os.path.basename(filename)}: {line}\nError: {e}")
                            continue

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while reading {filename}: {e}")
        return None, None, None

    return (ages, rlaps, redshifts) if ages else (None, None, None)


def calculate_simple_mean(ages, rlaps, top_n=20):
    """
    Calculates the simple mean and standard deviation from the top N fits,
    sorted by RLAP value.

    Returns:
        tuple: (standard deviation, mean age).
    """
    if len(ages) < 2:
        return None, None

    sorted_fits = sorted(zip(ages, rlaps), key=lambda x: x[1], reverse=True)
    top_fits = sorted_fits[:top_n]

    if len(top_fits) < 2:
        return None, None

    top_ages = [fit[0] for fit in top_fits]
    mean_age = statistics.mean(top_ages)
    std_dev = statistics.stdev(top_ages)

    # MODIFIED: Return only the std_dev and mean_age
    return std_dev, mean_age


def calculate_median_age(ages, rlaps, top_n=8):
    """
    Calculates the median and standard deviation from the full list of ages.

    Returns:
        tuple: (standard deviation, median age).
    """
    if len(ages) < 2:
        return None, None

    sorted_fits = sorted(zip(ages, rlaps), key=lambda x: x[1], reverse=True)
    top_fits = sorted_fits[:top_n]

    if len(top_fits) < 2:
        return None, None

    top_ages = [fit[0] for fit in top_fits]
    if not ages or len(ages) < 2:
        return None, None

    median_age = statistics.median(top_ages)
    std_dev = statistics.stdev(top_ages)

    return std_dev, median_age


def calculate_bootstrap_age(ages, rlaps, top_n):
    """
    Calculates the mean age and uncertainty by concatenating top fits from bootstrap samples.

    Returns:
        tuple: (standard deviation, mean age).
    """
    if len(ages) < top_n:
        return None, None

    if top_n == 0:
        return None, None

    trial_means = []
    for _ in range(len(ages)):
        indices = random.choices(range(len(ages)), k=len(ages))
        random_ages = [ages[i] for i in indices]
        random_rlaps = [rlaps[i] for i in indices]

        sorted_pairs = sorted(zip(random_ages, random_rlaps), key=lambda x: x[1], reverse=True)

        if not sorted_pairs:
            continue

        top_pairs = sorted_pairs[:top_n]

        top_ages_from_sample = [pair[0] for pair in top_pairs]

        mean_age = sum(top_ages_from_sample) / len(top_ages_from_sample)
        trial_means.append(mean_age)

    if len(trial_means) < 2:
        return None, None

    final_mean = statistics.mean(trial_means)
    final_stdev = statistics.stdev(trial_means)

    return final_stdev, final_mean

def calculate_bootstrap_median_topn(ages, rlaps, top_n):
    """
    Calculates the mean age and uncertainty by concatenating top fits from bootstrap samples.

    Returns:
        tuple: (standard deviation, mean age).
    """
    if len(ages) < top_n:
        return None, None

    if top_n == 0:
        return None, None

    trial_medians = []
    for _ in range(len(ages)):
        indices = random.choices(range(len(ages)), k=len(ages))
        random_ages = [ages[i] for i in indices]
        random_rlaps = [rlaps[i] for i in indices]

        sorted_pairs = sorted(zip(random_ages, random_rlaps), key=lambda x: x[1], reverse=True)

        if not sorted_pairs:
            continue

        top_pairs = sorted_pairs[:top_n]

        top_ages_from_sample = [pair[0] for pair in top_pairs]

        median_age = statistics.median(top_ages_from_sample)
        trial_medians.append(median_age)

    if len(trial_medians) < 2:
        return None, None

    final_median = statistics.median(trial_medians)
    final_stdev = statistics.stdev(trial_medians)

    return final_stdev, final_median


def calculate_bootstrap_median(ages):
    """
    Calculates the mean age and uncertainty by concatenating top fits from bootstrap samples.

    Returns:
        tuple: (standard deviation, mean age).
    """

    trial_medians = []
    for _ in range(len(ages)):
        indices = random.choices(range(len(ages)), k=len(ages))
        random_ages = [ages[i] for i in indices]

        mean_age = statistics.median(random_ages)
        trial_medians.append(mean_age)

    if len(trial_medians) < 2:
        return None, None

    final_median = statistics.median(trial_medians)
    final_stdev = statistics.stdev(trial_medians)

    return final_stdev, final_median


def get_output_files(directory):
    """Get all .output files from the specified directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.output')]


def save_results(results, output_file='cfa_SNID_age_results_concatenated.csv'):
    """Save all results to a single CSV file."""
    with open(output_file, 'w') as f:
        f.write(
            'Spectrum,Simple_Mean_Age,Simple_Mean_StdDev,'
            'Bootstrap_Concat_Mean_Age,Bootstrap_Concat_Uncertainty\n'
        )
        for spec, sm_mean, sm_std, bs_mean, bs_std in results:
            f.write(f'{spec},'
                    f'{sm_mean if sm_mean is not None else -1.00:.2f},'
                    f'{sm_std if sm_std is not None else -1.00:.2f},'
                    f'{bs_mean if bs_mean is not None else -1.00:.2f},'
                    f'{bs_std if bs_std is not None else -1.00:.2f}\n')

def extract_spec_name(fn):
    fn = fn.lower()
    if fn.startswith('snf'):
        # For snf20080514-002-20080528..., we want the first 2 parts
        parts = fn.split('-')
        return f"{parts[0]}-{parts[1]}"
    else:
        # For sn2008bf-20080415..., we just want the first part
        return fn.split('-')[0]

def normalize_param_name(name):
    name = str(name).lower()
    return name if name.startswith('sn') else f"sn{name}"

def calculate_sn_ages(params_file, spectra_mjd_file, output_file):
    try:
        # --- 1. Load Parameters ---
        params = pd.read_csv(
            params_file, comment='#', delim_whitespace=True, header=None,
            usecols=[0, 1, 2, 3], names=['SN_name', 'zhel', 'mjd_max', 'mjd_max_err']
        )

        params['join_key'] = params['SN_name'].apply(normalize_param_name)

        # KEY CHANGE: Create a mask for valid dates instead of dropping rows
        # This keeps the SN_name available for the merge even if math isn't possible
        params['is_valid_mjd'] = params['mjd_max'] < 99990

        # --- 2. Load Spectra ---
        spectra = pd.read_csv(
            spectra_mjd_file, comment='#', delim_whitespace=True,
            header=None, names=['filename', 'mjd_obs']
        )
        spectra['join_key'] = spectra['filename'].apply(extract_spec_name)

        # --- 3. Merge and Calculate ---
        df = pd.merge(spectra, params, on='join_key', how='left')

        # Rest-frame age: (t_obs - t_max) / (1 + z)
        # Using .where ensures we only calculate for rows with valid MJD data
        mask = df['is_valid_mjd'] == True
        df.loc[mask, 'Age'] = (df['mjd_obs'] - df['mjd_max']) / (1 + df['zhel'])
        df.loc[mask, 'Age_Unc'] = df['mjd_max_err'] / (1 + df['zhel'])

        # --- 4. Final Export ---
        output_df = df[['filename', 'SN_name', 'Age', 'Age_Unc', 'zhel']].copy()
        output_df.columns = ["Filename", "SN_Name", "Age_(days)", "Age_Unc_(days)", "redshift"]

        # Ensure SN_Name isn't lost if the merge worked but Age didn't
        output_df.to_csv(output_file, index=False)

        print(f"Processed {len(output_df)} spectra. Results saved to {output_file}")

        return output_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def find_sn_subtypes(spectra_file, classification_file, scheme):
    try:
        scheme = scheme.lower()
        target_col = 5 if scheme == 'branch' else 6
        if scheme not in ['branch', 'wang']:
            raise ValueError("Scheme must be 'branch' or 'wang'")

        # 1. Load Classification Table
        # We name this 'subtypes' to avoid confusion
        subtypes = pd.read_csv(
            classification_file,
            comment='#',
            delim_whitespace=True,
            header=None,
            usecols=[0, target_col],
            names=['SN_Name', 'Subtype']
        )
        # Create the join key here to match the spectra
        subtypes['join_key'] = subtypes['SN_Name'].apply(normalize_param_name)

        # 2. Load Spectra File
        spectra = pd.read_csv(
            spectra_file, comment='#', delim_whitespace=True,
            header=None, usecols=[0], names=['Filename']
        )
        # Create the matching join key
        spectra['join_key'] = spectra['Filename'].apply(extract_spec_name)

        # 3. Merge
        # We use 'left' to keep all spectra even if they don't have a subtype
        final_df = pd.merge(spectra, subtypes[['join_key', 'Subtype']], on='join_key', how='left')

        # Cleanup: rename join_key to SN_Name for the final output
        final_df = final_df.rename(columns={'join_key': 'SN_Name'})

        print(f"Successfully matched {len(final_df)} spectra to {scheme} subtypes.")
        return final_df[['Filename', 'SN_Name', 'Subtype']]

    except Exception as e:
        print(f"Error in find_sn_subtypes: {e}")
        return None
