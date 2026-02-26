import csv
import numpy as np
import pandas as pd
import os
import statistics
import random

def parse_snid_file(filename):
    """
    Parses a SNID output file to extract all ages and rlap values
    until the rlap cutoff is reached.

    Args:
        filename (str): The path to the SNID output file.

    Returns:
        tuple: (list of all ages, list of all rlaps) or (None, None) if parsing fails.
    """
    ages = []
    rlaps = []
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
                            ages.append(age)
                            rlaps.append(rlap)
                        except (IndexError, ValueError) as e:
                            print(f"Could not parse line in {os.path.basename(filename)}: {line}\nError: {e}")
                            continue

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while reading {filename}: {e}")
        return None, None

    return (ages, rlaps) if ages else (None, None)


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

