'''
This script has been modified to only extract rlap values from SNID output files
and save them to a CSV file. All age calculation functionalities have been removed.
'''
import os
import csv

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


def get_output_files(directory):
    """Get all .output files from the specified directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.output')]


def save_results(results, output_file='cfa_ext_corrected_SNID_rlap_values.csv'):
    """Save spectrum names and their corresponding rlap values to a CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Spectrum', 'rlap_values'])
        for spec, rlap_list in results:
            rlap_str = ' '.join(map(str, rlap_list))
            writer.writerow([spec, rlap_str])


if __name__ == '__main__':
    # --- Configuration ---
    TARGET_DIRECTORY = '/Users/pxm588@student.bham.ac.uk/Desktop/snid/cfaspec_snIa/full_cfa_spectra_extinction_corrected/corrected_spectra'
    OUTPUT_FILENAME = 'cfa_ext_corrected_SNID_rlap_values.csv'

    # --- Main Processing ---
    results = []
    all_files = get_output_files(TARGET_DIRECTORY)

    print(f"Found {len(all_files)} files to process in '{os.path.basename(TARGET_DIRECTORY)}'...")

    for output_file in all_files:
        spectrum_name = os.path.basename(output_file).replace('.output', '')
        print(f"--- Processing {spectrum_name} ---")

        SN_ages, SN_rlaps = parse_snid_file(output_file)

        if SN_rlaps is None:
            print(f"Could not extract any data from {spectrum_name}. Skipping.")
            continue
        
        print(f"Parsed {len(SN_rlaps)} total template matches.")
        
        results.append((spectrum_name, SN_rlaps))

    if results:
        save_results(results, output_file=OUTPUT_FILENAME)
        print(f"\n✅ Successfully processed {len(results)} files. Results saved to '{OUTPUT_FILENAME}'.")
    else:
        print("\nNo files were successfully processed.")
