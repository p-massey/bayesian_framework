import os
import subprocess
import shutil

# Directories to create
DIRS = [
    'src/fitting',
    'src/data',
    'src/analysis',
    'src/plotting',
    'scripts/comparisons',
    'scripts/outliers',
    'tests',
    'notebooks/cohort_analysis',
    'notebooks/individual_fits',
    'notebooks/legacy_tests'
]

# Mapping from source path to destination path
RENAME_MAP = {
    # Core Science Modules (src/)
    'src/salt3_fitting_group.py': 'src/fitting/salt3.py',
    'src/deredden_all_spectra.py': 'src/data/deredden.py',
    'src/utils/snr_finder_group.py': 'src/analysis/snr_finder_group.py',
    'src/utils/snr_methods_comparison.py': 'src/analysis/snr_methods_comparison.py',
    'src/utils/helper_functions.py': 'src/fitting/helper_functions.py',
    
    # Plotting Modules (src/plotting/)
    'src/plot_cfa_results.py': 'src/plotting/cfa_results.py',
    'src/plot_nuisance_params.py': 'src/plotting/nuisance_params.py',
    'src/plot_nuisance_results.py': 'src/plotting/nuisance_results.py',
    'src/plot_nuisance_snr.py': 'src/plotting/nuisance_snr.py',
    'src/plot_snid_vs_dynesty.py': 'src/plotting/snid_vs_dynesty.py',
    'src/update_plot_with_labels.py': 'src/plotting/update_plot_with_labels.py',
    'src/plot_full_range.py': 'src/plotting/full_range.py',
    'src/plotting_example.py': 'src/plotting/example.py',
    
    # Runner and CLI Scripts (scripts/)
    'src/main.py': 'scripts/run_single_fit_legacy.py',
    'src/main_nuisance.py': 'scripts/run_nuisance_fit_legacy.py',
    'src/generate_plots_new.py': 'scripts/generate_plots.py',
    'src/comprehensive_analysis.py': 'scripts/run_comprehensive_analysis.py',
    'src/add_dm15.py': 'scripts/add_dm15.py',
    'src/calculate_residuals.py': 'scripts/calculate_residuals.py',
    'src/compare_methods.py': 'scripts/comparisons/compare_methods.py',
    'src/compare_methods_binned.py': 'scripts/comparisons/compare_methods_binned.py',
    'src/compare_all_methods_binned.py': 'scripts/comparisons/compare_all_methods_binned.py',
    'src/find_outliers.py': 'scripts/outliers/find_outliers.py',
    'src/find_filtered_outliers.py': 'scripts/outliers/find_filtered_outliers.py',
    'src/export_outliers.py': 'scripts/outliers/export_outliers.py',
    
    # Tests and Validation (tests/)
    'src/test_cfa_spectra.py': 'tests/test_cfa_spectra.py',
    'src/test_methods.py': 'tests/test_methods.py',
    'src/test_nlive_convergence.py': 'tests/test_nlive_convergence.py',
    
    # Notebooks (notebooks/)
    'notebooks/cfa_all_fits_summary.ipynb': 'notebooks/cohort_analysis/cfa_all_fits_summary.ipynb',
    'notebooks/cfa_spectra_analysis.ipynb': 'notebooks/cohort_analysis/cfa_spectra_analysis.ipynb',
    'notebooks/cfa_supersnid_bootstrap_pipeline.ipynb': 'notebooks/cohort_analysis/cfa_supersnid_bootstrap_pipeline.ipynb',
    'notebooks/snid_method_comparison_metrics_with_filters.ipynb': 'notebooks/cohort_analysis/snid_method_comparison_metrics_with_filters.ipynb',
    
    'notebooks/fit_2026ngr.ipynb': 'notebooks/individual_fits/fit_2026ngr.ipynb',
    'notebooks/hope_spectra_analysis.ipynb': 'notebooks/individual_fits/hope_spectra_analysis.ipynb',
    'notebooks/hope_spectra_joint_fit.ipynb': 'notebooks/individual_fits/hope_spectra_joint_fit.ipynb',
    'notebooks/single_spectrum_fit.ipynb': 'notebooks/individual_fits/single_spectrum_fit.ipynb',
    'notebooks/single_spectrum_fit_gp.ipynb': 'notebooks/individual_fits/single_spectrum_fit_gp.ipynb',
    'notebooks/single_spectrum_fit_intrinsic_dispersion.ipynb': 'notebooks/individual_fits/single_spectrum_fit_intrinsic_dispersion.ipynb',
    'notebooks/single_spectrum_fit_scaled_binned.ipynb': 'notebooks/individual_fits/single_spectrum_fit_scaled_binned.ipynb',
    
    'notebooks/outlier_spectra_analysis.ipynb': 'notebooks/legacy_tests/outlier_spectra_analysis.ipynb',
    'notebooks/plotting_spectra.ipynb': 'notebooks/legacy_tests/plotting_spectra.ipynb',
    'notebooks/testing.ipynb': 'notebooks/legacy_tests/testing.ipynb',
    'src/plotting_results.ipynb': 'notebooks/legacy_tests/plotting_results.ipynb',
    'src/plotting_results_restructured.ipynb': 'notebooks/legacy_tests/plotting_results_restructured.ipynb',
    'src/testing.ipynb': 'notebooks/legacy_tests/testing_src.ipynb'
}

def main():
    workspace = '/Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework'
    os.chdir(workspace)
    
    # Create necessary directories
    for d in DIRS:
        path = os.path.join(workspace, d)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {d}")
            
    # Perform git mv
    for src, dst in RENAME_MAP.items():
        if os.path.exists(src):
            try:
                subprocess.run(['git', 'mv', src, dst], check=True)
                print(f"Git mv: {src} -> {dst}")
            except Exception as e:
                print(f"Failed git mv for {src} to {dst}: {e}")
        else:
            print(f"Source file not found (skipping): {src}")
            
    # Clean up empty directories
    utils_dir = 'src/utils'
    if os.path.exists(utils_dir) and not os.listdir(utils_dir):
        os.rmdir(utils_dir)
        print(f"Removed empty directory: {utils_dir}")
        
    print("Restructuring script execution finished.")

if __name__ == '__main__':
    main()
