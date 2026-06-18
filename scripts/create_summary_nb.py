import json

notebook_path = "/Users/pxm588@student.bham.ac.uk/PhD/bayesian_framework/notebooks/cfa_all_fits_summary.ipynb"

cells = []

# Cell 0: Markdown header
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# CfA Supernova Spectra: Nested Sampling Phase Determination Summary\n",
        "\n",
        "This notebook compiles and analyzes the results of fitting the CfA SN Ia spectral database using our single-spectrum nested sampling framework. We restrict the sample to rest-frame phases between $-12$ and $+35$ days.\n",
        "\n",
        "### Key Analysis Steps:\n",
        "1. Load the results from `data/cfa_fits_results.csv`.\n",
        "2. Compute the spectroscopically derived rest-frame phase:\n",
        "   $$A_{\\text{spec}} = \\frac{T_{\\text{spec}} - t_0}{1 + z}$$\n",
        "3. Compare $A_{\\text{spec}}$ against the true photometric rest-frame phase $A_{\\text{true}}$ (from `Age_(days)`).\n",
        "4. Calculate statistics on the residuals: bias, scatter (standard deviation), and robust scatter (MAD).\n",
        "5. Plot the correlation between true and estimated phase, residual histograms, and phase-dependent trends."
    ]
})

# Cell 1: Imports
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from scipy.stats import norm\n",
        "\n",
        "%matplotlib inline\n",
        "plt.style.use('default')\n",
        "plt.rcParams['figure.figsize'] = (10, 6)\n",
        "plt.rcParams['font.size'] = 11\n",
        "plt.rcParams['axes.grid'] = True\n",
        "plt.rcParams['grid.alpha'] = 0.2"
    ]
})

# Cell 2: Load and clean data
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "csv_path = '../data/cfa_fits_results.csv'\n",
        "if not os.path.exists(csv_path):\n",
        "    raise FileNotFoundError(f\"Results file not found at {csv_path}. Make sure the fitting script has run and saved results.\")\n",
        "\n",
        "df = pd.read_csv(csv_path)\n",
        "print(f\"Total rows loaded: {len(df)}\")\n",
        "\n",
        "# Filter to successful fits\n",
        "df_success = df[df['Status'] == 'Success'].copy()\n",
        "print(f\"Successful fits: {len(df_success)} ({len(df_success)/len(df)*100:.1f}% of total)\")\n",
        "\n",
        "# Compute estimated phase and error\n",
        "df_success['phase_est'] = (df_success['MJD'] - df_success['t0']) / (1.0 + df_success['redshift'])\n",
        "df_success['phase_est_err'] = df_success['t0_err'] / (1.0 + df_success['redshift'])\n",
        "df_success['phase_diff'] = df_success['phase_est'] - df_success['Age_(days)']\n",
        "\n",
        "df_success.head(10)"
    ]
})

# Cell 3: Metrics calculations
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "mean_diff = df_success['phase_diff'].mean()\n",
        "median_diff = df_success['phase_diff'].median()\n",
        "std_diff = df_success['phase_diff'].std()\n",
        "\n",
        "# Robust scatter using MAD\n",
        "mad_diff = np.median(np.abs(df_success['phase_diff'] - median_diff))\n",
        "robust_std = 1.4826 * mad_diff\n",
        "\n",
        "print(\"=== Phase Determination Metrics ===\")\n",
        "print(f\"Mean Bias (Est - True):  {mean_diff:+.3f} days\")\n",
        "print(f\"Median Bias:            {median_diff:+.3f} days\")\n",
        "print(f\"Scatter (Std Dev):       {std_diff:.3f} days\")\n",
        "print(f\"Robust Scatter (MAD):    {robust_std:.3f} days\")\n",
        "print(f\"Fraction within 1 day:   {(np.abs(df_success['phase_diff']) <= 1.0).mean()*100:.1f}%\")\n",
        "print(f\"Fraction within 2 days:  {(np.abs(df_success['phase_diff']) <= 2.0).mean()*100:.1f}%\")"
    ]
})

# Cell 4: Plot 1: Age vs Age Correlation
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "plt.figure(figsize=(8, 8))\n",
        "plt.errorbar(df_success['Age_(days)'], df_success['phase_est'], \n",
        "             yerr=df_success['phase_est_err'], \n",
        "             fmt='o', color='#2b5c8f', ecolor='#a5c0db', alpha=0.5, capsize=0, elinewidth=0.8,\n",
        "             label=f'CfA Spectra (N={len(df_success)})')\n",
        "\n",
        "# Plot 1-to-1 line\n",
        "limits = [-15, 38]\n",
        "plt.plot(limits, limits, color='#e05a47', linestyle='--', lw=2, label='1-to-1 Line')\n",
        "\n",
        "plt.xlim(limits)\n",
        "plt.ylim(limits)\n",
        "plt.xlabel('True Photometric Restframe Phase (days)', fontsize=12)\n",
        "plt.ylabel('Derived Spectroscopic Restframe Phase (days)', fontsize=12)\n",
        "plt.title('SALT3 Age Recovery for CfA Supernova Spectra', fontsize=14, pad=15)\n",
        "plt.legend(loc='upper left', fontsize=11)\n",
        "plt.tight_layout()\n",
        "plt.savefig('../outputs/cfa_age_recovery.png', dpi=300)\n",
        "plt.show()"
    ]
})

# Cell 5: Plot 2: Residual Histogram
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "plt.figure(figsize=(10, 6))\n",
        "residuals = df_success['phase_diff']\n",
        "\n",
        "# Fit normal distribution\n",
        "mu, std_fit = norm.fit(residuals)\n",
        "\n",
        "n, bins, patches = plt.hist(residuals, bins=40, range=(-10, 10), density=True, \n",
        "                            color='#3f8f8f', edgecolor='white', alpha=0.7, label='Residuals')\n",
        "\n",
        "xmin, xmax = plt.xlim()\n",
        "x_plot = np.linspace(xmin, xmax, 200)\n",
        "p_plot = norm.pdf(x_plot, mu, std_fit)\n",
        "plt.plot(x_plot, p_plot, color='#e05a47', lw=2.5, \n",
        "         label=f'Gaussian Fit ($\\mu$={mu:+.2f}d, $\\sigma$={std_fit:.2f}d)')\n",
        "\n",
        "plt.xlabel('Residual: Spectroscopic - Photometric Phase (days)', fontsize=12)\n",
        "plt.ylabel('Probability Density', fontsize=12)\n",
        "plt.title('Distribution of Phase Residuals (Spectroscopic - Photometric)', fontsize=14, pad=15)\n",
        "plt.legend(fontsize=11)\n",
        "plt.tight_layout()\n",
        "plt.savefig('../outputs/cfa_phase_residuals_hist.png', dpi=300)\n",
        "plt.show()"
    ]
})

# Cell 6: Plot 3: Residuals vs Phase
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "plt.figure(figsize=(10, 6))\n",
        "plt.scatter(df_success['Age_(days)'], df_success['phase_diff'], \n",
        "            color='#2b5c8f', alpha=0.4, s=20, label='Individual Epochs')\n",
        "\n",
        "plt.axhline(0, color='black', linestyle='-', lw=1.5, alpha=0.5)\n",
        "plt.axhline(1, color='gray', linestyle='--', lw=1, alpha=0.5)\n",
        "plt.axhline(-1, color='gray', linestyle='--', lw=1, alpha=0.5)\n",
        "\n",
        "# Compute binned running median/mean\n",
        "bins = np.arange(-12, 36, 4)\n",
        "bin_centers = 0.5 * (bins[:-1] + bins[1:])\n",
        "binned_mean = []\n",
        "binned_std = []\n",
        "\n",
        "for i in range(len(bins)-1):\n",
        "    mask_bin = (df_success['Age_(days)'] >= bins[i]) & (df_success['Age_(days)'] < bins[i+1])\n",
        "    if mask_bin.sum() > 2:\n",
        "        binned_mean.append(df_success.loc[mask_bin, 'phase_diff'].median())\n",
        "        # robust scatter\n",
        "        mad_bin = np.median(np.abs(df_success.loc[mask_bin, 'phase_diff'] - binned_mean[-1]))\n",
        "        binned_std.append(1.4826 * mad_bin)\n",
        "    else:\n",
        "        binned_mean.append(np.nan)\n",
        "        binned_std.append(np.nan)\n",
        "\n",
        "plt.errorbar(bin_centers, binned_mean, yerr=binned_std, fmt='s-', color='#e05a47', \n",
        "             capsize=4, elinewidth=2, capthick=2, label='Running Binned Median (robust $\\sigma$)')\n",
        "\n",
        "plt.xlabel('True Photometric Restframe Phase (days)', fontsize=12)\n",
        "plt.ylabel('Residual: Spectroscopic - Photometric Phase (days)', fontsize=12)\n",
        "plt.title('Phase residual trends as a function of epoch', fontsize=14, pad=15)\n",
        "plt.ylim(-10, 10)\n",
        "plt.legend(loc='lower left', fontsize=11)\n",
        "plt.tight_layout()\n",
        "plt.savefig('../outputs/cfa_residuals_vs_phase.png', dpi=300)\n",
        "plt.show()"
    ]
})

# Cell 7: Plot 4: Chi2_red vs Phase
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "plt.figure(figsize=(10, 6))\n",
        "plt.scatter(df_success['Age_(days)'], df_success['chi2_red'], \n",
        "            color='#7f5f9f', alpha=0.5, s=20, label='Pre-fit $\\chi^2_{\\text{red}}$')\n",
        "\n",
        "# running median of chi2_red\n",
        "binned_chi2 = []\n",
        "for i in range(len(bins)-1):\n",
        "    mask_bin = (df_success['Age_(days)'] >= bins[i]) & (df_success['Age_(days)'] < bins[i+1])\n",
        "    if mask_bin.sum() > 2:\n",
        "        binned_chi2.append(df_success.loc[mask_bin, 'chi2_red'].median())\n",
        "    else:\n",
        "        binned_chi2.append(np.nan)\n",
        "\n",
        "plt.plot(bin_centers, binned_chi2, 'o-', color='#e05a47', label='Running Median $\\chi^2_{\\text{red}}$')\n",
        "\n",
        "plt.yscale('log')\n",
        "plt.xlabel('True Photometric Restframe Phase (days)', fontsize=12)\n",
        "plt.ylabel('Best-fit Reduced $\\chi^2$', fontsize=12)\n",
        "plt.title('Goodness of Fit (Reduced $\\chi^2$) vs. Phase', fontsize=14, pad=15)\n",
        "plt.legend(loc='upper right', fontsize=11)\n",
        "plt.tight_layout()\n",
        "plt.savefig('../outputs/cfa_chi2_vs_phase.png', dpi=300)\n",
        "plt.show()"
    ]
})

# Construct notebook JSON
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Notebook {notebook_path} created successfully.")
