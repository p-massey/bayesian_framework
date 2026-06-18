import numpy as np
import os
import re

def der_snr(flux):
    """
    Calculates the SNR of a spectrum using the DER_SNR algorithm.
    Reference: Stoehr et al. 2008, "DER_SNR: A Robust SNR Estimator"
    
    Args:
        flux (np.ndarray): The flux values of the spectrum.
        
    Returns:
        float: The calculated SNR.
    """
    # Remove NaNs and zeros
    flux = flux[np.isfinite(flux)]
    flux = flux[flux != 0]
    
    if len(flux) < 5:
        return 0.0

    # Calculate noise using the median of absolute differences
    # Noise = 1.2848 * median(|2*f_i - f_{i-1} - f_{i+1}|) / sqrt(6)
    noise = 1.2848 * np.median(np.abs(2.0 * flux[1:-1] - flux[0:-2] - flux[2:])) / np.sqrt(6.0)
    
    # Signal is the median flux
    signal = np.median(flux)
    
    if noise > 0:
        return signal / noise
    return 0.0

def classical_snr(wavelength, flux, window=(5400, 5600)):
    """
    Calculates SNR as Mean Flux / Std Dev in a specific window.
    
    Args:
        wavelength (np.ndarray): Wavelength values.
        flux (np.ndarray): Flux values.
        window (tuple): (min_wl, max_wl) for the calculation.
        
    Returns:
        float: The calculated SNR.
    """
    mask = (wavelength >= window[0]) & (wavelength <= window[1])
    window_flux = flux[mask]
    
    if len(window_flux) < 3:
        return 0.0
    
    mean_flux = np.mean(window_flux)
    std_dev = np.std(window_flux)
    
    if std_dev > 0:
        return mean_flux / std_dev
    return 0.0

def pixel_wise_snr(flux, uncertainty):
    """
    Calculates SNR using the provided uncertainty column.
    
    Args:
        flux (np.ndarray): Flux values.
        uncertainty (np.ndarray): Uncertainty values.
        
    Returns:
        float: Median SNR across all pixels with valid uncertainty.
    """
    if uncertainty is None or not np.any(uncertainty > 0):
        return 0.0
    
    # Avoid division by zero
    mask = (uncertainty > 0) & np.isfinite(flux) & np.isfinite(uncertainty)
    if not np.any(mask):
        return 0.0
        
    snr_pixels = flux[mask] / uncertainty[mask]
    return np.median(snr_pixels)

def feature_depth_snr(wavelength, flux, redshift, uncertainty=None):
    """
    Calculates the Fully Adaptive S/N of Si II λ6355.
    Exactly matching the logic in snr_finder_group.py.
    """
    # --- Adaptive Parameters (Rest Frame Angstroms) ---
    SEARCH_WINDOW_REST = (5800, 6400)
    CONTINUUM_WIDTH = 50
    CONTINUUM_PADDING = 50

    # --- Step 1: Broad Search for the Feature Minimum ---
    observed_search_window = tuple(wl * (1 + redshift) for wl in SEARCH_WINDOW_REST)
    search_mask = (wavelength >= observed_search_window[0]) & (wavelength <= observed_search_window[1])
    if not np.any(search_mask):
        return 0.0

    min_flux_idx_global = np.where(wavelength == wavelength[search_mask][np.argmin(flux[search_mask])])[0][0]
    found_min_wavelength = wavelength[min_flux_idx_global]
    min_flux = flux[min_flux_idx_global]

    # --- Step 2: First-Pass Continuum Fit to Find Feature Edges ---
    temp_feature_width_obs = 150 * (1 + redshift)
    blue_start = found_min_wavelength - (temp_feature_width_obs / 2) - (CONTINUUM_PADDING * (1 + redshift)) - (CONTINUUM_WIDTH * (1 + redshift))
    blue_end = blue_start + (CONTINUUM_WIDTH * (1 + redshift))
    red_start = found_min_wavelength + (temp_feature_width_obs / 2) + (CONTINUUM_PADDING * (1 + redshift))
    red_end = red_start + (CONTINUUM_WIDTH * (1 + redshift))

    continuum_mask_pass1 = ((wavelength >= blue_start) & (wavelength <= blue_end)) | \
                           ((wavelength >= red_start) & (wavelength <= red_end))

    if np.sum(continuum_mask_pass1) < 3:
        return 0.0

    coeffs_pass1 = np.polyfit(wavelength[continuum_mask_pass1], flux[continuum_mask_pass1], 2)
    continuum_fit_pass1 = np.poly1d(coeffs_pass1)

    # --- Step 3: Find intersection to determine feature edges ---
    blue_edge_idx = min_flux_idx_global
    while blue_edge_idx > 0 and flux[blue_edge_idx] < continuum_fit_pass1(wavelength[blue_edge_idx]):
        blue_edge_idx -= 1
    red_edge_idx = min_flux_idx_global
    while red_edge_idx < len(wavelength) - 1 and flux[red_edge_idx] < continuum_fit_pass1(wavelength[red_edge_idx]):
        red_edge_idx += 1

    observed_feature_region = (wavelength[blue_edge_idx], wavelength[red_edge_idx])

    # --- Step 4: Final Continuum Regions and Second Fit ---
    continuum_padding_obs = CONTINUUM_PADDING * (1 + redshift)
    continuum_width_obs = CONTINUUM_WIDTH * (1 + redshift)

    blue_continuum_end = observed_feature_region[0] - continuum_padding_obs
    blue_continuum_start = blue_continuum_end - continuum_width_obs
    red_continuum_start = observed_feature_region[1] + continuum_padding_obs
    red_continuum_end = red_continuum_start + continuum_width_obs

    continuum_mask_pass2 = ((wavelength >= blue_continuum_start) & (wavelength <= blue_continuum_end)) | \
                           ((wavelength >= red_continuum_start) & (wavelength <= red_continuum_end))

    if np.sum(continuum_mask_pass2) < 3:
        return 0.0

    # --- Step 5: Final S/N Calculation ---
    coeffs_final = np.polyfit(wavelength[continuum_mask_pass2], flux[continuum_mask_pass2], 2)
    continuum_fit_final = np.poly1d(coeffs_final)

    continuum_level_at_min = continuum_fit_final(found_min_wavelength)
    feature_depth = continuum_level_at_min - min_flux
    
    noise = None
    if uncertainty is not None:
        continuum_uncertainties = uncertainty[continuum_mask_pass2]
        if np.any(continuum_uncertainties > 0):
            noise = np.mean(continuum_uncertainties)

    if noise is None:
        blue_mask = (wavelength >= blue_continuum_start) & (wavelength <= blue_continuum_end)
        red_mask = (wavelength >= red_continuum_start) & (wavelength <= red_continuum_end)
        std_devs = []
        if np.sum(blue_mask) > 1: std_devs.append(np.std(flux[blue_mask]))
        if np.sum(red_mask) > 1: std_devs.append(np.std(flux[red_mask]))
        if not std_devs: return 0.0
        noise = np.mean(std_devs)

    snr = feature_depth / noise if noise > 0 else 0.0
    return snr

if __name__ == "__main__":
    print("SNR Methods Comparison Utility Loaded.")
    print("This script provides functions for: DER_SNR, Classical SNR, Pixel-wise SNR, and Feature Depth SNR.")
