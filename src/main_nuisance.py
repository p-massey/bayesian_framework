
import os
import argparse
import numpy as np
import sncosmo
import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt

def load_flm_spectrum(file_path):
    """
    Loads an .flm file which is expected to have 3 columns:
    wavelength (Angstrom), flux, flux_error.
    Handles rows with missing columns by skipping or filling.
    """
    data = np.genfromtxt(file_path, invalid_raise=False)
    
    if data.shape[1] < 3:
        wavelength = data[:, 0]
        flux = data[:, 1]
        flux_err = 0.1 * np.abs(flux) 
    else:
        wavelength = data[:, 0]
        flux = data[:, 1]
        flux_err = data[:, 2]
        
        mask = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
        wavelength = wavelength[mask]
        flux = flux[mask]
        flux_err = flux_err[mask]

    return wavelength, flux, flux_err

def fit_spectrum_with_dynesty(wavelength, flux, flux_err, model_name='salt3', redshift=None):
    """
    Fits a single spectrum using dynesty and sncosmo.
    Treats x0 as a nuisance parameter by analytically optimizing it.
    """
    model = sncosmo.Model(source=model_name)
    
    if redshift is not None:
        model.set(z=redshift)
        params = ['t0', 'x1', 'c']
        priors = {
            't0': (-100, 100), 
            'x1': (-3, 3),
            'c': (-0.8, 0.8)
        }
    else:
        params = ['z', 't0', 'x1', 'c']
        priors = {
            'z': (0.005, 0.1),
            't0': (-100, 100),
            'x1': (-3, 3),
            'c': (-0.8, 0.8)
        }

    obs_time = 0.0

    def prior_transform(utheta):
        theta = np.zeros_like(utheta)
        for i, p in enumerate(params):
            low, high = priors[p]
            theta[i] = utheta[i] * (high - low) + low
        return theta

    def loglike(theta):
        param_dict = dict(zip(params, theta))
        model.set(**param_dict)
        
        try:
            # Set unit x0 to get the model shape
            model.set(x0=1.0)
            model_flux_unit = model.flux(obs_time, wavelength)
        except:
            return -1e10
            
        # Analytically marginalize over x0 (Nuisance parameter)
        # x0_best = sum(data * model / err^2) / sum(model^2 / err^2)
        w = 1.0 / flux_err**2
        num = np.sum(flux * model_flux_unit * w)
        den = np.sum(model_flux_unit**2 * w)
        
        if den <= 0:
            return -1e10
            
        x0_best = num / den
        if x0_best <= 0:
            return -1e10
            
        model_flux = x0_best * model_flux_unit
        chisq = np.sum(((flux - model_flux) / flux_err)**2)
        
        # Note: Proper marginalization would include -0.5 * ln(den) 
        # but for parameter estimation of other variables, chisq is the primary driver.
        return -0.5 * chisq if not np.isnan(chisq) else -1e10

    ndim = len(params)
    sampler = dynesty.NestedSampler(loglike, prior_transform, ndim, nlive=500, sample='rslice', bootstrap=0)
    
    print(f"Starting dynesty fit with {ndim} parameters (x0 as nuisance)...")
    sampler.run_nested()
    return sampler.results, params

def main():
    parser = argparse.ArgumentParser(description='Fit SN spectra age (x0 as nuisance)')
    parser.add_argument('file', type=str, help='Path to .flm spectrum file')
    parser.add_argument('--z', type=float, default=None, help='Redshift of the SN')
    parser.add_argument('--model', type=str, default='salt3', help='sncosmo model name')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return

    base_name = os.path.splitext(os.path.basename(args.file))[0] + "_nuisance"
    wave, flux, err = load_flm_spectrum(args.file)
    
    mask = (wave > 3500) & (wave < 8000)
    wave, flux, err = wave[mask], flux[mask], err[mask]
    
    results, params = fit_spectrum_with_dynesty(wave, flux, err, model_name=args.model, redshift=args.z)
    
    weights = np.exp(results.logwt - results.logz[-1])
    samples = dynesty.utils.resample_equal(results.samples, weights)
    
    t0_samples = samples[:, params.index('t0')]
    age_samples = -t0_samples
    
    print(f"\n--- Results for {os.path.basename(args.file)} ---")
    print(f"Mean Age: {np.mean(age_samples):.2f} +/- {np.std(age_samples):.2f} days")
    
    # Save plots
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)
    fig, axes = dyplot.cornerplot(results, labels=params)
    plt.savefig(os.path.join("outputs", "plots", f"{base_name}_corner.png"))
    plt.close(fig)
    
    fig, axes = dyplot.runplot(results)
    plt.savefig(os.path.join("outputs", "plots", f"{base_name}_run.png"))
    plt.close(fig)

    fig, axes = dyplot.traceplot(results, labels=params, show_titles=True)
    plt.savefig(os.path.join("outputs", "plots", f"{base_name}_trace.png"))
    plt.close(fig)
    
    # Re-calculate best fit for plotting
    best_idx = np.argmax(results.logl)
    best_params = results.samples[best_idx]
    p_dict = dict(zip(params, best_params))
    
    model = sncosmo.Model(source=args.model)
    if args.z: model.set(z=args.z)
    model.set(**p_dict)
    model.set(x0=1.0)
    m_flux_unit = model.flux(0.0, wave)
    x0_best = np.sum(flux * m_flux_unit / err**2) / np.sum(m_flux_unit**2 / err**2)
    
    plt.figure(figsize=(10, 6))
    plt.step(wave, flux, where='mid', label='Data', color='black', alpha=0.5)
    plt.plot(wave, x0_best * m_flux_unit, label=f'Best fit (Age={-p_dict["t0"]:.1f})', color='red')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux')
    plt.title(f"Fit for {base_name} (Best x0={x0_best:.2e})")
    plt.legend()
    plt.savefig(f"{base_name}_fit.png")
    plt.close()
    print(f"Plots saved with prefix: {base_name}")

if __name__ == "__main__":
    main()
