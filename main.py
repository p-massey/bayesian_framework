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
    Now samples x0 explicitly via log10_x0.
    """
    model = sncosmo.Model(source=model_name)
    
    if redshift is not None:
        model.set(z=redshift)
        params = ['t0', 'x1', 'c', 'log10_x0']
        priors = {
            't0': (-100, 100), 
            'x1': (-3, 3),
            'c': (-0.8, 0.8),
            'log10_x0': (-12, -2) # Sample over wide flux range
        }
    else:
        params = ['z', 't0', 'x1', 'c', 'log10_x0']
        priors = {
            'z': (0.005, 0.1),
            't0': (-100, 100),
            'x1': (-3, 3),
            'c': (-0.8, 0.8),
            'log10_x0': (-12, -2)
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
        # Handle log10 conversion for x0
        if 'log10_x0' in param_dict:
            param_dict['x0'] = 10**param_dict.pop('log10_x0')
            
        model.set(**param_dict)
        
        try:
            model_flux = model.flux(obs_time, wavelength)
        except:
            return -1e10
            
        chisq = np.sum(((flux - model_flux) / flux_err)**2)
        
        return -0.5 * chisq if not np.isnan(chisq) else -1e10

    ndim = len(params)
    sampler = dynesty.NestedSampler(loglike, prior_transform, ndim, nlive=200)
    
    print(f"Starting dynesty fit with {ndim} parameters (including x0)...")
    sampler.run_nested()
    return sampler.results, params

def main():
    parser = argparse.ArgumentParser(description='Fit SN spectra age using sncosmo and dynesty')
    parser.add_argument('file', type=str, help='Path to .flm spectrum file')
    parser.add_argument('--z', type=float, default=None, help='Redshift of the SN')
    parser.add_argument('--model', type=str, default='salt3', help='sncosmo model name (e.g., salt3, salt2)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return

    base_name = os.path.splitext(os.path.basename(args.file))[0]
    wave, flux, err = load_flm_spectrum(args.file)
    
    # Pre-processing: Trim to model range
    mask = (wave > 3500) & (wave < 8000)
    wave, flux, err = wave[mask], flux[mask], err[mask]
    
    # Run fit
    results, params = fit_spectrum_with_dynesty(wave, flux, err, model_name=args.model, redshift=args.z)
    
    # Process results
    weights = np.exp(results.logwt - results.logz[-1])
    samples = dynesty.utils.resample_equal(results.samples, weights)
    
    t0_samples = samples[:, params.index('t0')]
    age_samples = -t0_samples
    
    print(f"\n--- Results for {os.path.basename(args.file)} ---")
    print(f"Mean Age: {np.mean(age_samples):.2f} +/- {np.std(age_samples):.2f} days")
    
    # Saving plots with unique names
    corner_file = f"{base_name}_corner.png"
    fit_file = f"{base_name}_fit.png"
    run_file = f"{base_name}_run.png"
    trace_file = f"{base_name}_trace.png"
    
    # Corner plot
    fig, axes = dyplot.cornerplot(results, labels=params)
    plt.savefig(corner_file)
    plt.close(fig)
    
    # Run plot (diagnostics)
    fig, axes = dyplot.runplot(results)
    plt.savefig(run_file)
    plt.close(fig)

    # Trace plot
    fig, axes = dyplot.traceplot(results, labels=params, 
                                 show_titles=True,
                                 trace_cmap='viridis', connect=True)
    plt.savefig(trace_file)
    plt.close(fig)
    
    best_idx = np.argmax(results.logl)
    best_params_raw = results.samples[best_idx]
    p_dict = dict(zip(params, best_params_raw))
    
    # Prep model for plotting
    model = sncosmo.Model(source=args.model)
    if args.z: model.set(z=args.z)
    
    plot_p_dict = p_dict.copy()
    if 'log10_x0' in plot_p_dict:
        plot_p_dict['x0'] = 10**plot_p_dict.pop('log10_x0')
    
    model.set(**plot_p_dict)
    m_flux = model.flux(0.0, wave)
    
    plt.figure(figsize=(10, 6))
    plt.step(wave, flux, where='mid', label='Data', color='black', alpha=0.5)
    plt.plot(wave, m_flux, label=f'Best fit {args.model} (Age={-p_dict["t0"]:.1f})', color='red')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux')
    plt.title(f"Fit for {base_name} (x0={10**p_dict['log10_x0']:.2e})")
    plt.legend()
    plt.savefig(fit_file)
    plt.close()
    print(f"Plots saved: {corner_file}, {fit_file}")

if __name__ == "__main__":
    main()
