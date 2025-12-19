"""
Analysis and visualization tools for LVILC MCMC results.
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
from lvilc_model import LVILCModel


class MCMCAnalyzer:
    """
    Analyzer for MCMC results.
    """
    
    def __init__(self, results, data):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        results : dict
            Results from LVILC_MCMC.get_results()
        data : CosmologyData
            Observational data
        """
        self.results = results
        self.data = data
        self.samples = results['samples']
        self.param_names = results['param_names']
        
    def plot_corner(self, filename='corner_plot.png', truths=None):
        """
        Create corner plot showing parameter correlations.
        
        Parameters:
        -----------
        filename : str
            Output filename
        truths : array, optional
            True parameter values to mark on plot
            
        Returns:
        --------
        fig : matplotlib.Figure
            The corner plot figure
        """
        # Create labels with LaTeX formatting
        labels = [r'$H_0$ [km/s/Mpc]', 
                  r'$M_{BH}$ [$M_\odot$]', 
                  r'$t_{fall}$ [Gyr]']
        
        # Create corner plot
        fig = corner.corner(
            self.samples,
            labels=labels,
            truths=truths,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 14}
        )
        
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Corner plot saved to {filename}")
        
        return fig
    
    def plot_chains(self, filename='chains.png'):
        """
        Plot MCMC chains to visualize convergence.
        
        Parameters:
        -----------
        filename : str
            Output filename
            
        Returns:
        --------
        fig : matplotlib.Figure
            The chains figure
        """
        # Reshape samples back to (nwalkers, nsteps, ndim)
        # Note: samples are already flattened, so we'll just plot histograms
        fig, axes = plt.subplots(len(self.param_names), 1, 
                                 figsize=(10, 8), sharex=False)
        
        labels = [r'$H_0$ [km/s/Mpc]', 
                  r'$M_{BH}$ [$M_\odot$]', 
                  r'$t_{fall}$ [Gyr]']
        
        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.hist(self.samples[:, i], bins=50, alpha=0.7, 
                   color='steelblue', edgecolor='black')
            ax.set_ylabel('Frequency')
            ax.set_xlabel(label)
            ax.axvline(self.results[self.param_names[i]]['median'], 
                      color='red', linestyle='--', linewidth=2, 
                      label='Median')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Chain histograms saved to {filename}")
        
        return fig
    
    def plot_model_comparison(self, filename='model_comparison.png'):
        """
        Plot model predictions vs observational data.
        
        Parameters:
        -----------
        filename : str
            Output filename
            
        Returns:
        --------
        fig : matplotlib.Figure
            The comparison figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # Get best-fit parameters (median)
        H0_best = self.results['H0']['median']
        M_bh_best = self.results['M_bh']['median']
        t_fall_best = self.results['t_fall']['median']
        
        model = LVILCModel(H0=H0_best, M_bh=M_bh_best, t_fall=t_fall_best)
        
        # Plot 1: Supernova Hubble diagram
        ax = axes[0]
        sn_data = self.data.get_sn_data()
        
        # Observational data
        ax.errorbar(sn_data['z'], sn_data['mu'], yerr=sn_data['mu_err'],
                   fmt='o', alpha=0.6, label='Observations', 
                   color='black', markersize=4)
        
        # Model prediction
        z_model = np.linspace(0.01, max(sn_data['z']), 100)
        mu_model = model.distance_modulus(z_model)
        ax.plot(z_model, mu_model, 'r-', linewidth=2, 
               label='LVILC Best Fit')
        
        # Sample from posterior to show uncertainty band
        n_samples = min(100, len(self.samples))
        indices = np.random.choice(len(self.samples), n_samples, replace=False)
        
        for idx in indices[:20]:  # Plot 20 random samples
            params = self.samples[idx]
            model_sample = LVILCModel(H0=params[0], M_bh=params[1], 
                                     t_fall=params[2])
            mu_sample = model_sample.distance_modulus(z_model)
            ax.plot(z_model, mu_sample, 'r-', alpha=0.05, linewidth=1)
        
        ax.set_xlabel('Redshift $z$', fontsize=12)
        ax.set_ylabel('Distance Modulus $\\mu$', fontsize=12)
        ax.set_title('Supernova Hubble Diagram', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        # Plot 2: Hubble parameter evolution
        ax = axes[1]
        z_range = np.linspace(0, 2.0, 100)
        H_z = model.hubble_parameter(z_range)
        
        ax.plot(z_range, H_z, 'b-', linewidth=2, label='LVILC Best Fit')
        
        # Sample from posterior
        for idx in indices[:20]:
            params = self.samples[idx]
            model_sample = LVILCModel(H0=params[0], M_bh=params[1], 
                                     t_fall=params[2])
            H_z_sample = model_sample.hubble_parameter(z_range)
            ax.plot(z_range, H_z_sample, 'b-', alpha=0.05, linewidth=1)
        
        ax.set_xlabel('Redshift $z$', fontsize=12)
        ax.set_ylabel('$H(z)$ [km/s/Mpc]', fontsize=12)
        ax.set_title('Hubble Parameter Evolution', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {filename}")
        
        return fig
    
    def print_results(self):
        """
        Print formatted results summary.
        """
        print("\n" + "="*60)
        print("LVILC MCMC Results Summary")
        print("="*60)
        
        for param_name in self.param_names:
            res = self.results[param_name]
            print(f"\n{param_name}:")
            print(f"  Median: {res['median']:.4e}")
            print(f"  Mean:   {res['mean']:.4e}")
            print(f"  Std:    {res['std']:.4e}")
            print(f"  68% CI: [{res['median'] - res['lower_1sigma']:.4e}, "
                  f"{res['median'] + res['upper_1sigma']:.4e}]")
        
        print("\n" + "="*60)
        
    def calculate_goodness_of_fit(self):
        """
        Calculate chi-square and reduced chi-square for best-fit model.
        
        Returns:
        --------
        dict
            Dictionary with goodness-of-fit statistics
        """
        # Get best-fit parameters
        H0_best = self.results['H0']['median']
        M_bh_best = self.results['M_bh']['median']
        t_fall_best = self.results['t_fall']['median']
        
        model = LVILCModel(H0=H0_best, M_bh=M_bh_best, t_fall=t_fall_best)
        
        # Calculate chi-square for supernova data
        sn_data = self.data.get_sn_data()
        mu_theory = model.distance_modulus(sn_data['z'])
        chi2_sn = np.sum(((sn_data['mu'] - mu_theory) / sn_data['mu_err'])**2)
        
        n_data = len(sn_data['z'])
        n_params = len(self.param_names)
        dof = n_data - n_params
        
        chi2_reduced = chi2_sn / dof
        
        results = {
            'chi2': chi2_sn,
            'dof': dof,
            'chi2_reduced': chi2_reduced,
            'n_data': n_data,
            'n_params': n_params
        }
        
        print(f"\nGoodness of Fit:")
        print(f"  χ² = {chi2_sn:.2f}")
        print(f"  DOF = {dof}")
        print(f"  χ²/DOF = {chi2_reduced:.2f}")
        
        return results
