"""
MCMC Implementation for LoopVacuum Invisible Light Cosmology (LVILC)

This module implements the Markov Chain Monte Carlo sampling for fitting
the LVILC cosmological model to observational data.
"""

import numpy as np
import emcee
from lvilc_model import LVILCModel
from data_loader import CosmologyData


class LVILC_MCMC:
    """
    MCMC sampler for LVILC cosmological parameters.
    
    Parameters to fit:
    - H0: Hubble constant (km/s/Mpc)
    - M_bh: Black hole mass (solar masses)
    - t_fall: Fall time parameter (Gyr)
    """
    
    def __init__(self, data_loader=None):
        """
        Initialize MCMC sampler.
        
        Parameters:
        -----------
        data_loader : CosmologyData, optional
            Data loader instance. If None, creates a new one.
        """
        if data_loader is None:
            self.data = CosmologyData()
        else:
            self.data = data_loader
            
        # Load observational data
        self.sn_data = self.data.get_sn_data()
        self.bao_data = self.data.get_bao_data()
        self.cmb_data = self.data.get_cmb_data()
        
        # Parameter names
        self.param_names = ['H0', 'M_bh', 't_fall']
        
        # MCMC settings
        self.nwalkers = 32
        self.nsteps = 5000
        self.burn_in = 1000
        
    def log_prior(self, theta):
        """
        Calculate log prior probability for parameters.
        
        Parameters:
        -----------
        theta : array
            Parameter vector [H0, M_bh, t_fall]
            
        Returns:
        --------
        float
            Log prior probability
        """
        H0, M_bh, t_fall = theta
        
        # Priors based on physical constraints and LVILC theory
        # H0 should be negative and near -6.73 km/s/Mpc
        if -15.0 < H0 < 0.0:
            log_prior_H0 = 0.0  # Uniform prior
        else:
            return -np.inf
            
        # Black hole mass should be enormous (universe-scale)
        # log10(M_bh) between 20 and 26 solar masses
        if 1e20 < M_bh < 1e26:
            log_prior_Mbh = 0.0  # Uniform in log space
        else:
            return -np.inf
            
        # Fall time should be comparable to age of universe
        # Between 10 and 20 Gyr
        if 10.0 < t_fall < 20.0:
            log_prior_tfall = 0.0  # Uniform prior
        else:
            return -np.inf
            
        return log_prior_H0 + log_prior_Mbh + log_prior_tfall
    
    def log_likelihood(self, theta):
        """
        Calculate log likelihood for given parameters.
        
        Parameters:
        -----------
        theta : array
            Parameter vector [H0, M_bh, t_fall]
            
        Returns:
        --------
        float
            Log likelihood
        """
        H0, M_bh, t_fall = theta
        
        # Create model with these parameters
        model = LVILCModel(H0=H0, M_bh=M_bh, t_fall=t_fall)
        
        # Calculate chi-square for supernova data
        try:
            mu_theory = model.distance_modulus(self.sn_data['z'])
            chi2_sn = np.sum(((self.sn_data['mu'] - mu_theory) / self.sn_data['mu_err'])**2)
        except:
            return -np.inf
            
        # Calculate chi-square for BAO data
        # BAO measures D_M/r_d where D_M is comoving angular diameter distance
        try:
            chi2_bao = 0.0
            for i, z_bao in enumerate(self.bao_data['z']):
                # For simplicity, use comoving distance as proxy
                # In full analysis, would need sound horizon r_d
                d_c_theory = model.comoving_distance(z_bao)
                # Assume r_d ~ 147 Mpc (typical value)
                DM_over_rd_theory = d_c_theory / 147.0
                
                diff = (self.bao_data['DM_over_rd'][i] - DM_over_rd_theory)
                chi2_bao += (diff / self.bao_data['DM_over_rd_err'][i])**2
        except:
            chi2_bao = 1e10  # Large penalty if calculation fails
            
        # Calculate chi-square for CMB data
        # CMB shift parameter R = sqrt(Omega_m) * D_A(z_cmb) / c * H0
        # For simplicity, we'll use a soft constraint
        try:
            z_cmb = 1089.92  # Redshift of last scattering
            # This is approximate - full analysis would be more complex
            chi2_cmb = 0.0  # Simplified for this implementation
        except:
            chi2_cmb = 0.0
            
        # Total chi-square
        chi2_total = chi2_sn + chi2_bao + chi2_cmb
        
        # Log likelihood
        log_like = -0.5 * chi2_total
        
        return log_like
    
    def log_probability(self, theta):
        """
        Calculate log posterior probability.
        
        Parameters:
        -----------
        theta : array
            Parameter vector [H0, M_bh, t_fall]
            
        Returns:
        --------
        float
            Log posterior probability
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)
    
    def initialize_walkers(self, initial_params=None):
        """
        Initialize walker positions for MCMC.
        
        Parameters:
        -----------
        initial_params : array, optional
            Starting parameters [H0, M_bh, t_fall]
            If None, uses default values near expected values
            
        Returns:
        --------
        array
            Initial positions for all walkers
        """
        if initial_params is None:
            # Default starting point based on LVILC theory
            initial_params = np.array([-6.73, 1e23, 13.8])
        
        ndim = len(initial_params)
        
        # Small perturbations around initial values
        # Use relative perturbations that respect the parameter scales
        pos = initial_params + np.array([
            0.1 * np.random.randn(self.nwalkers),  # H0
            1e22 * np.random.randn(self.nwalkers),  # M_bh
            0.5 * np.random.randn(self.nwalkers)   # t_fall
        ]).T
        
        return pos
    
    def run_mcmc(self, initial_params=None, nwalkers=None, nsteps=None, 
                 burn_in=None, progress=True):
        """
        Run MCMC sampling.
        
        Parameters:
        -----------
        initial_params : array, optional
            Starting parameters
        nwalkers : int, optional
            Number of walkers
        nsteps : int, optional
            Number of steps
        burn_in : int, optional
            Number of burn-in steps to discard
        progress : bool
            Show progress bar
            
        Returns:
        --------
        sampler : emcee.EnsembleSampler
            The MCMC sampler with results
        """
        if nwalkers is not None:
            self.nwalkers = nwalkers
        if nsteps is not None:
            self.nsteps = nsteps
        if burn_in is not None:
            self.burn_in = burn_in
            
        # Initialize walkers
        pos = self.initialize_walkers(initial_params)
        ndim = len(self.param_names)
        
        # Create sampler
        sampler = emcee.EnsembleSampler(
            self.nwalkers, ndim, self.log_probability
        )
        
        # Run MCMC
        print(f"Running MCMC with {self.nwalkers} walkers for {self.nsteps} steps...")
        sampler.run_mcmc(pos, self.nsteps, progress=progress)
        
        print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
        
        # Check convergence
        try:
            tau = sampler.get_autocorr_time(quiet=True)
            print(f"Autocorrelation time: {tau}")
        except:
            print("Autocorrelation time could not be calculated (chain may be too short)")
        
        return sampler
    
    def get_results(self, sampler):
        """
        Extract results from MCMC sampler.
        
        Parameters:
        -----------
        sampler : emcee.EnsembleSampler
            Completed MCMC sampler
            
        Returns:
        --------
        dict
            Dictionary with parameter estimates and uncertainties
        """
        # Get samples after burn-in
        samples = sampler.get_chain(discard=self.burn_in, flat=True)
        
        # Calculate statistics for each parameter
        results = {}
        for i, param_name in enumerate(self.param_names):
            param_samples = samples[:, i]
            
            # Median and percentiles
            median = np.median(param_samples)
            lower = np.percentile(param_samples, 16)
            upper = np.percentile(param_samples, 84)
            
            results[param_name] = {
                'median': median,
                'lower_1sigma': median - lower,
                'upper_1sigma': upper - median,
                'mean': np.mean(param_samples),
                'std': np.std(param_samples)
            }
            
        # Store full samples for further analysis
        results['samples'] = samples
        results['param_names'] = self.param_names
        
        return results
