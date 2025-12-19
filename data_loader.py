"""
Data loader for cosmological observations.

Handles loading and processing of:
- Supernova data (Pantheon+, SN 2025wny)
- Baryon Acoustic Oscillation (BAO) data from DESI
- Cosmic Microwave Background (CMB) data
"""

import numpy as np
import pandas as pd


class CosmologyData:
    """
    Container for cosmological observational data.
    """
    
    def __init__(self):
        self.sn_data = None
        self.bao_data = None
        self.cmb_data = None
        
    def load_supernova_data(self, filename=None):
        """
        Load supernova data including distance modulus and redshift.
        
        If no file is provided, uses sample data based on Pantheon+ 
        and SN 2025wny observations.
        
        Parameters:
        -----------
        filename : str, optional
            Path to supernova data file
            
        Returns:
        --------
        dict
            Dictionary with 'z' (redshift), 'mu' (distance modulus), 
            'mu_err' (error in distance modulus)
        """
        if filename:
            # Load from file
            data = pd.read_csv(filename)
            self.sn_data = {
                'z': data['z'].values,
                'mu': data['mu'].values,
                'mu_err': data['mu_err'].values
            }
        else:
            # Use sample data based on typical supernova observations
            # This includes characteristics similar to Pantheon+ and SN 2025wny
            np.random.seed(42)
            n_sn = 50
            
            # Sample redshifts from 0.01 to 2.0
            z_sample = np.logspace(-2, 0.3, n_sn)
            
            # Generate distance moduli based on a fiducial model
            # Using approximate ΛCDM values for realistic data
            mu_sample = 5 * np.log10(3000 * z_sample * (1 + z_sample/2)) + 25
            
            # Add realistic uncertainties
            mu_err_sample = 0.15 + 0.05 * np.random.rand(n_sn)
            
            # Add scatter
            mu_sample += np.random.normal(0, mu_err_sample)
            
            self.sn_data = {
                'z': z_sample,
                'mu': mu_sample,
                'mu_err': mu_err_sample
            }
            
        return self.sn_data
    
    def load_bao_data(self, filename=None):
        """
        Load Baryon Acoustic Oscillation data from DESI 2025.
        
        Parameters:
        -----------
        filename : str, optional
            Path to BAO data file
            
        Returns:
        --------
        dict
            Dictionary with BAO measurements
        """
        if filename:
            data = pd.read_csv(filename)
            self.bao_data = {
                'z': data['z'].values,
                'DM_over_rd': data['DM_over_rd'].values,
                'DM_over_rd_err': data['DM_over_rd_err'].values
            }
        else:
            # Sample DESI 2025 BAO data points
            # Based on typical DESI measurements
            self.bao_data = {
                'z': np.array([0.51, 0.70, 0.93, 1.32, 1.49]),
                'DM_over_rd': np.array([13.5, 17.9, 22.3, 30.2, 33.5]),
                'DM_over_rd_err': np.array([0.15, 0.18, 0.25, 0.35, 0.40])
            }
            
        return self.bao_data
    
    def load_cmb_data(self, filename=None):
        """
        Load Cosmic Microwave Background data.
        
        Parameters:
        -----------
        filename : str, optional
            Path to CMB data file
            
        Returns:
        --------
        dict
            Dictionary with CMB parameters
        """
        if filename:
            data = pd.read_csv(filename)
            self.cmb_data = data.to_dict('records')[0]
        else:
            # Planck 2018 compressed CMB data
            self.cmb_data = {
                'R': 1.7488,  # CMB shift parameter
                'R_err': 0.0074,
                'la': 301.76,  # Acoustic scale
                'la_err': 0.14,
                'omega_b_h2': 0.02237,  # Baryon density
                'omega_b_h2_err': 0.00015
            }
            
        return self.cmb_data
    
    def get_sn_data(self):
        """Get supernova data, loading if necessary."""
        if self.sn_data is None:
            self.load_supernova_data()
        return self.sn_data
    
    def get_bao_data(self):
        """Get BAO data, loading if necessary."""
        if self.bao_data is None:
            self.load_bao_data()
        return self.bao_data
    
    def get_cmb_data(self):
        """Get CMB data, loading if necessary."""
        if self.cmb_data is None:
            self.load_cmb_data()
        return self.cmb_data
