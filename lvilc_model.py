"""
LoopVacuum Invisible Light Cosmology (LVILC) Model

This module implements the cosmological model where the universe is inside
a giant black hole, with expansion as an illusion from horizon dilation.
"""

import numpy as np
from scipy.integrate import quad
from astropy import constants as const
from astropy import units as u


class LVILCModel:
    """
    LVILC cosmological model with horizon dilation effects.
    
    Parameters:
    -----------
    H0 : float
        Hubble constant in km/s/Mpc (typically negative for LVILC)
    M_bh : float
        Black hole mass in solar masses
    t_fall : float
        Fall time parameter in Gyr
    """
    
    def __init__(self, H0=-6.73, M_bh=1e23, t_fall=13.8):
        self.H0 = H0  # km/s/Mpc
        self.M_bh = M_bh  # solar masses
        self.t_fall = t_fall  # Gyr
        
        # Physical constants
        self.c = const.c.to(u.km/u.s).value  # speed of light in km/s
        self.H0_si = abs(H0) * 1e3 / (3.086e22)  # Convert to SI units (1/s)
        
    def horizon_dilation_factor(self, z):
        """
        Calculate the horizon dilation factor as a function of redshift.
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        float or array
            Horizon dilation factor
        """
        # Model horizon dilation as exponential with redshift
        # This creates the illusion of expansion
        return np.exp(self.H0_si * self.t_fall * 1e9 * 365.25 * 86400 * z / (1 + z))
    
    def comoving_distance(self, z):
        """
        Calculate comoving distance including horizon dilation effects.
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        float or array
            Comoving distance in Mpc
        """
        if np.isscalar(z):
            if z <= 0:
                return 0.0
            result, _ = quad(lambda zp: self.c / (abs(self.H0) * self.horizon_dilation_factor(zp)), 
                           0, z)
            return result
        else:
            return np.array([self.comoving_distance(zi) for zi in z])
    
    def luminosity_distance(self, z):
        """
        Calculate luminosity distance for LVILC model.
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        float or array
            Luminosity distance in Mpc
        """
        d_c = self.comoving_distance(z)
        # Standard cosmological relation
        d_L = (1 + z) * d_c
        return d_L
    
    def distance_modulus(self, z):
        """
        Calculate distance modulus for LVILC model.
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        float or array
            Distance modulus in magnitudes
        """
        d_L = self.luminosity_distance(z)
        # Distance modulus: μ = 5 log10(d_L/10pc) = 5 log10(d_L) + 25
        # where d_L is in Mpc
        if np.isscalar(d_L):
            if d_L <= 0:
                return -np.inf
            return 5 * np.log10(d_L) + 25
        else:
            result = np.zeros_like(d_L)
            mask = d_L > 0
            result[mask] = 5 * np.log10(d_L[mask]) + 25
            result[~mask] = -np.inf
            return result
    
    def angular_diameter_distance(self, z):
        """
        Calculate angular diameter distance.
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        float or array
            Angular diameter distance in Mpc
        """
        d_c = self.comoving_distance(z)
        d_A = d_c / (1 + z)
        return d_A
    
    def hubble_parameter(self, z):
        """
        Calculate Hubble parameter H(z) for LVILC model.
        
        Parameters:
        -----------
        z : float or array
            Redshift
            
        Returns:
        --------
        float or array
            H(z) in km/s/Mpc
        """
        # In LVILC, H(z) changes due to horizon dilation
        dilation = self.horizon_dilation_factor(z)
        return abs(self.H0) * dilation
