"""
Tests for LVILC MCMC implementation.

This module provides basic tests to validate the MCMC implementation
for the LoopVacuum Invisible Light Cosmology model.
"""

import numpy as np
import sys


def test_lvilc_model():
    """Test basic functionality of LVILCModel."""
    print("Testing LVILCModel...")
    
    from lvilc_model import LVILCModel
    
    # Create model with default parameters
    model = LVILCModel(H0=-6.73, M_bh=1e23, t_fall=13.8)
    
    # Test luminosity distance calculation
    z_test = np.array([0.1, 0.5, 1.0])
    d_L = model.luminosity_distance(z_test)
    
    assert len(d_L) == len(z_test), "Output length mismatch"
    assert np.all(d_L > 0), "Luminosity distances should be positive"
    assert np.all(d_L[1:] > d_L[:-1]), "Distances should increase with redshift"
    
    # Test distance modulus
    mu = model.distance_modulus(z_test)
    assert len(mu) == len(z_test), "Output length mismatch"
    assert np.all(np.isfinite(mu)), "Distance moduli should be finite"
    
    # Test Hubble parameter
    H_z = model.hubble_parameter(z_test)
    assert len(H_z) == len(z_test), "Output length mismatch"
    assert np.all(H_z > 0), "H(z) should be positive"
    
    print("  ✓ LVILCModel tests passed")
    return True


def test_data_loader():
    """Test data loading functionality."""
    print("Testing CosmologyData...")
    
    from data_loader import CosmologyData
    
    data = CosmologyData()
    
    # Test supernova data loading
    sn_data = data.load_supernova_data()
    assert 'z' in sn_data, "Missing redshift data"
    assert 'mu' in sn_data, "Missing distance modulus data"
    assert 'mu_err' in sn_data, "Missing error data"
    assert len(sn_data['z']) > 0, "No data loaded"
    
    # Test BAO data loading
    bao_data = data.load_bao_data()
    assert 'z' in bao_data, "Missing BAO redshift data"
    assert len(bao_data['z']) > 0, "No BAO data loaded"
    
    # Test CMB data loading
    cmb_data = data.load_cmb_data()
    assert 'R' in cmb_data, "Missing CMB shift parameter"
    
    print("  ✓ CosmologyData tests passed")
    return True


def test_mcmc_initialization():
    """Test MCMC initialization."""
    print("Testing LVILC_MCMC initialization...")
    
    from mcmc_lvilc import LVILC_MCMC
    from data_loader import CosmologyData
    
    data = CosmologyData()
    mcmc = LVILC_MCMC(data_loader=data)
    
    # Test walker initialization
    pos = mcmc.initialize_walkers()
    assert pos.shape == (mcmc.nwalkers, len(mcmc.param_names)), \
        "Walker positions have wrong shape"
    
    # Test prior function
    theta_valid = np.array([-6.73, 1e23, 13.8])
    log_prior = mcmc.log_prior(theta_valid)
    assert np.isfinite(log_prior), "Prior should be finite for valid parameters"
    
    theta_invalid = np.array([10.0, 1e23, 13.8])  # H0 should be negative
    log_prior_invalid = mcmc.log_prior(theta_invalid)
    assert log_prior_invalid == -np.inf, "Prior should reject invalid parameters"
    
    # Test likelihood function
    log_like = mcmc.log_likelihood(theta_valid)
    assert np.isfinite(log_like), "Likelihood should be finite for valid parameters"
    
    print("  ✓ LVILC_MCMC initialization tests passed")
    return True


def test_short_mcmc_run():
    """Test a short MCMC run."""
    print("Testing short MCMC run...")
    
    from mcmc_lvilc import LVILC_MCMC
    from data_loader import CosmologyData
    
    data = CosmologyData()
    mcmc = LVILC_MCMC(data_loader=data)
    
    # Run very short MCMC for testing
    sampler = mcmc.run_mcmc(nwalkers=8, nsteps=50, burn_in=10, progress=False)
    
    # Check that sampler ran
    assert sampler.get_chain().shape[0] == 50, "Wrong number of steps"
    assert sampler.get_chain().shape[1] == 8, "Wrong number of walkers"
    
    # Check acceptance fraction
    acceptance = np.mean(sampler.acceptance_fraction)
    assert 0 < acceptance < 1, f"Acceptance fraction should be in (0,1), got {acceptance}"
    
    # Get results
    results = mcmc.get_results(sampler)
    assert 'H0' in results, "Missing H0 results"
    assert 'M_bh' in results, "Missing M_bh results"
    assert 't_fall' in results, "Missing t_fall results"
    
    print("  ✓ Short MCMC run tests passed")
    return True


def test_analysis():
    """Test analysis functionality."""
    print("Testing MCMCAnalyzer...")
    
    from mcmc_lvilc import LVILC_MCMC
    from data_loader import CosmologyData
    from analysis import MCMCAnalyzer
    
    data = CosmologyData()
    mcmc = LVILC_MCMC(data_loader=data)
    
    # Run short MCMC
    sampler = mcmc.run_mcmc(nwalkers=8, nsteps=50, burn_in=10, progress=False)
    results = mcmc.get_results(sampler)
    
    # Create analyzer
    analyzer = MCMCAnalyzer(results, data)
    
    # Test goodness of fit calculation
    gof = analyzer.calculate_goodness_of_fit()
    assert 'chi2' in gof, "Missing chi-square"
    assert 'chi2_reduced' in gof, "Missing reduced chi-square"
    assert gof['chi2'] > 0, "Chi-square should be positive"
    
    print("  ✓ MCMCAnalyzer tests passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running LVILC MCMC Tests")
    print("="*60 + "\n")
    
    tests = [
        test_lvilc_model,
        test_data_loader,
        test_mcmc_initialization,
        test_short_mcmc_run,
        test_analysis
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
