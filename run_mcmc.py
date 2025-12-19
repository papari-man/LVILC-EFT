"""
Main execution script for LVILC MCMC analysis.

This script runs the complete MCMC analysis for the LoopVacuum Invisible 
Light Cosmology model, including:
1. Loading observational data
2. Running MCMC sampling
3. Analyzing results
4. Generating visualizations
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import argparse
import sys

from lvilc_model import LVILCModel
from data_loader import CosmologyData
from mcmc_lvilc import LVILC_MCMC
from analysis import MCMCAnalyzer


def main():
    """
    Main function to run LVILC MCMC analysis.
    """
    parser = argparse.ArgumentParser(
        description='Run MCMC analysis for LVILC cosmology model'
    )
    parser.add_argument('--nwalkers', type=int, default=32,
                       help='Number of MCMC walkers (default: 32)')
    parser.add_argument('--nsteps', type=int, default=5000,
                       help='Number of MCMC steps (default: 5000)')
    parser.add_argument('--burn-in', type=int, default=1000,
                       help='Number of burn-in steps to discard (default: 1000)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for plots (default: current directory)')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    parser.add_argument('--initial-params', type=float, nargs=3,
                       metavar=('H0', 'M_bh', 't_fall'),
                       help='Initial parameters: H0 M_bh t_fall')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LVILC MCMC Analysis")
    print("LoopVacuum Invisible Light Cosmology")
    print("="*70)
    print()
    
    # Step 1: Load observational data
    print("Step 1: Loading observational data...")
    data_loader = CosmologyData()
    sn_data = data_loader.load_supernova_data()
    bao_data = data_loader.load_bao_data()
    cmb_data = data_loader.load_cmb_data()
    
    print(f"  - Loaded {len(sn_data['z'])} supernova observations")
    print(f"  - Loaded {len(bao_data['z'])} BAO measurements")
    print(f"  - Loaded CMB data")
    print()
    
    # Step 2: Initialize MCMC
    print("Step 2: Initializing MCMC sampler...")
    mcmc = LVILC_MCMC(data_loader=data_loader)
    
    # Set initial parameters if provided
    initial_params = None
    if args.initial_params:
        initial_params = np.array(args.initial_params)
        print(f"  - Using initial parameters: H0={initial_params[0]:.2f}, "
              f"M_bh={initial_params[1]:.2e}, t_fall={initial_params[2]:.2f}")
    else:
        print("  - Using default initial parameters from LVILC theory")
        print("    H0 = -6.73 km/s/Mpc, M_bh = 1e23 M_sun, t_fall = 13.8 Gyr")
    print()
    
    # Step 3: Run MCMC
    print("Step 3: Running MCMC sampling...")
    print(f"  - Walkers: {args.nwalkers}")
    print(f"  - Steps: {args.nsteps}")
    print(f"  - Burn-in: {args.burn_in}")
    print()
    
    sampler = mcmc.run_mcmc(
        initial_params=initial_params,
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        burn_in=args.burn_in,
        progress=not args.no_progress
    )
    print()
    
    # Step 4: Extract results
    print("Step 4: Extracting results...")
    results = mcmc.get_results(sampler)
    print()
    
    # Step 5: Analyze results
    print("Step 5: Analyzing results...")
    analyzer = MCMCAnalyzer(results, data_loader)
    
    # Print results summary
    analyzer.print_results()
    
    # Calculate goodness of fit
    gof = analyzer.calculate_goodness_of_fit()
    print()
    
    # Step 6: Generate visualizations
    print("Step 6: Generating visualizations...")
    
    import os
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Corner plot
    corner_file = os.path.join(output_dir, 'lvilc_corner_plot.png')
    analyzer.plot_corner(filename=corner_file)
    
    # Chain histograms
    chains_file = os.path.join(output_dir, 'lvilc_chains.png')
    analyzer.plot_chains(filename=chains_file)
    
    # Model comparison
    comparison_file = os.path.join(output_dir, 'lvilc_model_comparison.png')
    analyzer.plot_model_comparison(filename=comparison_file)
    
    print()
    print("="*70)
    print("Analysis complete!")
    print("="*70)
    print("\nKey Results:")
    print(f"  H0 = {results['H0']['median']:.2f} ± {results['H0']['std']:.2f} km/s/Mpc")
    print(f"  M_bh = {results['M_bh']['median']:.2e} ± {results['M_bh']['std']:.2e} M_sun")
    print(f"  t_fall = {results['t_fall']['median']:.2f} ± {results['t_fall']['std']:.2f} Gyr")
    print(f"\n  χ²/DOF = {gof['chi2_reduced']:.2f}")
    print(f"\nPlots saved to: {output_dir}/")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
