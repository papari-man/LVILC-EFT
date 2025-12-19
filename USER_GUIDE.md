# LVILC MCMC User Guide

## LoopVacuum Invisible Light Cosmology - MCMC Implementation

This package provides a complete MCMC (Markov Chain Monte Carlo) implementation for fitting the LoopVacuum Invisible Light Cosmology (LVILC) model to observational data.

### Theory Overview

The LVILC model proposes that:
- The universe exists inside a single giant black hole
- We are eternally falling toward its center
- The observed cosmic expansion is an illusion from horizon dilation
- This accelerating time dilation explains dark energy
- The model fits DESI 2025 and JWST data with H₀ ≈ -6.73 ± 0.96 km/s/Mpc

### Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

Run the MCMC analysis with default settings:
```bash
python run_mcmc.py
```

This will:
1. Load sample observational data (supernova, BAO, CMB)
2. Run MCMC with 32 walkers for 5000 steps
3. Generate corner plots, chain histograms, and model comparison plots
4. Print parameter constraints and goodness-of-fit statistics

### Usage Options

```bash
python run_mcmc.py [options]

Options:
  --nwalkers N          Number of MCMC walkers (default: 32)
  --nsteps N            Number of MCMC steps (default: 5000)
  --burn-in N           Number of burn-in steps (default: 1000)
  --output-dir DIR      Output directory for plots (default: current dir)
  --no-progress         Disable progress bar
  --initial-params H0 M_bh t_fall
                        Initial parameter values
```

### Examples

1. Quick test run (fewer steps):
```bash
python run_mcmc.py --nwalkers 16 --nsteps 1000 --burn-in 200
```

2. Production run with custom initial parameters:
```bash
python run_mcmc.py --nwalkers 64 --nsteps 10000 --burn-in 2000 \
    --initial-params -6.5 1e23 14.0 --output-dir ./results
```

3. Run without progress bar (for batch jobs):
```bash
python run_mcmc.py --no-progress --output-dir ./batch_results
```

### Module Structure

- **lvilc_model.py**: Core cosmology model implementation
  - `LVILCModel` class with methods for:
    - Luminosity distance calculations
    - Distance modulus
    - Hubble parameter evolution
    - Horizon dilation effects

- **data_loader.py**: Observational data handling
  - `CosmologyData` class for loading:
    - Supernova observations (Pantheon+, SN 2025wny)
    - BAO measurements (DESI 2025)
    - CMB data (Planck)

- **mcmc_lvilc.py**: MCMC implementation
  - `LVILC_MCMC` class with:
    - Prior and likelihood functions
    - Walker initialization
    - MCMC execution
    - Result extraction

- **analysis.py**: Analysis and visualization
  - `MCMCAnalyzer` class for:
    - Corner plots (parameter correlations)
    - Chain convergence plots
    - Model vs data comparisons
    - Goodness-of-fit statistics

- **run_mcmc.py**: Main execution script

### Model Parameters

The MCMC fits three key parameters:

1. **H₀** (Hubble constant): km/s/Mpc
   - Expected value: -6.73 (negative in LVILC framework)
   - Prior range: [-15.0, 0.0]

2. **M_bh** (Black hole mass): Solar masses
   - Expected value: ~10²³ M☉
   - Prior range: [10²⁰, 10²⁶]

3. **t_fall** (Fall time parameter): Gyr
   - Expected value: ~13.8 Gyr
   - Prior range: [10.0, 20.0]

### Output Files

The analysis produces three main plots:

1. **lvilc_corner_plot.png**: 
   - Shows parameter correlations
   - Displays 1D and 2D marginalized distributions
   - Indicates 68% confidence intervals

2. **lvilc_chains.png**:
   - Histograms of parameter distributions
   - Shows median values
   - Useful for checking convergence

3. **lvilc_model_comparison.png**:
   - Top panel: Supernova Hubble diagram (distance modulus vs redshift)
   - Bottom panel: Hubble parameter evolution H(z)
   - Shows best-fit model and uncertainty bands

### Using Custom Data

To use your own observational data, modify the data loading in `data_loader.py` or provide CSV files with the following format:

**Supernova data (sn_data.csv)**:
```
z,mu,mu_err
0.01,33.5,0.15
0.05,36.2,0.12
...
```

**BAO data (bao_data.csv)**:
```
z,DM_over_rd,DM_over_rd_err
0.51,13.5,0.15
...
```

Then run:
```python
from data_loader import CosmologyData
data = CosmologyData()
data.load_supernova_data('sn_data.csv')
data.load_bao_data('bao_data.csv')
```

### Interpreting Results

The output provides:
- **Median and mean**: Best-fit parameter values
- **68% CI**: 1-sigma confidence intervals
- **χ²/DOF**: Goodness of fit (values near 1 indicate good fit)
- **Autocorrelation time**: Convergence diagnostic

Example output:
```
H0:
  Median: -6.73e+00
  68% CI: [-7.69e+00, -5.77e+00]
  
Goodness of Fit:
  χ² = 45.23
  DOF = 47
  χ²/DOF = 0.96
```

### Advanced Usage

For more control, use the modules directly in a Python script:

```python
from lvilc_model import LVILCModel
from data_loader import CosmologyData
from mcmc_lvilc import LVILC_MCMC
from analysis import MCMCAnalyzer

# Load data
data = CosmologyData()
data.load_supernova_data()

# Run MCMC
mcmc = LVILC_MCMC(data_loader=data)
sampler = mcmc.run_mcmc(nwalkers=32, nsteps=5000)

# Analyze
results = mcmc.get_results(sampler)
analyzer = MCMCAnalyzer(results, data)
analyzer.plot_corner()
analyzer.print_results()
```

### Troubleshooting

**Issue**: "Autocorrelation time could not be calculated"
- **Solution**: Chain may be too short. Increase `--nsteps`

**Issue**: Low acceptance fraction (<0.2)
- **Solution**: Walker initialization may be poor. Try different `--initial-params`

**Issue**: χ²/DOF >> 1
- **Solution**: Model may not fit data well, or uncertainties may be underestimated

### References

- LVILC Theory: See README.md
- MCMC Method: Foreman-Mackey et al. (2013), emcee package
- Cosmological Observations: Pantheon+, DESI 2025, Planck 2018

### Contact

For questions or issues, please open an issue on the GitHub repository.
