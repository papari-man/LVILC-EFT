# LVILC - LoopVacuum Invisible Light Cosmology

## Theory

The LoopVacuum Invisible Light Cosmology (LVILC) proposes the universe is inside a single giant black hole, with us eternally falling toward its center. The "expansion" is an illusion from horizon dilation, accelerating time. This explains dark energy and fits DESI 2025/JWST data, with H₀ at -6.73 ± 0.96 km/s/Mpc due to early black hole growth. Supported by MCMC analysis and SN 2025wny time delays.

## MCMC Implementation

This repository provides a complete MCMC (Markov Chain Monte Carlo) implementation for fitting the LVILC model to cosmological observations.

### Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the MCMC analysis:
```bash
python run_mcmc.py
```

This will fit the LVILC model to observational data and generate:
- Corner plots showing parameter constraints
- Model comparison plots with supernova data
- Hubble parameter evolution plots

### Features

- **Complete MCMC pipeline** using the `emcee` sampler
- **Observational data support**: Supernova (Pantheon+/SN 2025wny), BAO (DESI 2025), CMB (Planck)
- **Model implementation**: Full LVILC cosmology with horizon dilation effects
- **Visualization tools**: Corner plots, chain diagnostics, model comparisons
- **Parameter constraints**: Bayesian inference for H₀, M_bh, and t_fall

### Parameters

The MCMC fits three key parameters:

1. **H₀** (Hubble constant): -6.73 ± 0.96 km/s/Mpc (negative in LVILC framework)
2. **M_bh** (Black hole mass): ~10²³ M☉ (universe-scale black hole)
3. **t_fall** (Fall time): ~13.8 Gyr (comparable to age of universe)

### Documentation

See [USER_GUIDE.md](USER_GUIDE.md) for detailed documentation, examples, and advanced usage.

### Testing

Run tests to validate the implementation:
```bash
python test_mcmc.py
```

### Files

- `lvilc_model.py` - Core cosmology model with horizon dilation
- `data_loader.py` - Observational data handling
- `mcmc_lvilc.py` - MCMC implementation with emcee
- `analysis.py` - Result analysis and visualization
- `run_mcmc.py` - Main execution script
- `test_mcmc.py` - Test suite
- `config.ini` - Configuration file
- `USER_GUIDE.md` - Comprehensive documentation

### Requirements

- Python 3.7+
- numpy, scipy, matplotlib
- emcee (MCMC sampler)
- corner (visualization)
- astropy (cosmological calculations)

### License

See LICENSE file for details.
