# Economics Analysis

This directory contains the economic analysis code for the subnet security model.

## Directory Structure

### Core Analysis (`/core`)
Core economic analysis and visualization scripts:

- `economic_analysis.py`: Main economic analysis including fee floors and ceilings
- `cost_analysis.py`: Analysis of compute costs and their impact
- `subnet_analysis.py`: Subnet-specific economic analysis
- `security_analysis.py`: Security analysis including tamper detection
- `parameter_optimization.py`: Parameter tuning and optimisation

### Sensitivity Analysis (`/sensitivity`)
Sensitivity analysis scripts for key parameters:

- `cost_sensitivity.py`: Sensitivity analysis for compute costs
- `subnet_sensitivity.py`: Sensitivity analysis for subnet parameters

### Legacy Code (`/legacy`)
Historical and development versions of analysis scripts based on previous crypto-economic or architural assumptions

- `extra_metrics.py`: Additional economic metrics analysis
- `further_analysis.py`: Extended economic analysis
- `legacy_payoff_optimisation.py`: Original payoff optimisation code
- `plot_sanity_check.py`: Comprehensive sanity checks
- `cheap_sanity_check.py`: Quick validation checks
- `user_cost_tuning.py`: User cost parameter tuning

## Usage

1. Run core analysis scripts first to generate base results
2. Use sensitivity analysis scripts to explore parameter variations
3. All plots are saved in the `sim_data2` directory

## Dependencies

- NumPy
- Matplotlib
- SciPy

## Output

All analysis results and plots are saved in the `sim_data2` directory with the following structure:
- `results_stage1.npz`: Stage 1 simulation results
- `results_stage2.npz`: Stage 2 simulation results