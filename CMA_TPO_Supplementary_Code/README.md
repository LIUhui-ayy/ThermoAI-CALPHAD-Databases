# CMA-TPO Supplementary Code

This folder contains the parameter-optimization workflow used for the CALPHAD assessment.

## Files

### 1. `CMA_TPO_Optimization.py`
Main thermodynamic parameter optimization program.

Main functions:
- Reads the initial TDB and ESPEI-compatible JSON datasets.
- Groups thermodynamic-property data by property type and source.
- Evaluates ZPF phase-equilibrium constraints using thermodynamic driving-force penalties.
- Applies additional physical-stability checks for high-temperature liquid/solid behavior.
- Combines phase-equilibrium and thermodynamic-property contributions into the optimization score.
- Optimizes Gibbs-energy parameters with Optuna's CMA-ES sampler.
- Executes objective-function evaluations in parallel with Dask.
- Stores all trials in an SQLite Optuna database.

Key run settings in the supplied code:
- Optuna sampler: `CmaEsSampler`
- Startup trials: 500
- Population size: 64
- Restart strategy: IPOP
- Dask workers: 16
- Total trials: 20,000
- Maximum trials per process batch: 2,000
- Study name: `entropy_opt`

Required inputs:
- `Cu-Mg-generated1.tdb`
- `input-data/` containing ESPEI-compatible JSON datasets

Primary outputs:
- `B6-CU-MG_Final_Optimized.db`
- `B6-CU-MG_Optimization_Final.log`
- `ZPF_Score_Summary.log`
- `ZPF_Phase_Details.log`
- `Optimization_Scheduler.json`

### 2. `Export_Best_TDB.py`
Reads the Optuna SQLite database, identifies the globally best completed trial, writes its optimized symbolic parameters into the initial TDB, and exports the optimized thermodynamic database.

Required inputs:
- `B6-CU-MG_Final_Optimized.db`
- `Cu-Mg-generated1.tdb`

Output:
- `B6-CU-MG_Best_Final.tdb`

### 3. `Run_CMA_TPO_Batches.py`
Batch execution helper for long optimization jobs.

The main optimizer is configured to stop after at most 2,000 new trials per process so that Dask workers, memory, ports, and SQLite locks can be released cleanly. This helper repeatedly launches the optimizer until the 20,000-trial target can be reached.

For a fresh calculation:
- 20,000 total trials / 2,000 trials per batch = 10 batches.

## Recommended execution order

1. Place the initial TDB file and `input-data` directory in the working directory.
2. Run:

   `python Run_CMA_TPO_Batches.py`

3. After the optimization is complete, export the best parameter set:

   `python Export_Best_TDB.py`

## Python dependencies

The supplied scripts use:
- NumPy
- SymPy
- Optuna
- Dask / distributed
- PyCalphad
- ESPEI
- TinyDB
- SQLite (Python standard library)

Exact package versions should be reported from the environment used for the calculations if full computational reproducibility is required.

## Notes on reproducibility

The numerical objective function, thermodynamic driving-force treatment, property weighting, physical constraints, CMA-ES configuration, and Dask evaluation logic in `CMA_TPO_Optimization.py` are preserved from the supplied optimization code.

For Supplementary Code packaging, only the following operational changes were made:
1. Files were renamed using descriptive English filenames.
2. The batch runner was updated to call `CMA_TPO_Optimization.py`.
3. The batch count was set to 10 so that a fresh run is consistent with 20,000 total trials and 2,000 trials per batch.

No thermodynamic model equations, objective-function terms, penalty equations, weights, parameter-search ranges, or optimization settings in the main optimization program were changed.
