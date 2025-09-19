# multipathogen-sero

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Modelling multipathogen serological dynamics

Setup includes:
1. cloning this repository
2. creating an environment e.g.
   ```bash
   conda env create -f environment.yml
   ```
3. Install the Python kernel for Jupyter:
   ```bash
   python -m ipykernel install --user --name multipathogen-sero --display-name "Python3.11 (multipathogen-sero)"
   ```

Warning: this may not work without some fiddling around with cmdstanpy

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── Makefile           <- Makefile with commands like `make requirements` or `make clean`
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         multipathogen_sero and configuration for tools like black
├── environment.yml    <- Conda environment specification
│
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── hpc_work           <- Scripts and configurations for HPC execution
│   ├── archive        <- Archived experiment scripts
│   ├── experiments    <- Current HPC experiment scripts and SLURM job files
│   └── utils          <- Utility scripts for HPC environment
│
├── notebooks          <- Jupyter notebooks for analysis and exploration
│
├── outputs            <- Generated outputs from experiments and models
│   ├── from_hpc       <- Results from HPC runs
│   └── from_local     <- Results from local runs
│
└── src
    └── multipathogen_sero   <- Source code for use in this project
        │
        ├── __init__.py             <- Makes multipathogen_sero a Python module
        ├── config.py               <- Store useful variables and configuration
        ├── analyse_chains.py       <- Functions for analyzing MCMC chains
        ├── io.py                   <- Input/output utilities
        ├── simulate.py             <- Code for simulation
        │
        ├── experiments             <- Experiment runners and configurations
        │   └── frailty_known.py    <- Specific experiment implementations
        │
        └── models                  <- Model implementations
            ├── compile_stan.py     <- Stan model compilation script
            ├── model.py            <- Main model class (PairwiseModel)
            └── stan                <- Stan model files
                ├── functions.stan  <- Shared Stan functions
                ├── pairwise_serology_*.stan <- Various model implementations
                └── multiplex_serology_*.stan <- Multiplex model variants