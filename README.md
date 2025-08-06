# multipathogen-sero

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Modelling multipathogen serological dynamics

Setup includes:
1. cloning this repository
2. creating an environment e.g.
   ```bash
   conda create -n multipathogen-sero python=3.11 ipykernel jupyter_client
   conda activate multipathogen-sero
   ```
3. Install packages from `environment.yml`:
   ```bash
   conda env update --file environment.yml --prune
   ```
4. Install the Python kernel for Jupyter:
   ```bash
   python -m ipykernel install --user --name multipathogen-sero --display-name "Python3.11 (multipathogen-sero)"
   ```

Warning: this may not work without some fiddling around with cmdstanpy

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         multipathogen_sero and configuration for tools like black
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── multipathogen_sero   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes multipathogen_sero a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    └── simulate.py             <- Code for simulation