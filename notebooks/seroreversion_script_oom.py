import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.multitest import multipletests
from scipy.special import erf
from lifelines import CoxTimeVaryingFitter
import cmdstanpy
from cmdstanpy import CmdStanModel
import arviz as az

from multipathogen_sero import (
    simulate_infections_seroreversion,
    simulation_to_regression_df,
    simulation_to_survey_wide,
    get_constant_foi,
    get_constant_foi_survivor
)

from multipathogen_sero.config import PROJ_ROOT, STAN_DIR, MODEL_FITS_DIR

n_people = 100 #number of individuals n
np.random.seed(42)  # For reproducibility
t_max=100
birth_times = np.random.uniform(0, t_max, size=n_people)  # Random birth times for individuals
n_pathogens=3 #number of pathogens K

pathogen_names = [f'Pathogen {i}' for i in range(1,n_pathogens+1)]  # Names for pathogens
#foi_list = [get_exponential_foi(0, 1) for k, pathogen_name in enumerate(pathogen_names)]

baseline_hazards = [0.05*k for k in range(1,n_pathogens+1)]  # Example baseline hazards
seroreversion_rates = [0.02 for k in range(1,n_pathogens+1)]  # Example seroreversion rates

foi_list = [
    get_constant_foi(a=baseline_hazards[k]) for k in range(n_pathogens)
]
survivor_list = [
    get_constant_foi_survivor(a=baseline_hazards[k]) for k in range(n_pathogens)
]
# spike and slab on each coefficient of the interaction matrix
np.random.seed(42)  # For reproducibility
interaction_indicator = np.random.binomial(1, 0.1, size=(n_pathogens, n_pathogens))
interaction_indicator[np.arange(n_pathogens), np.arange(n_pathogens)] = 0
beta_mat = np.random.normal(0, 1, size=(n_pathogens, n_pathogens)) * interaction_indicator
interaction_mat = np.exp(beta_mat)

# log_frailty_covariance = np.ones((n_pathogens, n_pathogens)) * 0.3

simulate_infections_seroreversion_df = simulate_infections_seroreversion(
    n_people,
    n_pathogens,
    foi_list,
    interaction_mat=interaction_mat,
    seroreversion_rates=seroreversion_rates,
    birth_times=birth_times,
    #log_frailty_covariance=log_frailty_covariance,
    end_times=t_max,
    max_fois=None,
    random_seed=42
)

survey_every = 10.0
survey_times = {
    i + 1: survey_every * np.arange(np.floor(birth_times[i]/survey_every)+1, np.floor(t_max/survey_every)+1)
    # i+1: np.insert(
    #     survey_every * np.arange(np.floor(birth_times[i]/survey_every)+1, np.floor(t_max/survey_every)+1),
    #     0, birth_times[i]
    # )
    for i in range(n_people)
}
survey_wide = simulation_to_survey_wide(
    simulate_infections_seroreversion_df,
    survey_times=survey_times
)
# exclude individuals with only one row in survey_wide
survey_wide = survey_wide.groupby('individual').filter(lambda x: len(x) > 1)
n_nontrivial_individuals = len(survey_wide['individual'].unique())

stan_data = {
    "N": n_nontrivial_individuals,
    "K": n_pathogens,
    "num_tests": survey_wide.groupby('individual').size().values,
    "num_tests_total": len(survey_wide),
    "test_times": survey_wide['time'].values,
    "serostatus": survey_wide[[col for col in survey_wide.columns if col.startswith('serostatus_')]].values.astype(int),  # Convert to int for Stan
    "interval": survey_every,
    "log_baseline_hazard_mean": -1,
    "log_baseline_hazard_scale": 0.5,
    "beta_scale": 1.0,  # scale for Laplace prior on log hazard ratios
}

model = CmdStanModel(
    stan_file=os.path.join(STAN_DIR, "multiplex_serology_seroreversion.stan")
)
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_sampling=500,
    iter_warmup=500,
    parallel_chains=4,
    seed=42,
    show_console=False
)

import pickle
import time
name = f"fit_{int(time.time())}"

fit.save_csvfiles(
    MODEL_FITS_DIR / name
)
fit_metadata = {
    "data": {
        "n_people": n_people,
        "n_pathogens": n_pathogens,
        "baseline_hazards": baseline_hazards,
        "seroreversion_rates": seroreversion_rates,
        "survey_every": survey_every,
        "t_max": t_max,
    },
    "model": "multiplex_serology_seroreversion"
}
with open(PROJ_ROOT / "model_fits" / name / "metadata.pkl", "wb") as f:
    pickle.dump(fit_metadata, f)