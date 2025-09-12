import pickle
import time
import numpy as np

from cmdstanpy import CmdStanModel
import arviz as az

from multipathogen_sero import (
    simulate_infections_seroreversion,
    simulation_to_survey_wide,
    get_constant_foi,
    get_constant_foi_survivor
)

from multipathogen_sero.config import PROJ_ROOT, STAN_DIR, MODEL_FITS_DIR

include_interations = True  # Whether to include interaction terms in the model
include_frailty = True  # Whether to include frailty in the model

n_people = 1000  # number of individuals n
np.random.seed(42)  # For reproducibility
t_max = 100
birth_times = np.random.uniform(0, t_max, size=n_people)  # Random birth times for individuals
n_pathogens=2 #number of pathogens K

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
log_frailty_covariance = np.ones((n_pathogens, n_pathogens)) * (0.1 if include_frailty else 0.0)
true_beta_scale = 1.0
true_proportion_interactions = 1.0 if include_interations else 0.0
interaction_indicator = np.random.binomial(1, true_proportion_interactions, size=(n_pathogens, n_pathogens))
interaction_indicator[np.arange(n_pathogens), np.arange(n_pathogens)] = 0
beta_mat = np.random.normal(0, true_beta_scale, size=(n_pathogens, n_pathogens)) * interaction_indicator
interaction_mat = np.exp(beta_mat)

simulate_infections_seroreversion_df = simulate_infections_seroreversion(
    n_people,
    n_pathogens,
    foi_list,
    interaction_mat=interaction_mat,
    seroreversion_rates=seroreversion_rates,
    birth_times=birth_times,
    # log_frailty_covariance=log_frailty_covariance,
    end_times=t_max,
    max_fois=None,
    random_seed=42
)

survey_every = 10.0
survey_times = {
    # i + 1: survey_every * np.arange(np.floor(birth_times[i]/survey_every)+1, np.floor(t_max/survey_every)+1)
    i+1: np.insert(
        survey_every * np.arange(np.floor(birth_times[i]/survey_every)+1, np.floor(t_max/survey_every)+1),
        0, birth_times[i]
    )
    for i in range(n_people)
}
survey_wide = simulation_to_survey_wide(
    simulate_infections_seroreversion_df,
    survey_times=survey_times
)
# exclude individuals with only one row in survey_wide
survey_wide = survey_wide.groupby('individual').filter(lambda x: len(x) > 1)
n_nontrivial_individuals = len(survey_wide['individual'].unique())

"""
data {
    int<lower=1> N;                         // Number of individuals
    array[N] int<lower=1> num_tests;                 // Number of serological tests for each individual
    int<lower=1> num_tests_total; // Total number of serological tests across all individuals
    int<lower=2, upper=2> K;                         // Number of pathogens
    array[num_tests_total] real test_times; // Time of each serological test
    array[num_tests_total,K] int<lower=0, upper=1> serostatus; // Seropositivity for each test and pathogen
    
    real log_baseline_hazard_mean; // Mean for normal prior on log baseline hazards
    real <lower=0> log_baseline_hazard_scale; // Scale for normal prior on log baseline hazards
    real <lower=0> beta_scale; // scale for Laplace prior on log hazard ratios
    real <lower=0> seroreversion_rate_scale; // scale for exponential prior on seroreversion rates
    real <lower=0> frailty_scale_scale; // scale for exponential prior on variance of individual frailties
}
"""
# TODO: write tests
stan_data = {
    "N": n_nontrivial_individuals,
    "K": n_pathogens,
    "num_tests": survey_wide.groupby('individual').size().values,
    "num_tests_total": len(survey_wide),
    "test_times": survey_wide['time'].values,
    "serostatus": survey_wide[[col for col in survey_wide.columns if col.startswith('serostatus_')]].values.astype(int),  # Convert to int for Stan
    "log_baseline_hazard_mean": -1,
    "log_baseline_hazard_scale": 1,
    "beta_scale": 1.0,  # scale for Laplace prior on log hazard ratios
    "seroreversion_rate_scale": 1.0,
    "frailty_scale_scale": 1.0,
}

model = CmdStanModel(
    stan_file=STAN_DIR / "pairwise_serology_seroreversion_frailty.stan"
)

chains = 2
iter_sampling = 50
iter_warmup = 50
parallel_chains = 1
seed = 123
max_treedepth = 10
fit = model.sample(
    data=stan_data,
    chains=chains,
    iter_sampling=iter_sampling,
    iter_warmup=iter_warmup,
    parallel_chains=parallel_chains,
    seed=seed,
    max_treedepth=max_treedepth,
    show_progress=True,
    show_console=False
)

name = f"fit_{int(time.time())}"

fit.save_csvfiles(
    MODEL_FITS_DIR / name
)
fit_metadata = {
    "time": int(time.time()),
    "data": {
        "n_people": n_people,
        "n_pathogens": n_pathogens,
        "baseline_hazards": baseline_hazards,
        "survey_every": survey_every,
        "t_max": t_max,
        "log_frailty_covariance": log_frailty_covariance,
        "beta_mat": beta_mat,
    },
    "model": "multiplex_serology",
    "model_params": {
        "chains": chains,
        "iter_sampling": iter_sampling,
        "iter_warmup": iter_warmup,
        "seed": seed,
        "max_treedepth": max_treedepth
    },
    "notes": ""
}
with open(PROJ_ROOT / "model_fits" / name / "metadata.pkl", "wb") as f:
    pickle.dump(fit_metadata, f)
