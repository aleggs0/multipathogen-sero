# imports
import time
import os

import numpy as np
from cmdstanpy import CmdStanModel

from multipathogen_sero.io import save_metadata_json
from multipathogen_sero.config import MODEL_FITS_DIR, STAN_DIR
from multipathogen_sero.simulate import (
    get_constant_foi,
    generate_uniform_birth_times,
    simulate_infections_seroreversion,
    simulation_to_survey_wide
)
from multipathogen_sero.analyse_chains import save_fit_diagnose

# N_REPEATS = 2

ARRAY_INDEX = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
JOB_ID = int(os.environ.get('SLURM_ARRAY_JOB_ID', 0))
HOSTNAME = os.environ.get('HOSTNAME', '')
TIMESTAMP = int(time.time()),

# initialise settings of the experiment
EXPT_SETTINGS = {
    "runtime_info": {
        "job_id": JOB_ID,
        "array_index": ARRAY_INDEX,
        "hostname": HOSTNAME,
        "timestamp": TIMESTAMP
    },
    "ground_truth_params": {
        "n_pathogens": 2,
        "baseline_hazards": [0.05, 0.10],  # TODO: choose from prior
        "seroreversion_rates": [0.1, 0.1],
        "log_frailty_std": 0.5,
        "beta_mat": [[0, 0.5], [0.5, 0]],
        "seed": 42
    },
    "train_data": {
        "n_people": 250,  # TODO: make this variable
        "t_min": 0,
        "t_max": 100,
        "survey_every": 10,
        "seed": 42 + ARRAY_INDEX
    },
    "test_data": {
        "n_people": 1000,
        "t_min": 0,
        "t_max": 100,
        "survey_every": 10,
        "seed": 42 + ARRAY_INDEX
    },
    "inference_params": {
        "log_baseline_hazard_mean": -1,
        "log_baseline_hazard_scale": 1,
        "beta_scale": 1.0,
        "seroreversion_rate_scale": 1.0,
        "log_frailty_std_scale": 1.0,  # only when frailty is modelled
        "n_frailty_samples": 10**ARRAY_INDEX,  # number of Monte Carlo samples for integration over frailty
        "chains": 4,
        "iter_sampling": 500,
        "iter_warmup": 500,
        "seed": 42
    },
    "notes": ""
}

OUTPUT_DIR = MODEL_FITS_DIR / f"j{JOB_ID}" / f"a{ARRAY_INDEX}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_metadata_json(OUTPUT_DIR, EXPT_SETTINGS)
# define the parameter grid (simulation params, random seed)

# simulate the data
log_frailty_covariance = (
    EXPT_SETTINGS["ground_truth_params"]["log_frailty_std"] ** 2
    * np.eye(EXPT_SETTINGS["ground_truth_params"]["n_pathogens"])
)
birth_times = generate_uniform_birth_times(
    n_people=EXPT_SETTINGS["train_data"]["n_people"],
    t_min=EXPT_SETTINGS["train_data"]["t_min"],
    t_max=EXPT_SETTINGS["train_data"]["t_max"],
    random_seed=EXPT_SETTINGS["train_data"]["seed"]
)
foi_list = [
    get_constant_foi(a=baseline_hazard) for baseline_hazard in EXPT_SETTINGS["ground_truth_params"]["baseline_hazards"]
]


infections_df = simulate_infections_seroreversion(
    n_people=EXPT_SETTINGS["train_data"]["n_people"],
    n_pathogens=EXPT_SETTINGS["ground_truth_params"]["n_pathogens"],
    foi_list=foi_list,
    birth_times=birth_times,
    end_times=EXPT_SETTINGS["train_data"]["t_max"],
    frailty_distribution="lognormal",
    log_frailty_covariance=log_frailty_covariance,
    beta_mat=EXPT_SETTINGS["ground_truth_params"]["beta_mat"],
    seroreversion_rates=EXPT_SETTINGS["ground_truth_params"]["seroreversion_rates"],
    random_seed=EXPT_SETTINGS["ground_truth_params"]["seed"]
)

survey_every = EXPT_SETTINGS["train_data"]["survey_every"]
survey_times = {
    # i + 1: survey_every * np.arange(np.floor(birth_times[i]/survey_every)+1, np.floor(t_max/survey_every)+1)
    i + 1: np.insert(
        survey_every * np.arange(np.floor(birth_times[i] / survey_every) + 1, np.floor(EXPT_SETTINGS["train_data"]["t_max"] / survey_every) + 1),
        0, birth_times[i]
    )
    for i in range(EXPT_SETTINGS["train_data"]["n_people"])
}
survey_wide = simulation_to_survey_wide(
    infections_df,
    survey_times=survey_times
)
# exclude individuals with only one row in survey_wide
survey_wide = survey_wide.groupby('individual').filter(lambda x: len(x) > 1)
n_nontrivial_individuals = len(survey_wide['individual'].unique())

birth_times_test = generate_uniform_birth_times(
    n_people=EXPT_SETTINGS["test_data"]["n_people"],
    t_min=EXPT_SETTINGS["test_data"]["t_min"],
    t_max=EXPT_SETTINGS["test_data"]["t_max"],
    random_seed=EXPT_SETTINGS["test_data"]["seed"]
)
infections_df_test = simulate_infections_seroreversion(
    n_people=EXPT_SETTINGS["test_data"]["n_people"],
    n_pathogens=EXPT_SETTINGS["ground_truth_params"]["n_pathogens"],
    foi_list=foi_list,
    birth_times=birth_times_test,
    end_times=EXPT_SETTINGS["test_data"]["t_max"],
    frailty_distribution="lognormal",
    log_frailty_covariance=log_frailty_covariance,
    beta_mat=EXPT_SETTINGS["ground_truth_params"]["beta_mat"],
    seroreversion_rates=EXPT_SETTINGS["ground_truth_params"]["seroreversion_rates"],
    random_seed=EXPT_SETTINGS["ground_truth_params"]["seed"]
)
survey_every_test = EXPT_SETTINGS["test_data"]["survey_every"]
survey_times_test = {
    # i + 1: survey_every * np.arange(np.floor(birth_times_test[i]/survey_every)+1, np.floor(t_max/survey_every)+1)
    i + 1: np.insert(
        survey_every * np.arange(np.floor(birth_times_test[i] / survey_every) + 1, np.floor(EXPT_SETTINGS["test_data"]["t_max"] / survey_every) + 1),
        0, birth_times_test[i]
    )
    for i in range(EXPT_SETTINGS["test_data"]["n_people"])
}
survey_wide_test = simulation_to_survey_wide(
    infections_df_test,
    survey_times=survey_times_test
)
survey_wide_test = survey_wide_test.groupby('individual').filter(lambda x: len(x) > 1)
n_nontrivial_individuals_test = len(survey_wide_test['individual'].unique())

# fit both models on the data
stan_data = {
    "K": EXPT_SETTINGS["ground_truth_params"]["n_pathogens"],
    "N": n_nontrivial_individuals,
    "num_obs": survey_wide.groupby('individual').size().values,
    "num_obs_total": len(survey_wide),
    "obs_times": survey_wide['time'].values,
    "serostatus": survey_wide[[col for col in survey_wide.columns if col.startswith('serostatus_')]].values.astype(int),
    "N_test": n_nontrivial_individuals_test,
    "num_obs_test": survey_wide_test.groupby('individual').size().values,
    "num_obs_total_test": len(survey_wide_test),
    "obs_times_test": survey_wide_test['time'].values,
    "serostatus_test": survey_wide_test[[col for col in survey_wide_test.columns if col.startswith('serostatus_')]].values.astype(int),
    "n_frailty_samples": EXPT_SETTINGS["inference_params"]["n_frailty_samples"],
    "log_baseline_hazard_mean": EXPT_SETTINGS["inference_params"]["log_baseline_hazard_mean"],
    "log_baseline_hazard_scale": EXPT_SETTINGS["inference_params"]["log_baseline_hazard_scale"],
    "beta_scale": EXPT_SETTINGS["inference_params"]["beta_scale"],
    "seroreversion_rate_scale": EXPT_SETTINGS["inference_params"]["seroreversion_rate_scale"],
    "log_frailty_std_scale": EXPT_SETTINGS["inference_params"]["log_frailty_std_scale"]
}
model = CmdStanModel(
    stan_file=os.path.join(STAN_DIR, "pairwise_serology_seroreversion_frailty.stan")
)
fit = model.sample(
    data=stan_data,
    chains=EXPT_SETTINGS["inference_params"]["chains"],
    iter_sampling=EXPT_SETTINGS["inference_params"]["iter_sampling"],
    iter_warmup=EXPT_SETTINGS["inference_params"]["iter_warmup"],
    parallel_chains=EXPT_SETTINGS["inference_params"]["chains"],
    seed=EXPT_SETTINGS["inference_params"]["seed"],
    show_progress=False
)
fit.save_csvfiles(OUTPUT_DIR / "pairwise_serology_seroreversion_frailty")
# print diagnostics
print(save_fit_diagnose(fit, OUTPUT_DIR / "pairwise_serology_seroreversion_frailty"))

# repeat for other fit
# save relevant plots


# do elpd
# simulate the data
