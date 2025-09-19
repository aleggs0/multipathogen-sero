# imports
import time
import os

import numpy as np
from cmdstanpy import CmdStanModel
import arviz as az

from multipathogen_sero.io import save_metadata_json
from multipathogen_sero.config import STAN_DIR, LOCAL_MODEL_FITS_DIR, HPC_MODEL_FITS_DIR
from multipathogen_sero.simulate import (
    get_constant_foi,
    generate_uniform_birth_times,
    simulate_infections_seroreversion,
    simulation_to_survey_wide
)
from multipathogen_sero.analyse_chains import (
    model_comparison_report
)
from multipathogen_sero.models.model import PairwiseModel
from multipathogen_sero.experiments.experiments import get_runtime_info


runtime_info = get_runtime_info()
IS_SLURM_JOB = runtime_info["is_slurm_job"]
JOB_NAME = runtime_info["job_name"]
JOB_ID = runtime_info["job_id"]
ARRAY_INDEX = runtime_info["array_index"]
HOSTNAME = runtime_info["hostname"]
TIMESTAMP = runtime_info["timestamp"]
if IS_SLURM_JOB:
    OUTPUT_DIR = HPC_MODEL_FITS_DIR / f"{JOB_NAME}_j{JOB_ID}" / f"a{ARRAY_INDEX}"
else:
    OUTPUT_DIR = LOCAL_MODEL_FITS_DIR / f"{JOB_NAME}_{TIMESTAMP}"


def get_param_grid(array_index):
    """
    This defines the parameters used for each array index of the experiment.
    """
    beta_mats = [
        [[0, 0], [0, 0]],
        [[0, 0.5], [0.5, 0]]
    ]
    log_frailty_stds = [0.3, 0.0]
    n_beta = len(beta_mats)
    n_frailty = len(log_frailty_stds)
    total = n_beta * n_frailty
    idx = (array_index - 1) % total
    beta_idx = idx // n_frailty
    frailty_idx = idx % n_frailty
    return beta_mats[beta_idx], log_frailty_stds[frailty_idx]


beta_mat, log_frailty_std = get_param_grid(ARRAY_INDEX)
print(f"Output directory: {OUTPUT_DIR}")

EXPT_SETTINGS = {
    "runtime_info": runtime_info,
    "ground_truth_params": {
        "n_pathogens": 2,
        "baseline_hazards": [0.05, 0.10],
        "seroreversion_rates": [0.05, 0.02],
        "log_frailty_std": log_frailty_std,
        "beta_mat": beta_mat,
        "seed": 42
    },
    "train_data": {
        "n_people": 20,
        "t_min": 0,
        "t_max": 100,
        "survey_every": 10,
        "seed": 42 + ARRAY_INDEX
    },
    "test_data": {
        "n_people": 20,
        "t_min": 0,
        "t_max": 100,
        "survey_every": 10,
        "seed": 2411 + ARRAY_INDEX  # must be different from train seed
    },
    "models": {
        "frailty": "pairwise_serology_seroreversion_frailty.stan",
        "frailty_known": "pairwise_serology_seroreversion_frailty_known.stan",
        "no_frailty": "pairwise_serology_seroreversion.stan",
        "noncentred_frailty": "pairwise_serology_seroreversion_frailty_noncentred.stan"
    },
    "prior_config": {
        "n_pathogens": 2,
        "baseline_hazard_scale": 1.0,
        "beta_scale": 1.0,
        "seroreversion_rate_scale": 1.0,
        "log_frailty_std_scale": 1.0,
        "log_frailty_std": log_frailty_std
    },
    "sampling_config": {
        "n_frailty_samples": 20,
        "chains": 4,
        "iter_sampling": 10,
        "iter_warmup": 10,
        "seed": 42
    },
    "notes": ""
}

save_metadata_json(OUTPUT_DIR, EXPT_SETTINGS)

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


models = {}

for model_name, stan_file_name in EXPT_SETTINGS["models"].items():
    print(f"Running model: {model_name} ({stan_file_name})")

    # Initialize the model
    model = PairwiseModel(
        stan_file_name=stan_file_name,
        stan_dir=STAN_DIR,
        prior_config=EXPT_SETTINGS["prior_config"],
        fit_dir=OUTPUT_DIR / model_name
    )

    # Fit the model
    fit = model.fit_model(
        survey_wide,
        survey_wide_test,
        **EXPT_SETTINGS["sampling_config"]
    )

    # Save the fit
    model.save_fit()

    # Convert to ArviZ InferenceData and generate plots
    idata = model.get_arviz()
    model.generate_plots(
        ground_truth={
            "betas": [
                EXPT_SETTINGS["ground_truth_params"]["beta_mat"][0][1],
                EXPT_SETTINGS["ground_truth_params"]["beta_mat"][1][0]
            ],
            "log_frailty_std": EXPT_SETTINGS["ground_truth_params"]["log_frailty_std"],
            "baseline_hazards": EXPT_SETTINGS["ground_truth_params"]["baseline_hazards"],
            "seroreversion_rates": EXPT_SETTINGS["ground_truth_params"]["seroreversion_rates"]
        }
    )
    models[model_name] = model

model_comparison_report_text = model_comparison_report(
    {model_name: model.idata for model_name, model in models.items()},
    save_dir=OUTPUT_DIR
)
