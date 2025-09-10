# imports
import time
import os
import json

from multipathogen_sero.io import save_metadata_json
from multipathogen_sero.config import MODEL_FITS_DIR

N_REPEATS = 2
ARRAY_INDEX = int(os.environ['SLURM_ARRAY_TASK_ID'])


# initialise settings of the experiment
fit_metadata = {
    "time": int(time.time()),
    "data": {
        "n_people": 100,  #TODO: make this variable
        "n_pathogens": 2,
        "baseline_hazards": [0.05, 0.10],  # TODO: choose from prior
        "survey_every": 10,
        "t_max": 100,
        "log_frailty_covariance": 1,
        "beta_mat": [[0, 0.5], [-0.5, 0]],
    },
    "model": "pairwise",
    "model_params": {
        "chains": 4,
        "iter_sampling": 500,
        "iter_warmup": 500,
        "seed": 42,
        "max_treedepth": 10
    },
    "notes": ""
}


save_metadata_json(MODEL_FITS_DIR / "temp", fit_metadata)
# define the parameter grid (simulation params, random seed)
# get array job number


# simulate the data
# fit both models on the data
# save both outputs
# save relevant plots
# ssh these out to gate