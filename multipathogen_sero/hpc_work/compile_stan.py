import os
from cmdstanpy import CmdStanModel
from multipathogen_sero.config import STAN_DIR
model1 = CmdStanModel(
    stan_file=os.path.join(STAN_DIR, "pairwise_serology_seroreversion_frailty.stan")
)
model2 = CmdStanModel(
    stan_file=os.path.join(STAN_DIR, "pairwise_serology_seroreversion.stan")
)