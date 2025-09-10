import os
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

from multipathogen_sero.config import PROJ_ROOT, MODEL_FITS_DIR

## TODO: Add coverage claculator
# just for beta12, beta21, frailty variance
## TODO: Add posterior predictive checks (elpd calculator)
# generate data according to ground truth
# generate frailties according to posterior
# calculate likelihoods
# report elpd and std error
## TODO: Add pair plots
# just for beta12, beta21, frailty variance pairwise
## TODO: Include plots for truth vs fitted posterior
"""
print(f"True baseline hazards: {baseline_hazards}")
axes = az.plot_forest(idata, var_names=["baseline_hazards"], 
                      hdi_prob=0.95, combined=True)
ax = axes[0] if isinstance(axes, (list, np.ndarray)) else axes
yticks = ax.get_yticks()
for i, val in enumerate(baseline_hazards[::-1]):
    ax.scatter(val, yticks[i], marker='x', color='red', s=100, label='True value' if i == 0 else "")
if len(baseline_hazards) > 0:
    ax.legend(loc='best')
plt.title("95% Credible Intervals")
plt.tight_layout()
plt.show()

print(f"True beta coefficients: {beta_mat}")
axes = az.plot_forest(idata, var_names=["beta_matrix"], 
                      hdi_prob=0.95, combined=True)
ax = axes[0] if isinstance(axes, (list, np.ndarray)) else axes
plt.axvline(0, color='black', linestyle='--', alpha=0.7)
yticks = ax.get_yticks()
for i, val in enumerate(beta_mat.flatten()[::-1]):
    ax.scatter(val, yticks[i], marker='x', color='red', s=100, label='True value' if i == 0 else "")
if beta_mat.size > 0:
    ax.legend(loc='best')
plt.title("95% Credible Intervals")
plt.tight_layout()
plt.show()
"""
## TODO: Include diagnostic/summary printing

"""
# Print summary
print(fit.diagnose())

# Check R-hat and ESS
df_summary = fit.summary()
print("Any R-hat > 1.01?", (df_summary["R_hat"] > 1.01).any())
print("Any ESS < 400?", (df_summary["ESS_bulk"] < 400).any())
print(df_summary)

# Optional: convert to ArviZ for easier plotting
idata = az.from_cmdstanpy(posterior=fit)

# Plot trace plots
az.plot_trace(idata, var_names=["baseline_hazards", "betas", "seroreversion_rates"])
plt.tight_layout()
plt.show()
"""

def load_fit_results(fit_dir):
    """Load CmdStanPy chain CSVs and metadata pickle from a fit directory."""
    # Load chains
    chain_files = sorted(glob.glob(os.path.join(fit_dir, '*.csv')))
    if not chain_files:
        raise FileNotFoundError("No chain CSV files found in directory.")
    inference_data = az.from_cmdstan(chain_files)
    
    # Load metadata
    metadata_path = os.path.join(fit_dir, 'metadata.pkl')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("metadata.pkl not found in directory.")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return inference_data, metadata

def basic_summary(inference_data):
    """Print basic summary statistics for posterior samples.
    Note: division by zero warnings will appear if any of the
    parameters are deterministically fixed.
    """
    print(az.summary(inference_data, round_to=2))

def save_trace_plot(inference_data, var_names=None, fit_dir=None):
    """Plot trace for selected variables and save image in fit_dir/analysis/."""
    fig = az.plot_trace(inference_data, var_names=var_names)
    if fit_dir is not None:
        analysis_dir = os.path.join(fit_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        img_path = os.path.join(analysis_dir, "trace_plot.png")
        plt.gcf().savefig(img_path)

def save_posterior_plot(inference_data, var_names=None, fit_dir=None):
    """Plot posterior distributions for selected variables."""
    fig = az.plot_posterior(inference_data, var_names=var_names)
    if fit_dir is not None:
        analysis_dir = os.path.join(fit_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        img_path = os.path.join(analysis_dir, "posterior_plot.png")
        plt.gcf().savefig(img_path)

def load_metadata(fit_dir):
    """Load metadata.pkl from a fit directory."""
    metadata_path = os.path.join(fit_dir, 'metadata.pkl')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("metadata.pkl not found in directory.")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return metadata

# Example usage:
if __name__ == "__main__":
    fit_dir = MODEL_FITS_DIR / "fit_1757355989"
    inference_data, metadata = load_fit_results(fit_dir)
    print("Metadata:", metadata)
    # basic_summary(inference_data)
    save_trace_plot(inference_data, var_names=["baseline_hazards", "beta_matrix"], fit_dir=fit_dir)
    save_posterior_plot(inference_data, var_names=["baseline_hazards", "beta_matrix"], fit_dir=fit_dir)