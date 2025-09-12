import os
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import warnings

from multipathogen_sero.config import PROJ_ROOT, MODEL_FITS_DIR

## TODO: Add coverage claculator
# just for beta12, beta21, frailty variance
## TODO: Add posterior predictive checks (elpd calculator)
# generate data according to ground truth
# generate frailties according to posterior
# calculate likelihoods
# report elpd and std error
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


def save_fit_diagnose(fit, fit_dir, filename="fit_diagnose.txt"):
    """
    Save the diagnostics from CmdStanPy fit.diagnose() to a text file.

    Parameters
    ----------
    fit : CmdStanMCMC or CmdStanMLE
        The fitted CmdStanPy model object.
    output_path : str or Path
        Path to the output file.
    """
    diagnose_str = fit.diagnose()
    with open(fit_dir / filename, "w") as f:
        f.write(diagnose_str)
    return diagnose_str


def read_fit_csv_dir(fit_dir):
    """Load CmdStanPy chain CSVs and metadata pickle from a fit directory."""
    # Load chains
    chain_files = sorted(glob.glob(os.path.join(fit_dir, '*.csv')))
    if not chain_files:
        raise FileNotFoundError("No chain CSV files found in directory.")
    inference_data = az.from_cmdstan(chain_files)
    return inference_data


def basic_summary(inference_data, suppress_warnings=True):
    """Print basic summary statistics for posterior samples.
    Note: division by zero warnings will appear if any of the
    parameters are deterministically fixed.
    """
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(az.summary(inference_data, round_to=2))
    else:
        print(az.summary(inference_data, round_to=2))


def trace_plot(inference_data, var_names=None, save_dir=None):
    """Plot trace for selected variables and save image in fit_dir/analysis/."""
    axes = az.plot_trace(inference_data, var_names=var_names)
    if save_dir is not None:
        analysis_dir = os.path.join(fit_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        img_path = os.path.join(analysis_dir, "trace_plot.png")
        plt.gcf().savefig(img_path)
    return axes


def pairs_plot(inference_data, var_names=None, save_dir=None, figsize=None):
    """Plot trace for selected variables and save image in fit_dir/analysis/."""
    axes = az.plot_pair(
        inference_data,
        var_names=["betas", "frailty_variance"],  # replace with your variable names
        # kind='scatter',      # or 'kde' for density
        # marginals=True,      # show marginal distributions
        figsize=figsize
    )
    if save_dir is not None:
        analysis_dir = os.path.join(fit_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        img_path = os.path.join(analysis_dir, "pairs_plot.png")
        plt.gcf().savefig(img_path)
    return axes


def posterior_plot(
        inference_data,
        var_names=None,
        ground_truth=None,
        save_dir=None,
        fig_grid=None,
        fig_size=None,
        hdi_prob=0.94):
    """Plot posterior distributions for selected variables, overlay ground truth if provided.
    Also return a dict indicating if ground truth is inside the HDI for each variable/component.
    """
    axes = az.plot_posterior(
        inference_data, var_names=var_names,
        grid=fig_grid,
        figsize=fig_size,
        hdi_prob=hdi_prob
    )
    # Flatten axes if it's a numpy array
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    elif not isinstance(axes, list):
        axes = [axes]
    ax_idx = 0
    coverage_dict = {}
    if ground_truth is not None and var_names is not None:
        for var in var_names:
            coverage_dict[var] = []
            if var in ground_truth:
                true_val = np.ravel(ground_truth[var])
                # Get HDI for each component
                try:
                    summary = az.summary(inference_data, var_names=[var], hdi_prob=hdi_prob)
                    hdi_lower = summary['hdi_{}%'.format(int(100*(1-hdi_prob)/2))]
                    hdi_upper = summary['hdi_{}%'.format(int(100*(1-(1-hdi_prob)/2)))]
                except Exception:
                    hdi_lower = None
                    hdi_upper = None
                for i, v in enumerate(true_val):
                    ax = axes[ax_idx]
                    ax.axvline(v, color='red', linestyle='--')
                    ax.legend(loc='best')
                    # Check coverage
                    if hdi_lower is not None and hdi_upper is not None:
                        # If variable is multidimensional, summary index is (var, component)
                        try:
                            lower = hdi_lower.iloc[i]
                            upper = hdi_upper.iloc[i]
                        except Exception:
                            lower = hdi_lower
                            upper = hdi_upper
                        coverage_dict[var].append(lower <= v <= upper)
                    else:
                        coverage_dict[var].append(None)
                    ax_idx += 1
            else:
                # If no ground truth, still increment ax_idx by number of components
                try:
                    shape = np.shape(inference_data.posterior[var].values)
                    ax_idx += np.prod(shape[1:]) if len(shape) > 1 else 1
                except Exception:
                    ax_idx += 1
    if save_dir is not None:
        analysis_dir = os.path.join(save_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        img_path = os.path.join(analysis_dir, "posterior_plot.png")
        plt.gcf().savefig(img_path)
    return axes, coverage_dict


# Example usage:
if __name__ == "__main__":
    fit_dir = MODEL_FITS_DIR / "fit_1757355989"
    inference_data, metadata = read_fit_csv_dir(fit_dir)
    print("Metadata:", metadata)
    # basic_summary(inference_data)
    trace_plot(inference_data, var_names=["baseline_hazards", "beta_matrix"], save_dir=fit_dir)
    posterior_plot(inference_data, var_names=["baseline_hazards", "beta_matrix"], save_dir=fit_dir)
