import os
import pickle
import glob
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import arviz as az
import warnings

from multipathogen_sero.config import PROJ_ROOT, MODEL_FITS_DIR


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
        var_names=["betas", "frailty_scale"],  # replace with your variable names
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
                    hdi_lower = summary['hdi_{}%'.format(int(100 * (1 - hdi_prob) / 2))]
                    hdi_upper = summary['hdi_{}%'.format(int(100 * (1 - (1 - hdi_prob) / 2)))]
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


def elpd_using_test_set(
    idata,
    var_name="log_lik_test"
):
    """Compute approximate ELPD using a test set from an inference dataset.
    Parameters
    ----------
    idata : arviz.InferenceData
        The inference dataset containing posterior samples and pointwise log-likelihoods.
    var_name : str
        The name of the variable containing pointwise log-likelihoods.
    Returns
    -------
    elpd : float
        The estimated expected log predictive density.
    se_elpd : float
        The standard error of the ELPD estimate [over individuals in the test set].
    lse : np.ndarray
        The pointwise log predictive densities for each inidividual in the test set.
    """
    if var_name not in idata.posterior:
        raise ValueError(f"Variable '{var_name}' not found in posterior.")
    log_lik = idata.posterior[var_name].values  # shape (chains, draws, n_data)
    # Combine chains and draws
    log_lik_reshaped = log_lik.reshape(-1, log_lik.shape[-1])  # shape (samples, n_data)
    # Compute pointwise log predictive density using log-sum-exp trick
    lse = logsumexp(log_lik_reshaped, axis=0) - np.log(log_lik_reshaped.shape[0])  # shape (n_data,)
    elpd = np.sum(lse)  # scalar
    # Standard error of ELPD
    se_elpd = np.sqrt(len(lse) * np.var(lse))  # scalar
    return elpd, se_elpd, lse


def compare_using_test_set(
    idata1, idata2,
    var_name="log_lik_test"
):
    """Compare two inference datasets using approximate ELPD.
    Parameters
    ----------
    idata1, idata2 : arviz.InferenceData
        The inference datasets to compare.
    var_name : str
        The name of the variable containing pointwise log-likelihoods.
    Returns
    -------
    elpd_diff : float
        The difference in ELPD between the two datasets.
    se_diff : float
        The standard error of the ELPD difference.
    """
    _, _, lse1 = elpd_using_test_set(idata1, var_name=var_name)
    _, _, lse2 = elpd_using_test_set(idata2, var_name=var_name)
    diff = lse1 - lse2
    elpd_diff = np.sum(diff)
    se_diff = np.sqrt(len(diff) * np.var(diff))
    return elpd_diff, se_diff


def plot_energy_vs_lp_and_params(
    inference_data, var_names=["betas", "frailty_scale"]
):
    """
    Plot energy (from sample_stats) against lp (from sample_stats),
    and against each parameter in var_names (from posterior).
    """
    energy = inference_data.sample_stats["energy"].values.flatten()
    lp = inference_data.sample_stats["lp"].values.flatten()

    # Plot energy vs lp
    plt.figure(figsize=(5, 5))
    plt.scatter(energy, lp, alpha=0.5)
    plt.xlabel("energy")
    plt.ylabel("lp")
    plt.title("energy vs lp")
    plt.show()

    for param in var_names:
        if param not in inference_data.posterior:
            continue
        values = inference_data.posterior[param]
        param_dims = values.dims[2:]  # skip chain and draw
        param_coords = {dim: values.coords[dim].values for dim in param_dims}
        arr = values.values  # shape: (chain, draw, ...)
        arr_flat = arr.reshape(-1, *arr.shape[2:])

        if arr_flat.ndim == 1:
            # Scalar parameter
            plt.figure(figsize=(5, 5))
            plt.scatter(energy, arr_flat, alpha=0.5)
            plt.xlabel("energy")
            plt.ylabel(param)
            plt.title(f"energy vs {param}")
            plt.show()
        elif arr_flat.ndim == 2:
            # 1D parameter (vector)
            dim = param_dims[0]
            for i, coord in enumerate(param_coords[dim]):
                plt.figure(figsize=(5, 5))
                plt.scatter(energy, arr_flat[:, i], alpha=0.5)
                plt.xlabel("energy")
                plt.ylabel(f"{param}[{coord}]")
                plt.title(f"energy vs {param}[{coord}]")
                plt.show()
        elif arr_flat.ndim == 3:
            # 2D parameter (matrix)
            dim0, dim1 = param_dims
            for i, coord0 in enumerate(param_coords[dim0]):
                for j, coord1 in enumerate(param_coords[dim1]):
                    plt.figure(figsize=(5, 5))
                    plt.scatter(energy, arr_flat[:, i, j], alpha=0.5)
                    plt.xlabel("energy")
                    plt.ylabel(f"{param}[{coord0},{coord1}]")
                    plt.title(f"energy vs {param}[{coord0},{coord1}]")
                    plt.show()


# Example usage:
if __name__ == "__main__":
    fit_dir = MODEL_FITS_DIR / "fit_1757355989"
    inference_data, metadata = read_fit_csv_dir(fit_dir)
    print("Metadata:", metadata)
    # basic_summary(inference_data)
    trace_plot(inference_data, var_names=["baseline_hazards", "beta_matrix"], save_dir=fit_dir)
    posterior_plot(inference_data, var_names=["baseline_hazards", "beta_matrix"], save_dir=fit_dir)
