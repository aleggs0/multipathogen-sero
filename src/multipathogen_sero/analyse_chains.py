import os
import json
import glob
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import cmdstanpy
import arviz as az
import math
import warnings


VARS_OF_INTEREST = ["betas", "log_frailty_std", "baseline_hazards", "seroreversion_rates"]


def diagnose(fit, save_dir, filename="fit_diagnose.txt"):
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
    with open(save_dir / filename, "w") as f:
        f.write(diagnose_str)
    return diagnose_str


def read_fit_csv_dir(fit_dir):
    """Load CmdStanPy chain CSVs from a fit directory.
    Hint: an arviz inference data object can be obtained from the fit by
    az.from_cmdstanpy(fit)"""
    # Load chains
    chain_files = sorted(glob.glob(os.path.join(fit_dir, '*.csv')))
    if not chain_files:
        raise FileNotFoundError("No chain CSV files found in directory.")
    fit = cmdstanpy.from_csv(chain_files)
    return fit


def basic_summary(inference_data, save_dir=None, suppress_warnings=True):
    """Get basic summary statistics for posterior samples.
    Note: division by zero warnings will appear if any of the
    parameters are deterministically fixed.
    """
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summary = az.summary(inference_data, round_to=2)
    else:
        summary = az.summary(inference_data, round_to=2)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        summary_path = os.path.join(save_dir, "summary.csv")
        summary.to_csv(summary_path)
    return summary


def trace_plot(inference_data, var_names=None, save_dir=None):
    """Plot trace for selected variables and save image in save_dir/."""
    if var_names is None:
        var_names = [var for var in VARS_OF_INTEREST if var in inference_data.posterior]
    
    axes = az.plot_trace(inference_data, var_names=var_names)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, "trace_plot.png")
        plt.gcf().savefig(img_path)
    return axes


def pairs_plot(inference_data, var_names=None, save_dir=None, figsize=None):
    """Plot trace for selected variables and save image in save_dir/."""
    if var_names is None:
        var_names = [var for var in VARS_OF_INTEREST if var in inference_data.posterior]
    axes = az.plot_pair(
        inference_data,
        var_names=var_names,
        # kind='scatter',      # or 'kde' for density
        # marginals=True,      # show marginal distributions
        figsize=figsize
    )
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, "pairs_plot.png")
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
    if var_names is None:
        var_names = [var for var in VARS_OF_INTEREST if var in inference_data.posterior]
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
                    # Check coverage
                    if hdi_lower is not None and hdi_upper is not None:
                        # If variable is multidimensional, summary index is (var, component)
                        try:
                            lower = hdi_lower.iloc[i]
                            upper = hdi_upper.iloc[i]
                        except Exception:
                            lower = hdi_lower
                            upper = hdi_upper
                        coverage_dict[var].append(int(lower <= v <= upper))
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
    else:
        coverage_dict = None
    plt.subplots_adjust(hspace=0.4)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, "posterior_plot.png")
        plt.gcf().savefig(img_path)
        if ground_truth is not None:
            coverage_path = os.path.join(save_dir, "coverage.json")
            with open(coverage_path, "w") as f:
                json.dump(coverage_dict, f, indent=4)
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
    inference_data, var_names=None, save_dir=None
):
    """
    Plot energy (from sample_stats) against lp (from sample_stats),
    and against each parameter in var_names (from posterior).
    Save all plots in a single figure.
    """
    if var_names is None:
        var_names = [var for var in VARS_OF_INTEREST if var in inference_data.posterior]

    energy = inference_data.sample_stats["energy"].values.flatten()
    lp = inference_data.sample_stats["lp"].values.flatten()

    plots = []

    # Prepare plot data
    plots.append(("energy vs lp", energy, lp, "lp"))

    for param in var_names:
        if param not in inference_data.posterior:
            continue
        values = inference_data.posterior[param]
        param_dims = values.dims[2:]  # skip chain and draw
        param_coords = {dim: values.coords[dim].values for dim in param_dims}
        arr = values.values  # shape: (chain, draw, ...)
        arr_flat = arr.reshape(-1, *arr.shape[2:])

        if arr_flat.ndim == 1:
            plots.append((f"energy vs {param}", energy, arr_flat, param))
        elif arr_flat.ndim == 2:
            dim = param_dims[0]
            for i, coord in enumerate(param_coords[dim]):
                plots.append((f"energy vs {param}[{coord}]", energy, arr_flat[:, i], f"{param}[{coord}]"))
        elif arr_flat.ndim == 3:
            dim0, dim1 = param_dims
            for i, coord0 in enumerate(param_coords[dim0]):
                for j, coord1 in enumerate(param_coords[dim1]):
                    plots.append((f"energy vs {param}[{coord0},{coord1}]", energy, arr_flat[:, i, j], f"{param}[{coord0},{coord1}]"))

    n_plots = len(plots)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for idx, (title, x, y, ylabel) in enumerate(plots):
        ax = axes[idx]
        ax.scatter(x, y, alpha=0.5)
        ax.set_xlabel("energy")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    # Hide unused axes
    for ax in axes[n_plots:]:
        ax.axis('off')

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir / "energy_vs_params.png")
    plt.close(fig)


# Example usage (doesn't actually work without a real fit directory)
if __name__ == "__main__":
    from multipathogen_sero.config import MODEL_FITS_DIR
    fit_dir = MODEL_FITS_DIR / "fit_1757355989"
    inference_data, metadata = read_fit_csv_dir(fit_dir)
    print("Metadata:", metadata)
    basic_summary(inference_data, save_dir=fit_dir)
    trace_plot(inference_data, save_dir=fit_dir)
    posterior_plot(inference_data, save_dir=fit_dir)
