import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
from cmdstanpy import CmdStanModel
import arviz as az
from multipathogen_sero.analyse_chains import (
    diagnose, trace_plot, pairs_plot, posterior_plot,
    plot_energy_vs_lp_and_params, basic_summary, read_fit_csv_dir
)
from multipathogen_sero.config import STAN_DIR


class PairwiseModel:
    def __init__(self,
                 stan_file_name: str,
                 prior_config: Dict[str, Any],
                 fit_dir: Path,
                 stan_dir: Path = STAN_DIR,
                 analysis_dir: Path = None):
        """
        Initialize ModelRunner with Stan file and prior configuration.

        Args:
            stan_file_name: Name of the Stan file
            stan_dir: Directory containing Stan files
            fit_dir: Directory for output files
            prior_config: Dictionary containing fixed parameters for the stan data
        """
        self.stan_file_name = stan_file_name
        self.stan_dir = stan_dir
        self.output_dir = fit_dir
        if analysis_dir is None:
            self.analysis_dir = self.output_dir / "analysis"
        else:
            self.analysis_dir = analysis_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # Store prior configuration
        self.prior_config = prior_config.copy()

        # Model state
        self.model = None
        self.fit = None
        self.idata = None

    def compile_model(self) -> None:
        """Compile the Stan model."""
        stan_file_path = self.stan_dir / self.stan_file_name
        if not stan_file_path.exists():
            raise FileNotFoundError(f"Stan file not found: {stan_file_path}")
        self.model = CmdStanModel(stan_file=str(stan_file_path))

    def fit_model(self, survey_df: pd.DataFrame, survey_df_test: pd.DataFrame, n_frailty_samples: int = None, **sampling_kwargs) -> Tuple[float, Any]:
        """Fit the model and return fitting time and fit object."""
        if self.model is None:
            self.compile_model()
        self.sampling_kwargs = sampling_kwargs
        self.n_frailty_samples = n_frailty_samples
        stan_data = {
            "K": self.prior_config["n_pathogens"],
            "N": len(survey_df['individual'].unique()),
            "num_obs": survey_df.groupby('individual').size().values,
            "num_obs_total": len(survey_df),
            "obs_times": survey_df['time'].values,
            "serostatus": survey_df[[col for col in survey_df.columns if col.startswith('serostatus_')]].values.astype(int),
            "N_test": len(survey_df_test['individual'].unique()),
            "num_obs_test": survey_df_test.groupby('individual').size().values,
            "num_obs_total_test": len(survey_df_test),
            "obs_times_test": survey_df_test['time'].values,
            "serostatus_test": survey_df_test[[col for col in survey_df_test.columns if col.startswith('serostatus_')]].values.astype(int),
            "n_frailty_samples": self.n_frailty_samples,
            "baseline_hazard_scale": self.prior_config["baseline_hazard_scale"],
            "beta_scale": self.prior_config["beta_scale"],
            "seroreversion_rate_scale": self.prior_config["seroreversion_rate_scale"],
            "log_frailty_std_scale": self.prior_config["log_frailty_std_scale"],
            "log_frailty_std": self.prior_config["log_frailty_std"]
        }
        start_time = time.time()
        self.fit = self.model.sample(
            data=stan_data,
            show_progress=False,
            **self.sampling_kwargs
        )
        end_time = time.time()

        self.fitting_time = end_time - start_time
        print(f"Fitting time for {self.stan_file_name}: {self.fitting_time:.2f} seconds")

        return self.fit

    def save_fit(self) -> str:
        if self.fit is None:
            raise ValueError("Model must be fitted before saving")
        self.fit.save_csvfiles(self.output_dir)
        # Save metadata to JSON (excluding directory paths)
        metadata = {
            "stan_file_name": self.stan_file_name,
            "prior_config": self.prior_config,
            "n_frailty_samples": getattr(self, "n_frailty_samples", None),
            "sampling_kwargs": getattr(self, "sampling_kwargs", {})
        }
        metadata_path = self.output_dir / "fit_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return str(metadata_path)

    @classmethod
    def load_fit(cls, fit_dir: Path, stan_dir: Path = STAN_DIR, analysis_dir: Path = None):
        """
        Load a previously saved fit and metadata, returning a PairwiseModel instance.
        
        Args:
            fit_dir: Directory containing the saved fit files
            stan_dir: Directory containing Stan files
            analysis_dir: Directory for analysis outputs (optional)
        """
        metadata_path = fit_dir / "fit_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Instantiate the model with provided directories
        model = cls(
            stan_file_name=metadata["stan_file_name"],
            fit_dir=fit_dir,
            prior_config=metadata["prior_config"],
            stan_dir=stan_dir,
            analysis_dir=analysis_dir
        )
        model.n_frailty_samples = metadata.get("n_frailty_samples", None)
        model.sampling_kwargs = metadata.get("sampling_kwargs", {})

        # Load the fit from CSV files
        model.fit = read_fit_csv_dir(fit_dir)
        model.get_arviz()
        return model

    def get_arviz(self) -> None:
        """Convert fit to ArviZ InferenceData."""
        if self.fit is None:
            raise ValueError("Model must be fitted before conversion")
        self.idata = az.from_cmdstanpy(self.fit)
        return self.idata

    def generate_plots(self, ground_truth: Dict[str, Any]) -> None:
        """Generate all plots for this model."""
        if self.idata is None:
            self.get_arviz()
        diagnose(self.fit, save_dir=self.analysis_dir)
        # Generate plots
        trace_plot(self.idata, save_dir=self.analysis_dir)
        pairs_plot(self.idata, save_dir=self.analysis_dir)
        posterior_plot(self.idata, ground_truth=ground_truth, save_dir=self.analysis_dir)
        plot_energy_vs_lp_and_params(self.idata, save_dir=self.analysis_dir)
        basic_summary(self.idata, save_dir=self.analysis_dir)
