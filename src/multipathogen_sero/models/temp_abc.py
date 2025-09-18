import time
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from cmdstanpy import CmdStanModel
import arviz as az
from multipathogen_sero.analyse_chains import (
    diagnose, trace_plot, pairs_plot, posterior_plot,
    plot_energy_vs_lp_and_params, basic_summary, read_fit_csv_dir
)


class BaseModelRunner(ABC):
    def __init__(self, stan_dir: Path, fit_dir: Path, prior_config: Dict[str, Any]):
        """
        Initialize base ModelRunner.
        
        Args:
            stan_dir: Directory containing Stan files
            fit_dir: Directory for output files
            prior_config: Dictionary containing fixed parameters for the stan data
        """
        self.stan_dir = stan_dir
        self.output_dir = fit_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store prior configuration
        self.prior_config = prior_config.copy()
        
        # Model state
        self.model = None
        self.fit = None
        self.idata = None
        
    @property
    @abstractmethod
    def stan_file_name(self) -> str:
        """Name of the Stan file for this model."""
        pass
    
    @property
    @abstractmethod
    def required_data_fields(self) -> List[str]:
        """List of required data fields for this model."""
        pass
    
    @property
    @abstractmethod
    def plot_var_names(self) -> Dict[str, List[str]]:
        """Variable names for different plot types."""
        pass
    
    def compile_model(self) -> None:
        """Compile the Stan model."""
        stan_file_path = self.stan_dir / self.stan_file_name
        if not stan_file_path.exists():
            raise FileNotFoundError(f"Stan file not found: {stan_file_path}")
        self.model = CmdStanModel(stan_file=str(stan_file_path))
        
    def _validate_stan_data(self, stan_data: Dict[str, Any]) -> None:
        """Validate that required data is present."""
        missing = [field for field in self.required_data_fields if field not in stan_data]
        if missing:
            raise ValueError(f"Missing required data fields for {self.stan_file_name}: {missing}")
    
    def _combine_data(self, observation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine prior configuration with observation data."""
        stan_data = self.prior_config.copy()
        stan_data.update(observation_data)
        return stan_data
        
    def fit_model(self, observation_data: Dict[str, Any], **sample_kwargs) -> Tuple[float, Any]:
        """Fit the model and return fitting time and fit object."""
        if self.model is None:
            self.compile_model()
        
        # Combine and validate data
        stan_data = self._combine_data(observation_data)
        self._validate_stan_data(stan_data)
            
        start_time = time.time()
        self.fit = self.model.sample(
            data=stan_data,
            show_progress=False,
            **sample_kwargs
        )
        end_time = time.time()
        
        fitting_time = end_time - start_time
        print(f"Fitting time for {self.stan_file_name}: {fitting_time:.2f} seconds")
        
        return fitting_time, self.fit
    
    def load_fit(self, source_fit_dir: Optional[Path] = None):
        """Load a previously saved fit from CSV files."""
        if source_fit_dir is None:
            source_fit_dir = self.output_dir
        self.fit = read_fit_csv_dir(source_fit_dir)
        return self.fit
        
    def save_fit(self) -> None:
        if self.fit is None:
            raise ValueError("Model must be fitted before saving")
        self.fit.save_csvfiles(self.output_dir)
        
    def get_arviz(self) -> az.InferenceData:
        """Convert fit to ArviZ InferenceData."""
        if self.fit is None:
            raise ValueError("Model must be fitted before conversion")
        self.idata = az.from_cmdstanpy(self.fit)
        return self.idata
        
    def generate_plots(self, ground_truth: Dict[str, Any]) -> None:
        """Generate all plots for this model."""
        if self.idata is None:
            self.get_arviz()
        
        diagnose(self.fit, self.output_dir)
        
        # Use model-specific variable names
        var_names = self.plot_var_names
        trace_plot(self.idata, var_names=var_names.get("trace", []), save_dir=self.output_dir)
        pairs_plot(self.idata, var_names=var_names.get("pairs", []), save_dir=self.output_dir)
        posterior_plot(self.idata, var_names=var_names.get("posterior", []), 
                      ground_truth=ground_truth, save_dir=self.output_dir)
        plot_energy_vs_lp_and_params(self.idata, var_names=var_names.get("energy", []), 
                                    save_dir=self.output_dir)
        basic_summary(self.idata, self.output_dir)

from .model import BaseModelRunner

class FrailtyModelRunner(BaseModelRunner):
    @property
    def stan_file_name(self) -> str:
        return "pairwise_serology_seroreversion_frailty.stan"
    
    @property
    def required_data_fields(self) -> List[str]:
        return [
            "n_people", "n_pathogens", "n_surveys", "survey_times", 
            "serology_data", "baseline_hazard_scale", "beta_scale",
            "seroreversion_rate_scale", "log_frailty_std_scale", 
            "log_frailty_std", "n_frailty_samples"
        ]
    
    @property
    def plot_var_names(self) -> Dict[str, List[str]]:
        return {
            "trace": ["betas", "log_frailty_std", "baseline_hazards", "seroreversion_rates"],
            "pairs": ["betas", "log_frailty_std"],
            "posterior": ["betas", "log_frailty_std", "baseline_hazards", "seroreversion_rates"],
            "energy": ["betas", "log_frailty_std"]
        }


class NoFrailtyModelRunner(BaseModelRunner):
    @property
    def stan_file_name(self) -> str:
        return "pairwise_serology_seroreversion.stan"
    
    @property
    def required_data_fields(self) -> List[str]:
        return [
            "n_people", "n_pathogens", "n_surveys", "survey_times",
            "serology_data", "baseline_hazard_scale", "beta_scale", 
            "seroreversion_rate_scale"
        ]
    
    @property
    def plot_var_names(self) -> Dict[str, List[str]]:
        return {
            "trace": ["betas", "baseline_hazards", "seroreversion_rates"],
            "pairs": ["betas"],
            "posterior": ["betas", "baseline_hazards", "seroreversion_rates"],
            "energy": ["betas"]
        }


class FrailtyKnownModelRunner(BaseModelRunner):
    @property
    def stan_file_name(self) -> str:
        return "pairwise_serology_seroreversion_frailty_known.stan"
    
    @property
    def required_data_fields(self) -> List[str]:
        return [
            "n_people", "n_pathogens", "n_surveys", "survey_times",
            "serology_data", "baseline_hazard_scale", "beta_scale", 
            "seroreversion_rate_scale", "log_frailty_std"
        ]
    
    @property
    def plot_var_names(self) -> Dict[str, List[str]]:
        return {
            "trace": ["betas", "baseline_hazards", "seroreversion_rates"],
            "pairs": ["betas"],
            "posterior": ["betas", "baseline_hazards", "seroreversion_rates"],
            "energy": ["betas"]
        }