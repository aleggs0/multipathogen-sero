from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    name: str
    stan_file: str
    var_names_trace: List[str]
    var_names_pairs: List[str]
    var_names_posterior: List[str]
    var_names_energy: List[str]
    
    @property
    def output_dir_name(self) -> str:
        return self.name.lower().replace(" ", "_")

# Define model configurations
MODEL_CONFIGS = {
    "frailty": ModelConfig(
        name="frailty",
        stan_file="pairwise_serology_seroreversion_frailty.stan",
        var_names_trace=["betas", "log_frailty_std", "baseline_hazards", "seroreversion_rates"],
        var_names_pairs=["betas", "log_frailty_std"],
        var_names_posterior=["betas", "log_frailty_std", "baseline_hazards", "seroreversion_rates"],
        var_names_energy=["betas", "log_frailty_std"]
    ),
    "no_frailty": ModelConfig(
        name="no_frailty",
        stan_file="pairwise_serology_seroreversion.stan",
        var_names_trace=["betas", "baseline_hazards", "seroreversion_rates"],
        var_names_pairs=["betas"],
        var_names_posterior=["betas", "baseline_hazards", "seroreversion_rates"],
        var_names_energy=["betas"]
    ),
    "frailty_known": ModelConfig(
        name="frailty_known",
        stan_file="pairwise_serology_seroreversion_frailty_known.stan",
        var_names_trace=["betas", "baseline_hazards", "seroreversion_rates"],
        var_names_pairs=["betas"],
        var_names_posterior=["betas", "baseline_hazards", "seroreversion_rates"],
        var_names_energy=["betas"]
    )
}

import time
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from cmdstanpy import CmdStanModel
import arviz as az

class ModelRunner:
    def __init__(self, config: ModelConfig, stan_dir: Path, output_base_dir: Path):
        self.config = config
        self.stan_dir = stan_dir
        self.output_dir = output_base_dir / self.config.output_dir_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.fit = None
        self.idata = None
        
    def compile_model(self) -> None:
        """Compile the Stan model."""
        stan_file_path = self.stan_dir / self.config.stan_file
        if not stan_file_path.exists():
            raise FileNotFoundError(f"Stan file not found: {stan_file_path}")
        
        self.model = CmdStanModel(stan_file=str(stan_file_path))
        
    def fit_model(self, stan_data: Dict[str, Any], **sample_kwargs) -> Tuple[float, Any]:
        """Fit the model and return fitting time and fit object."""
        if self.model is None:
            self.compile_model()
            
        start_time = time.time()
        self.fit = self.model.sample(data=stan_data, show_progress=False, **sample_kwargs)
        end_time = time.time()
        
        fitting_time = end_time - start_time
        print(f"Fitting time for {self.config.name}: {fitting_time:.2f} seconds")
        
        return fitting_time, self.fit
        
    def save_and_diagnose(self) -> str:
        """Save CSV files and run diagnostics."""
        from multipathogen_sero.analyse_chains import save_fit_diagnose
        
        if self.fit is None:
            raise ValueError("Model must be fitted before saving")
            
        self.fit.save_csvfiles(self.output_dir)
        return save_fit_diagnose(self.fit, self.output_dir)
        
    def convert_to_arviz(self) -> None:
        """Convert fit to ArviZ InferenceData."""
        if self.fit is None:
            raise ValueError("Model must be fitted before conversion")
            
        self.idata = az.from_cmdstanpy(self.fit)
        
    def generate_plots(self, ground_truth: Dict[str, Any]) -> None:
        """Generate all plots for this model."""
        from multipathogen_sero.analyse_chains import (
            trace_plot, pairs_plot, posterior_plot, 
            plot_energy_vs_lp_and_params, basic_summary
        )
        
        if self.idata is None:
            raise ValueError("Model must be converted to ArviZ before plotting")
            
        # Generate plots
        trace_plot(self.idata, var_names=self.config.var_names_trace, save_dir=self.output_dir)
        pairs_plot(self.idata, var_names=self.config.var_names_pairs, save_dir=self.output_dir)
        posterior_plot(self.idata, var_names=self.config.var_names_posterior, 
                      ground_truth=ground_truth, save_dir=self.output_dir)
        plot_energy_vs_lp_and_params(self.idata, var_names=self.config.var_names_energy, 
                                    save_dir=self.output_dir)
        basic_summary(self.idata, self.output_dir)

import os
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from multipathogen_sero.io import save_metadata_json
from multipathogen_sero.config import MODEL_FITS_DIR, STAN_DIR
from multipathogen_sero.simulate import (
    get_constant_foi, generate_uniform_birth_times,
    simulate_infections_seroreversion, simulation_to_survey_wide
)
from multipathogen_sero.analyse_chains import elpd_using_test_set, compare_using_test_set
from .model_config import MODEL_CONFIGS
from .model_runner import ModelRunner

class ExperimentRunner:
    def __init__(self, array_index: int = None):
        self.array_index = array_index or int(os.environ.get('SLURM_ARRAY_TASK_ID', 1))
        self.hostname = os.environ.get('HOSTNAME', 'local')
        self.timestamp = int(time.time())
        self.job_id = int(os.environ.get('SLURM_ARRAY_JOB_ID', self.timestamp))
        self.job_name = os.environ.get('SLURM_JOB_NAME', 'local')
        
        self.expt_settings = self._create_experiment_settings()
        self.output_dir = self._setup_output_directories()
        self.model_runners = {}
        
    def _create_experiment_settings(self) -> Dict[str, Any]:
        """Create experiment settings based on array index."""
        beta_mat, log_frailty_std = self._get_param_grid(self.array_index)
        
        return {
            "runtime_info": {
                "job_id": self.job_id,
                "array_index": self.array_index,
                "hostname": self.hostname,
                "timestamp": self.timestamp
            },
            "ground_truth_params": {
                "n_pathogens": 2,
                "baseline_hazards": [0.05, 0.10],
                "seroreversion_rates": [0.05, 0.02],
                "log_frailty_std": log_frailty_std,
                "beta_mat": beta_mat,
                "seed": 42
            },
            "train_data": {
                "n_people": 1200,
                "t_min": 0,
                "t_max": 100,
                "survey_every": 10,
                "seed": 42 + self.array_index
            },
            "test_data": {
                "n_people": 1200,
                "t_min": 0,
                "t_max": 100,
                "survey_every": 10,
                "seed": 2411 + self.array_index
            },
            "inference_params": {
                "baseline_hazard_scale": 1.0,
                "beta_scale": 1.0,
                "seroreversion_rate_scale": 1.0,
                "log_frailty_std_scale": 0.1,
                "log_frailty_std": log_frailty_std,
                "n_frailty_samples": 20,
                "chains": 4,
                "iter_sampling": 100,
                "iter_warmup": 100,
                "seed": 42
            },
            "notes": ""
        }
        
    def _get_param_grid(self, array_index: int) -> tuple:
        """Get parameters for given array index."""
        beta_mats = [
            [[0, 0], [0, 0]],
            [[0, 0.5], [0.5, 0]]
        ]
        log_frailty_stds = [0.3, 1.0]
        
        n_beta = len(beta_mats)
        n_frailty = len(log_frailty_stds)
        total = n_beta * n_frailty
        
        idx = (array_index - 1) % total
        beta_idx = idx // n_frailty
        frailty_idx = idx % n_frailty
        
        return beta_mats[beta_idx], log_frailty_stds[frailty_idx]
        
    def _setup_output_directories(self) -> Path:
        """Setup output directories."""
        output_dir = MODEL_FITS_DIR / f"{self.job_name}_j{self.job_id}" / f"a{self.array_index}"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_metadata_json(output_dir, self.expt_settings)
        return output_dir
        
    def simulate_data(self) -> tuple:
        """Simulate training and test data."""
        # Implementation of data simulation (extracted from original code)
        # ... (same as before)
        pass
        
    def setup_models(self, model_names: List[str] = None) -> None:
        """Setup model runners for specified models."""
        if model_names is None:
            model_names = list(MODEL_CONFIGS.keys())
            
        for model_name in model_names:
            if model_name not in MODEL_CONFIGS:
                raise ValueError(f"Unknown model: {model_name}")
                
            config = MODEL_CONFIGS[model_name]
            self.model_runners[model_name] = ModelRunner(
                config=config,
                stan_dir=STAN_DIR,
                output_base_dir=self.output_dir
            )
            
    def fit_all_models(self, stan_data: Dict[str, Any]) -> None:
        """Fit all configured models."""
        sample_kwargs = {
            "chains": self.expt_settings["inference_params"]["chains"],
            "iter_sampling": self.expt_settings["inference_params"]["iter_sampling"],
            "iter_warmup": self.expt_settings["inference_params"]["iter_warmup"],
            "parallel_chains": self.expt_settings["inference_params"]["chains"],
            "seed": self.expt_settings["inference_params"]["seed"],
        }
        
        for name, runner in self.model_runners.items():
            print(f"Fitting {name} model...")
            fitting_time, fit = runner.fit_model(stan_data, **sample_kwargs)
            diagnose_result = runner.save_and_diagnose()
            print(diagnose_result)
            runner.convert_to_arviz()
            
    def generate_all_plots(self) -> None:
        """Generate plots for all models."""
        ground_truth_betas = [
            self.expt_settings["ground_truth_params"]["beta_mat"][0][1],
            self.expt_settings["ground_truth_params"]["beta_mat"][1][0]
        ]
        
        base_ground_truth = {
            "betas": ground_truth_betas,
            "baseline_hazards": self.expt_settings["ground_truth_params"]["baseline_hazards"],
            "seroreversion_rates": self.expt_settings["ground_truth_params"]["seroreversion_rates"]
        }
        
        for name, runner in self.model_runners.items():
            ground_truth = base_ground_truth.copy()
            if name == "frailty":
                ground_truth["log_frailty_std"] = self.expt_settings["ground_truth_params"]["log_frailty_std"]
                
            runner.generate_plots(ground_truth)
            
    def run_elpd_analysis(self) -> None:
        """Run ELPD analysis and comparisons."""
        elpd_results = {}
        
        # Calculate ELPD for each model
        for name, runner in self.model_runners.items():
            elpd, se_elpd, _ = elpd_using_test_set(runner.idata)
            elpd_results[name] = {"elpd": elpd, "se": se_elpd}
            
        # Calculate pairwise comparisons
        comparisons = {}
        model_names = list(self.model_runners.keys())
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                runner1 = self.model_runners[name1]
                runner2 = self.model_runners[name2]
                diff, se_diff = compare_using_test_set(runner1.idata, runner2.idata)
                comparisons[f"{name1}_vs_{name2}"] = {"diff": diff, "se": se_diff}
                
        # Generate report
        self._generate_elpd_report(elpd_results, comparisons)
        
    def _generate_elpd_report(self, elpd_results: Dict, comparisons: Dict) -> None:
        """Generate ELPD report."""
        report_lines = []
        
        for name, result in elpd_results.items():
            report_lines.append(f"elpd ({name} model): {result['elpd']} (SE: {result['se']})")
            
        report_lines.append("")
        
        for comp_name, result in comparisons.items():
            report_lines.append(f"elpd difference ({comp_name}): {result['diff']} (SE: {result['se']})")
            
        elpd_report = "\n".join(report_lines)
        
        with open(self.output_dir / "elpd_report.txt", "w") as f:
            f.write(elpd_report)
            
    def run_experiment(self, model_names: List[str] = None) -> None:
        """Run the complete experiment."""
        print("Simulating data...")
        stan_data = self.simulate_data()
        
        print("Setting up models...")
        self.setup_models(model_names)
        
        print("Fitting models...")
        self.fit_all_models(stan_data)
        
        print("Generating plots...")
        self.generate_all_plots()
        
        print("Running ELPD analysis...")
        self.run_elpd_analysis()
        
        print("Experiment completed!")


from experiment_runner import ExperimentRunner

def main():
    # Run experiment with all models
    runner = ExperimentRunner()
    runner.run_experiment()
    
    # Or run with specific models only
    # runner.run_experiment(model_names=["frailty", "no_frailty"])

if __name__ == "__main__":
    main()