import os
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from multipathogen_sero.io import save_metadata_json
from multipathogen_sero.config import STAN_DIR
# from multipathogen_sero.simulate import (
#     get_constant_foi, generate_uniform_birth_times,
#     simulate_infections_seroreversion, simulation_to_survey_wide
# )
from multipathogen_sero.analyse_chains import elpd_using_test_set, compare_using_test_set
from multipathogen_sero.models.model import PairwiseModel


def get_runtime_info():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "is_slurm_job": 'SLURM_JOB_ID' in os.environ,
        "job_name": os.environ.get('SLURM_JOB_NAME', 'local'),
        "array_index": int(os.environ.get('SLURM_ARRAY_TASK_ID', 1)),
        "hostname": os.environ.get('HOSTNAME', 'local'),
        "timestamp": timestamp,
        "job_id": os.environ.get('SLURM_ARRAY_JOB_ID', timestamp),
    }


class ExperimentRunner:
    def __init__(self, array_index: int = None, config_path: str = None):
        runtime_info = get_runtime_info()
        self.runtime_info = runtime_info

        self.array_index = array_index or runtime_info.get("array_index", 1)
        self.job_name = runtime_info.get("job_name", "local")
        self.job_id = runtime_info.get("job_id", 0)

        if config_path:
            with open(config_path, "r") as f:
                self.expt_settings = json.load(f)
        else:
            self.expt_settings = self._create_experiment_settings()
        self.output_dir = self._setup_output_directories()
        self.model_runners = {}
        
    def _create_experiment_settings(self) -> Dict[str, Any]:
        """Create experiment settings based on array index."""
        beta_mat, log_frailty_std = self._get_param_grid(self.array_index)
        
        return {
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
            "prior_config": {
                "baseline_hazard_scale": 1.0,
                "beta_scale": 1.0,
                "seroreversion_rate_scale": 1.0,
                "log_frailty_std_scale": 0.1,
                "log_frailty_std": log_frailty_std
            },
            "fit_config": {
                "n_frailty_samples": 20,
                "sampling_kwargs": {
                    "chains": 4,
                    "iter_sampling": 100,
                    "iter_warmup": 100,
                    "seed": 42
                }
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
            model_names = [
                "pairwise_serology_seroreversion_frailty",
                "pairwise_serology_seroreversion",
                "pairwise_serology_seroreversion_frailty_known"
            ]
            self.model_runners["model_name"] = PairwiseModel(
                stan_file_name=config["stan_file"],
                stan_dir=STAN_DIR,
                prior_config=config["prior_config"],
                fit_dir=self.output_dir / model_name,
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


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_experiment()
