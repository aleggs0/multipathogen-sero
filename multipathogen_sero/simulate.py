"""
Simulate infections and serosurveys for multiple pathogens.
This module provides functions to simulate infections using a Poisson process with thinning,
simulate infections from a survivor function, and create serosurveys from the simulation results.
"""

from typing import Callable, Dict, Union, List, Optional
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy.optimize import root_scalar

def simulate_infections(
        n_people: int,
        n_pathogens: int,
        foi_list: List[Callable[[Union[float, np.ndarray]], np.ndarray]],
        interaction_mat: Optional[np.ndarray],
        t_max: float = 100,
        birth_times: Optional[np.ndarray] = None,
        foi_max: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None
    ) -> pd.DataFrame:
    """
    Simulate infections using thinning of homogeneous Poisson process.
    Parameters:
        n_people (int): Number of individuals.
        n_pathogens (int): Number of pathogens.
        fois (Callable[[float], np.ndarray]): Function that returns the forces of infection at time t for each pathogen.
        interaction_mat (Optional[np.ndarray]): Interaction matrix for pathogens. The k,l entry indicates the effect of pathogen k on pathogen l.
        birth_times (Optional[np.ndarray]): Birth times for each individual. If None, individuals are assumed to be born at time 0.
        t_max (float): Maximum time for the simulation.
        foi_max (Optional[np.ndarray]): Maximum force of infection values for each pathogen.
        random_seed (Optional[int]): Random seed for reproducibility.
    """
    if birth_times is None:
        birth_times = np.zeros(n_people, dtype=float)
    if random_seed is not None:
        np.random.seed(random_seed)
    if interaction_mat is None:
        interaction_mat = np.ones((n_pathogens, n_pathogens))
    if foi_max is None:
        t_grid = np.linspace(0, t_max, 1000)
        foi_max = np.array([foi_list[k](t_grid).max() for k in range(n_pathogens)])
        foi_max *= 1.1 # leeway for discretization errors
    assert foi_max is not None  # for type checkers
    
    infection_times = []
    for i in range(n_people):
        birth_time = birth_times[i]
        infection_times.append((birth_time, 'birth', i, None))
        t_current = birth_time
        infection_status = np.zeros(n_pathogens, dtype=bool)  # Track infections for each pathogen
        susceptibility_factors = np.prod(interaction_mat[infection_status], axis=0)
        while True: #simulate until t_max reached or np.all(infection_status)
            proposal_indices = np.where(infection_status == 0)[0] # Indices of pathogens not yet infected
            proposal_times = np.full(len(proposal_indices), t_current, dtype=float) # Initialize proposal times for pathogens not yet infected
            earliest_accepted_proposal = np.inf
            earliest_accepted_proposal_index = None
            while proposal_indices.size > 0:
                # generate proposal times for each pathogen
                proposal_times += np.random.exponential(
                    1 / foi_max[proposal_indices] / susceptibility_factors[proposal_indices],
                    size=len(proposal_indices)
                )
                #thinning step
                accept_probs = np.array(
                    [foi_list[k](proposal_times[j]) / foi_max[k] for j, k in enumerate(proposal_indices)]
                )
                accept_mask = np.random.uniform(0, 1, size=len(proposal_indices)) < accept_probs
                #updates
                if np.any(accept_mask):
                    if proposal_times[accept_mask].min() < earliest_accepted_proposal:
                        # Update the earliest accepted proposal if a new one is found
                        accepted_proposal_indices = proposal_indices[accept_mask]
                        accepted_proposal_times = proposal_times[accept_mask]
                        earliest_accepted_proposal_index = accepted_proposal_indices[np.argmin(accepted_proposal_times)]
                        earliest_accepted_proposal = accepted_proposal_times.min()
                # continue generating for pathogens that still have a chance to infect earlier than the current earliest
                continue_mask = proposal_times < min(earliest_accepted_proposal, t_max)
                proposal_indices = proposal_indices[continue_mask]
                proposal_times = proposal_times[continue_mask]
            if earliest_accepted_proposal > t_max:
                break
            else:
                assert earliest_accepted_proposal_index is not None
            t_current = earliest_accepted_proposal
            k = earliest_accepted_proposal_index
            infection_times.append((t_current, 'seroconversion', i, k))
            infection_status[k] = 1
            if np.all(infection_status):
                break
            susceptibility_factors = np.prod(interaction_mat[infection_status], axis=0)
    return pd.DataFrame(infection_times, columns=['time', 'event', 'individual', 'pathogen'])

def simulate_infections_survivor(
        n_people: int,
        n_pathogens: int,
        survivor_list: List[Callable[[float],np.ndarray]],
        interaction_mat: Optional[np.ndarray] = None,
        time_precision: float = 0.0011,
        t_max: float = 100,
        birth_times: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None
    ) -> pd.DataFrame:
    """
    Simulate infections from a survivor function
    
    """
    if birth_times is None:
        birth_times = np.zeros(n_people, dtype=float)
    if random_seed is not None:
        np.random.seed(random_seed)
    if interaction_mat is None:
        interaction_mat = np.ones((n_pathogens, n_pathogens))
    infection_times = []
    max_survivors = np.array([survivor_list[k](t_max) for k in range(n_pathogens)])
    for i in range(n_people):
        birth_time = birth_times[i]
        infection_times.append((birth_time, 'birth', i, None))
        t_current = birth_time
        infection_status = np.zeros(n_pathogens, dtype=bool)  # Track infections for each pathogen
        while True:
            susceptibility_factors = np.prod(interaction_mat[infection_status], axis=0)
            proposal_indices = np.where(infection_status == 0)[0]  # Indices of pathogens not yet infected
            proposal_times = np.full(len(proposal_indices), np.inf)
            for j,k in enumerate(proposal_indices):
                current_survivor = survivor_list[k](t_current)
                proposal_quantile = np.random.uniform(0, 1)
                proposal_survivor = (proposal_quantile ** (1 / susceptibility_factors[k])) * current_survivor
                # proposal_time is such that S(t_prop=s_prop, where [s_prop/S(t_curr)]^susceptibility_factor = q_prop
                assert proposal_survivor < current_survivor
                if proposal_survivor <= max_survivors[k]:
                    proposal_times[j] = np.inf #the event doesn't happen before t_max
                else:
                    root_result = root_scalar(
                        lambda t: survivor_list[k](t) - proposal_survivor,
                        bracket = [t_current, t_max],
                        method='bisect',
                        xtol=time_precision
                    )
                    if not root_result.converged:
                        raise ValueError(
                            f"Root finding did not converge for pathogen {k} at individual {i}."
                        )
                    proposal_times[j] = root_result.root
            t_current = np.min(proposal_times)
            if t_current >= t_max:
                break
            k_min = proposal_indices[np.argmin(proposal_times)]
            infection_times.append((t_current, 'seroconversion', i, k_min))
            infection_status[k_min] = 1
            if np.all(infection_status):
                break
    return pd.DataFrame(infection_times, columns=['time', 'event', 'individual', 'pathogen'])


def simulation_to_regression_df(
        simulation_df: pd.DataFrame,
        k: Optional[int] = None
    ) -> pd.DataFrame:

    #  1. Rename columns
    regression_df = simulation_df.rename(
        columns={
            "time": "start_time",
            "event": "start_event",
            "pathogen": "start_event_pathogen"
        }
    )

    # 2. Sort for groupby operations
    regression_df = regression_df.sort_values(['individual', 'start_time'])

    # 3. Add stop_time, stop_event, stop_event_pathogen
    regression_df[['stop_time', 'stop_event', 'stop_event_pathogen']] = (
        regression_df.groupby('individual')[['start_time', 'start_event', 'start_event_pathogen']]
        .shift(-1)
    )
    regression_df['stop_time'] = regression_df['stop_time'].fillna(t_max)
    regression_df['stop_event'] = regression_df['stop_event'].fillna('censor')
    regression_df['stop_event_pathogen'] = regression_df['stop_event_pathogen']

    # 4. Add serostatus columns for each pathogen at start_time
    n_pathogens = int(simulate_infections_df['pathogen'].dropna().max()) + 1

    for k in range(n_pathogens):
        # For each row, check if there was a seroconversion for pathogen k before or at start_time
        sero_times = simulate_infections_df[
            (simulate_infections_df['event'] == 'seroconversion') &
            (simulate_infections_df['pathogen'] == k)
        ][['individual', 'time']]
        sero_times = sero_times.rename(columns={'time': f'seroconv_time_{k}'})
        # Merge to get seroconversion time for each individual
        regression_df = regression_df.merge(
            sero_times, on='individual', how='left'
        )
        regression_df[f'serostatus_{k}'] = (
            (regression_df[f'seroconv_time_{k}'].notna()) &
            (regression_df['start_time'] >= regression_df[f'seroconv_time_{k}'])
        ).astype(int)
        regression_df = regression_df.drop(columns=[f'seroconv_time_{k}'])

    # Optional: reset index if needed
    regression_df = regression_df.reset_index(drop=True)

    if k is not None:
        # Filter for intervals where individual is seronegative for pathogen k at start_time
        regression_df_for_pathogen_k = regression_df[regression_df[f'serostatus_{k}'] == 0].copy()

        # Add event column: 1 if seroconversion for k occurs at stop_event, else 0
        regression_df_for_pathogen_k['event'] = (
            (regression_df_for_pathogen_k['stop_event'] == 'seroconversion') &
            (regression_df_for_pathogen_k['stop_event_pathogen'] == k)
        ).astype(int)

        # Drop unnecessary columns
        regression_df_for_pathogen_k = regression_df_for_pathogen_k.drop(
            columns=[
                'start_event', 'start_event_pathogen', 'stop_event', 'stop_event_pathogen', f'serostatus_{k}'
            ]
        )
        return regression_df_for_pathogen_k
    else:
        return regression_df

def simulation_to_survey_long(
    simulation_df: pd.DataFrame,
    survey_times: Union[float, ArrayLike, Dict[int, ArrayLike]],
) -> pd.DataFrame:
    """
    Create a survey DataFrame from the simulation DataFrame.

    Parameters:
    simulation_df (pd.DataFrame): DataFrame containing the simulation results.
    survey_times: Can be a single number (all individuals surveyed at the same time),
              an array-like (survey time for each individual),
              or a dict mapping individual index to survey times.
    Returns:
    pd.DataFrame: Wide DataFrame with columns 'time', 'individual', and serostatus for each pathogen.
    """
    n_pathogens = simulation_df['pathogen'].dropna().astype(int).max() + 1
    individuals = simulation_df['individual'].unique()
    individuals.sort()
    # Normalize survey_times to a dict: individual -> survey_time
    if isinstance(survey_times, dict):
        survey_time_dict = survey_times
    elif np.isscalar(survey_times):
        survey_time_dict = {ind: [survey_times] for ind in individuals}
    else:
        survey_times_arr = np.asarray(survey_times)
        if survey_times_arr.shape[0] != len(individuals):
            raise ValueError("Length of survey_times does not match number of individuals.")
        survey_time_dict = {ind: [float(survey_times_arr[i])] for i, ind in enumerate(individuals)}
    # Prepare output
    records = []
    for ind in individuals:
        ind_survey_times = survey_time_dict[ind]
        ind_events = simulation_df[(simulation_df['individual'] == ind) & (simulation_df['event'] == 'seroconversion')]
        assert isinstance(ind_survey_times, (list, np.ndarray)), \
            f"Survey times for individual {ind} should be a list or array-like." #for typing
        for survey_time in ind_survey_times:
            for k in range(n_pathogens):
                serostatus = 0
                if not ind_events.empty:
                    seroconversion_times = ind_events[ind_events['pathogen'] == k]['time']
                    if len(seroconversion_times) > 1:
                        raise NotImplementedError(
                            f"Multiple seroconversion events for individual {ind} and pathogen {k}."
                        )
                    if not seroconversion_times.empty and seroconversion_times.iloc[0] <= survey_time:
                        serostatus = 1
                records.append((survey_time, ind, k, serostatus))
    survey_df = pd.DataFrame(records, columns=['time', 'individual', 'pathogen', 'serostatus'])
    return survey_df
                    
def survey_long_to_wide(
        survey_df: pd.DataFrame
    ) -> pd.DataFrame:
    """Convert a long-format survey DataFrame to wide format.
    The long format should have columns: 'time', 'individual', 'pathogen', 'serostatus'.
    The wide format will have 'time' and 'individual' as indices, and 'serostatus_k' for each pathogen k.
    """
    survey_df = survey_df.pivot_table(
        index=['time', 'individual'],
        columns='pathogen',
        values='serostatus'
    ).reset_index()
    # Flatten the MultiIndex columns
    survey_df.columns.name = None  # Remove the name of the columns index
    survey_df.columns = [f'serostatus_{col}' if isinstance(col, int) else col for col in survey_df.columns]
    return survey_df