import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import Table2x2


def independence_test(serostatus_1, serostatus_2):
    """
    Perform an independence test between two serostatus arrays.
    Args:
        serostatus_1 (array-like): First serostatus array (e.g., for pathogen 1).
        serostatus_2 (array-like): Second serostatus array (e.g., for pathogen 2).
    Returns:
        tuple: p-value, observed proportion, expected proportion.
    """
    if len(serostatus_1) != len(serostatus_2):
        raise ValueError("Serostatus arrays must have the same length.")
    if len(serostatus_1) == 0:
        return np.nan
    
    # Observed proportion
    observed_proportion = (
        np.dot(serostatus_1, serostatus_2) / len(serostatus_1)
    )
    
    # Expected proportion
    expected_proportion = serostatus_1.sum() * serostatus_2.sum() / len(serostatus_1)**2  # Assuming independence
    
    # Calculate p-value (e.g., using a chi-squared test)
    contingency_table = pd.crosstab(
        serostatus_1, serostatus_2
    )
    if contingency_table.shape == (2, 2):
        tbl = Table2x2(contingency_table)
        p_value = tbl.test_nominal_association().pvalue
    else:
        p_value = np.nan
    return p_value, observed_proportion, expected_proportion


def significance(p_value):
    if p_value >= 0 and p_value <= 1:
        if p_value <= 0.001:
            asterisks = '***'
        elif p_value <= 0.01:
            asterisks = '**'
        elif p_value <= 0.05:
            asterisks = '*'
        else:
            asterisks = ''
        return asterisks
    else:
        raise ValueError("p_value must be a number between 0 and 1.")