import numpy as np

def ratio_of_improvement(sim_beqa, sim_baseline):
    """
    Computes the Ratio of Improvement (RI) for BEQA vs. a baseline approach.

    RI is defined as:
    
      RI = ( Σ Sim_BEQA(q_m) ) / ( Σ Sim_Baseline(q_m) )
      
    where the sums are taken over all queries m in {1, 2, ..., M}.
    
    Parameters
    ----------
    sim_beqa : list or np.ndarray
        A list/array of semantic similarity scores for queries using the BEQA approach.
    sim_baseline : list or np.ndarray
        A list/array of semantic similarity scores for queries using the baseline approach.
    
    Returns
    -------
    float
        The ratio of the sum of BEQA similarity scores to the sum of baseline similarity scores.
        A value > 1.0 indicates that BEQA outperforms the baseline on average.
    """
    sim_beqa = np.array(sim_beqa, dtype=float)
    sim_baseline = np.array(sim_baseline, dtype=float)
    
    # Safety checks
    if len(sim_beqa) == 0 or len(sim_baseline) == 0:
        raise ValueError("Input lists cannot be empty.")
    if len(sim_beqa) != len(sim_baseline):
        raise ValueError("The two lists must have the same length.")

    sum_beqa = np.sum(sim_beqa)
    sum_baseline = np.sum(sim_baseline)
    
    # Avoid division by zero in edge cases
    if sum_baseline == 0:
        raise ZeroDivisionError("The sum of baseline similarities is zero.")
    
    return sum_beqa / sum_baseline


# Example usage:
if __name__ == "__main__":
    # Suppose we have similarity scores for 5 queries
    sim_beqa_example = [0.82, 0.91, 0.78, 0.88, 0.85]
    sim_baseline_example = [0.80, 0.85, 0.74, 0.89, 0.80]
    
    ri_value = ratio_of_improvement(sim_beqa_example, sim_baseline_example)
    print(f"Ratio of Improvement (RI) = {ri_value:.3f}")
