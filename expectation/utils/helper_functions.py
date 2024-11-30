import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from typing import List, Dict
import pandas as pd

def plot_sequential_test(history_df: pd.DataFrame, alpha: float = 0.05, figsize: tuple = (15, 10), log=None):
    """
    Plot the evolution of a sequential test.
    
    Parameters:
    -----------
    history_df : pd.DataFrame
        DataFrame containing test history with columns:
        - step: test step number
        - eValue: individual e-values
        - cumulativeEValue: cumulative e-values
        - observations: observed values
    alpha : float
        Significance level (default 0.05)
    figsize : tuple
        Figure size (width, height)
    """
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=figsize)
    
    # Plot 1: Individual e-values
    ax1.plot(history_df['step'], history_df['eValue'], 
            marker='o', linestyle='-', color='blue', label='E-value')
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Individual E-values')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('E-value')
    if log:
        ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Cumulative e-values (e-process)
    ax2.plot(history_df['step'], history_df['cumulativeEValue'], 
            marker='o', linestyle='-', color='green', label='Cumulative E-value')
    ax2.axhline(y=1/alpha, color='red', linestyle='--', 
                alpha=0.5, label=f'Rejection Boundary (1/α = {1/alpha})')
    ax2.set_title('Cumulative E-values (E-Process)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative E-value')
    if log:
        ax2.set_yscale('log')
        
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # epower
    ax3.plot(history_df['step'], 100 * history_df["ePower"].fillna(0), 
            marker='o', linestyle='-', color='green', label='E-power')
    ax3.set_title('E-power (growth rate)')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('E-power')
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())

    # pvalues
    ax4.plot(history_df['step'], 100 * history_df["pValue"], 
            marker='o', linestyle='-', color='purple', label='p-value')
    ax4.axhline(y=alpha * 100, color='red', linestyle='--', 
                alpha=0.5, label=f'Rejection Boundary (α = {alpha})')
    ax4.set_title('p-values (converted from e-values)')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('p-value')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-10, 100)
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # Plot 3: Raw observations
    observations = np.concatenate(history_df['observations'].values)
    steps = np.repeat(history_df['step'], 
                     history_df['observations'].apply(len))
    
    ax5.scatter(steps, observations, color='orange', alpha=0.6, label='Observations')
    ax5.set_title('Raw Observations')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Value')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3, ax4, ax5)

# Example usage
if __name__ == "__main__":
    # Create test
    from sequential_test import SequentialTest, TestType
    
    test = SequentialTest(
        test_type=TestType.MEAN,
        null_value=0,
        alternative="greater"
    )
    
    # Generate some data
    np.random.seed(42)
    data_batches = [
        np.random.normal(0.5, 1, 3),  # Batch 1
        np.random.normal(0.5, 1, 2),  # Batch 2
        np.random.normal(0.5, 1, 3),  # Batch 3
    ]
    
    # Run test
    for batch in data_batches:
        result = test.update(batch)
        print(f"Batch mean: {np.mean(batch):.2f}")
        print(f"E-value: {result.e_value:.2f}")
        print(f"Cumulative: {result.e_process.cumulative_value:.2f}")
        print(f"Reject H0: {result.reject_null}\n")
    
    # Plot results
    history_df = test.get_history_df()
    fig, axes = plot_sequential_test(history_df)
    plt.show()