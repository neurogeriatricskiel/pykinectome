import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os


def permute(matrix1, matrix2, marker_list, task, matrix_type, kinematic, direction, result_base_path, correlation_method):
    
    # Convert avg_group1 (numpy array) into a DataFrame
    df_group1 = pd.DataFrame(matrix1, index=marker_list, columns=marker_list)
    df_group2 = pd.DataFrame(matrix2, index=marker_list, columns=marker_list)
    
    # observed_diff = np.abs(df_group1) - np.abs(np.group2)

    # Now lets measure the similarity 
    rho, p_value = stats.spearmanr(upper(df_group1), upper(df_group2))
    print(f'rho = {np.round(rho, 2)} p_value = {p_value}  during {task} ({matrix_type}) for {kinematic} in {direction} direction')


    # """Nonparametric permutation testing Monte Carlo"""
    np.random.seed(0)
    rhos = []
    n_iter = 5000
    true_rho, _ = stats.spearmanr(upper(df_group1), upper(df_group2))

    # Define larger, slightly looser subgroups:
    subgroups = {
        "upper_body": ['head', 'ster', 'sho_la', 'sho_ma', 'asis_la', 'asis_ma', 'psis_la', 'psis_ma', 
                    'elbl_la', 'wrist_la', 'hand_la', 'elbl_ma', 'wrist_ma', 'hand_ma'],
        "lower_body": ['th_la', 'sk_la', 'ank_la', 'toe_la', 'th_ma', 'sk_ma', 'ank_ma', 'toe_ma']
    }

    m2_v = upper(df_group2)

    for _ in range(n_iter):
        shuffled_markers = marker_list.copy()

        # Shuffle **within** each broader subgroup
        for group in subgroups.values():
            shuffled_group = np.random.permutation(group)
            for original, shuffled in zip(group, shuffled_group):
                shuffled_markers[shuffled_markers.index(original)] = shuffled

        # Apply shuffled marker order
        shuffled_df1 = df_group1.loc[shuffled_markers, shuffled_markers]

        r, _ = stats.spearmanr(upper(shuffled_df1), m2_v)
        rhos.append(r)

    # Compute two-tailed p-value
    perm_p = ((np.sum(np.abs(true_rho) <= np.abs(rhos))) + 1) / (n_iter + 1)
    
    plot_permutation_histogram(rhos, true_rho, perm_p, result_base_path, task, kinematic, direction, matrix_type, correlation_method)


def permute_difference_matrix(matrix1, matrix2, group1, group2, marker_list, task, kinematic, direction, result_base_path, matrix_type = 'diff', n_permutations=5000):
    """
    Perform a permutation test on the difference matrix (matrix1 - matrix2)
    to determine whether the observed differences are significantly different from chance.

    Parameters:
    - matrix1, matrix2: (numpy arrays) Symmetric matrices to compare.
    - marker_list: (list) List of marker names corresponding to rows/columns.
    - task, matrix_type, kinematic, direction: (str) Metadata for reporting.
    - result_base_path: (str) Path to save the histogram.
    - n_iter: (int) Number of permutation iterations.

    Returns:
    - perm_p: (float) Two-tailed p-value for the permutation test.
    """
 
 # Convert matrices to DataFrames
    df_group1 = pd.DataFrame(matrix1, index=marker_list, columns=marker_list)
    df_group2 = pd.DataFrame(matrix2, index=marker_list, columns=marker_list)

    # Compute the **true** difference matrix and extract its upper triangle
    true_diff_matrix = df_group1 - df_group2
    true_diffs = upper(true_diff_matrix)

    # Permutation setup
    np.random.seed(0)
    perm_diffs = []
    n_iter = 5000

    # Define marker subgroups for within-group shuffling
    subgroups = {
        "upper_body": ['head', 'ster', 'l_sho', 'r_sho', 'l_asis', 'r_asis', 'l_psis', 'r_psis', 
                    'l_elbl', 'l_wrist', 'l_hand', 'r_elbl', 'r_wrist', 'r_hand'],
        "lower_body": ['l_th', 'l_sk', 'l_ank', 'l_toe', 'r_th', 'r_sk', 'r_ank', 'r_toe']
    }

    for _ in range(n_iter):
        shuffled_markers = marker_list.copy()  # Copy original marker order

        # Shuffle within subgroups
        for group in subgroups.values():
            shuffled_group = np.random.permutation(group)  # Shuffle the subgroup
            indices = [shuffled_markers.index(m) for m in group]  # Find original indices
            for idx, shuffled_m in zip(indices, shuffled_group):
                shuffled_markers[idx] = shuffled_m  # Replace in original list

        # Debugging: Check if markers are actually shuffled
        print(f"Iteration {_}: {shuffled_markers[:5]}")  # Print first few markers

        # Apply shuffled order
        shuffled_df1 = df_group1.loc[shuffled_markers, shuffled_markers]

        # Compute shuffled difference matrix
        shuffled_diff_matrix = upper(shuffled_df1 - df_group2)

        # Debugging: Check if shuffled matrices are changing
        print(f"Iteration {_}, Sum of shuffled_diff_matrix: {shuffled_diff_matrix.sum()}")

        perm_diffs.append(shuffled_diff_matrix)
        
        # perm_diff = true_diffs - shuffled_diff_matrix

        # perm_diffs.append(perm_diff)

    perm_p = (np.sum(np.mean(np.abs(true_diffs)) <= np.mean(np.abs(perm_diffs)), axis=0) + 1) / (n_permutations + 1)
    
    # Convert perm_diffs list into a NumPy array for vectorized operations
    perm_diffs = np.array(perm_diffs)  # Shape: (5000, num_elements)

    # Compute two-tailed p-values for each matrix element
    p_values = np.mean(np.abs(perm_diffs) >= np.abs(true_diffs), axis=0)

    # Optionally, compute an overall significance level (average across all elements)
    mean_p_value = np.mean(p_values)
    


    # Plot histogram
    plot_diff_permutation_histogram(perm_diffs, true_diffs, perm_p, result_base_path, task, kinematic, direction, matrix_type='diff')

def plot_diff_permutation_histogram(perm_diffs, true_diffs, perm_p, results_path, task, kinematic, direction, matrix_type='diff'):
    plt.figure(figsize=(8, 6))

    # Compute mean absolute difference across all upper-triangle elements for each permutation
    perm_dist = np.mean(np.abs(perm_diffs), axis=1)  

    # Compute the true mean absolute difference
    true_mean_diff = np.mean(np.abs(true_diffs))

    # Plot histogram
    plt.hist(perm_dist, bins=50, color='gray', alpha=0.7, edgecolor='black', label="Permutation Distribution")
    plt.axvline(true_mean_diff, color='red', linestyle='dashed', linewidth=2, label="True Difference")

    # Labels and legend
    plt.xlabel("Mean Absolute Difference")
    plt.ylabel("Frequency")
    plt.title("Permutation Test Distribution")
    plt.legend()

    os.chdir(Path(results_path, "avg_std_matrices"))

    plt.savefig(f'permutation_{task}_{kinematic}_{direction}_{matrix_type}.png', dpi=600)


def plot_permutation_histogram(rhos, true_rho, perm_p, results_path, task, kinematic, direction, matrix_type, correlation_method):
    f,ax = plt.subplots()
    plt.hist(rhos,bins=20)
    ax.axvline(true_rho,  color = 'r', linestyle='--')
    ax.set(title=f"Permuted matrix difference p: {perm_p:.3f}", ylabel="counts", xlabel="rho")

    if matrix_type == 'allegiance' or matrix_type == 'allegiance_std':
        os.chdir(Path(results_path, "allegiance_matrices"))
    else:
        os.chdir(Path(results_path, "avg_std_matrices"))

    plt.savefig(f'permutation_{task}_{kinematic}_{direction}_{matrix_type}_{correlation_method}.png', dpi=600)


def upper(df):
    '''Returns the upper triangle of a correlation matrix.
    You can use scipy.spatial.distance.squareform to recreate matrix from upper triangle.
    Args:
      df: pandas or numpy correlation matrix
    Returns:
      list of values from upper triangle
    '''
    try:
        assert(type(df)==np.ndarray)
    except:
        if type(df)==pd.DataFrame:
            df = df.values
        else:
            raise TypeError('Must be np.ndarray or pd.DataFrame')
    mask = np.triu_indices(df.shape[0], k=1)
    return df[mask]