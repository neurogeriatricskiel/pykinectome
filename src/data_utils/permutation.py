import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os


def permute(matrix1, matrix2, marker_list, task, matrix_type, kinematic, direction, result_base_path):
    
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
        "upper_body": ['head', 'ster', 'l_sho', 'r_sho', 'l_asis', 'r_asis', 'l_psis', 'r_psis', 
                    'l_elbl', 'l_wrist', 'l_hand', 'r_elbl', 'r_wrist', 'r_hand'],
        "lower_body": ['l_th', 'l_sk', 'l_ank', 'l_toe', 'r_th', 'r_sk', 'r_ank', 'r_toe']
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
    
    plot_permutation_histogram(rhos, true_rho, perm_p, result_base_path, task, kinematic, direction, matrix_type)

def plot_permutation_histogram(rhos, true_rho, perm_p, results_path, task, kinematic, direction, matrix_type):
    f,ax = plt.subplots()
    plt.hist(rhos,bins=20)
    ax.axvline(true_rho,  color = 'r', linestyle='--')
    ax.set(title=f"Permuted p: {perm_p:.3f}", ylabel="counts", xlabel="rho")
    os.chdir(Path(results_path, "avg_std_matrices"))
    plt.savefig(f'permutation_{task}_{kinematic}_{direction}_{matrix_type}.png', dpi=600)


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