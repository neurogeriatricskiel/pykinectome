from src.data_utils.data_loader import load_kinectomes
from src.data_utils import plotting, groups, permutation
from src.graph_utils import kinectome2pattern
from src.modularity import build_graph
from src.kinectome_characteristics import calc_std_avg_matrices
import numpy as np
from collections import defaultdict
import networkx as nx
import pandas as pd
from pathlib import Path
import pickle
import os
from statsmodels.stats.multitest import multipletests

def get_pattern_for_subject(all_kinectomes, marker_list, full, pattern_length, start_node):
    """
       Parameters:
    -----------
    all_kinectomes : dict
        The nested dictionary containing kinectome data
    marker_list : list
        List of markers/nodes for the graph  
    full : bool
        Whether to use full kinectome or combined kinectome
    start_node : str
        Starting node for the pattern
    pattern_length : int
        Length of the pattern subgraph  

    Returns:
    --------
    subject_patterns : dict
        Dictionary of subject-specific patterns
 
    """

    # Dictionary to store subject-specific patterns
    subject_patterns = defaultdict(lambda: defaultdict(list))

    for group in all_kinectomes.keys():
        for sub_id in all_kinectomes[group].keys():
            for task in all_kinectomes[group][sub_id].keys():
                for kinematics in all_kinectomes[group][sub_id][task].keys():
                    directions_list = list(all_kinectomes[group][sub_id][task][kinematics].keys())

                    if 'AP' in directions_list and 'ML' in directions_list and 'V' in directions_list and all_kinectomes[group][sub_id][task][kinematics]['AP'] is not None:
                        # Extract the avg arrays for each direction
                        ap_kinectome = all_kinectomes[group][sub_id][task][kinematics]['AP']['avg']
                        ml_kinectome = all_kinectomes[group][sub_id][task][kinematics]['ML']['avg']
                        v_kinectome = all_kinectomes[group][sub_id][task][kinematics]['V']['avg']

                        # Stack the matrices along a new axis to create a 22x22x3 array
                        # The third dimension will have AP at index 0, ML at index 1, and V at index 2
                        combined_kinectome = np.stack([ap_kinectome, ml_kinectome, v_kinectome], axis=2)
                    
                    elif 'full' in directions_list and all_kinectomes[group][sub_id][task][kinematics]['full'] is not None:
                        full_kinectome = all_kinectomes[group][sub_id][task][kinematics]['full']['avg']

                    graphs = build_graph(full_kinectome if full else combined_kinectome, marker_list)

                    for idx, graph in enumerate(graphs):
                        direction = directions_list[idx]
                        max_pattern_graph = kinectome2pattern.strongest_pattern_subgraph(graph, pattern_length, start_node)
                        
                        # Store the pattern edges with their weights
                        pattern_edges = list(max_pattern_graph.edges(data=True))
                        
                        # Store in subject_patterns dictionary
                        subject_patterns[group][f"{sub_id}_{task}_{kinematics}_{direction}"].append({
                            'edges': pattern_edges,
                            'nodes': list(max_pattern_graph.nodes()),
                            'graph': max_pattern_graph
                        })

    return subject_patterns     


def get_avg_group_patterns(subject_patterns, pattern_length, start_node):

    """
    """
    
    group_patterns = {}

    # Average patterns within each group
    for group, subjects in subject_patterns.items():
        # Collect all unique edges across subjects in this group
        all_edges = set()
        for subject_data in subjects.values():
            for pattern in subject_data:
                all_edges.update([(u, v) for u, v, _ in pattern['edges']])
        
        # Dictionary to store weight sums and counts for averaging
        edge_weights = {edge: {'sum': 0.0, 'count': 0} for edge in all_edges}
        
        # Sum up weights for each edge
        for subject_data in subjects.values():
            for pattern in subject_data:
                for u, v, data in pattern['edges']:
                    if (u, v) in edge_weights:
                        edge_weights[(u, v)]['sum'] += data['weight']
                        edge_weights[(u, v)]['count'] += 1
        
        # Calculate average weights
        avg_edges = {}
        for edge, data in edge_weights.items():
            if data['count'] > 0:
                avg_edges[edge] = data['sum'] / data['count']
        
        # Create a graph with the averaged edges
        avg_graph = nx.DiGraph()
        for (u, v), weight in avg_edges.items():
            avg_graph.add_edge(u, v, weight=weight)
        
        # Find minimum pattern in averaged graph
        # Note: This might need adjustment if the averaged graph doesn't contain the start_node
        if start_node in avg_graph.nodes():
            group_pattern_graph = kinectome2pattern.strongest_pattern_subgraph(avg_graph, length=pattern_length, start_node=start_node)
            group_patterns[group] = {
                'edges': list(group_pattern_graph.edges(data=True)),
                'nodes': list(group_pattern_graph.nodes()),
                'graph': group_pattern_graph
            }
        else:
            print(f"Warning: Start node '{start_node}' not found in averaged graph for group {group}")
            # Choose an alternative start node if needed
            if avg_graph.nodes():
                alt_start = list(avg_graph.nodes())[0]
                group_pattern_graph = kinectome2pattern.strongest_pattern_subgraph(avg_graph, length=pattern_length, start_node=alt_start)
                group_patterns[group] = {
                    'edges': list(group_pattern_graph.edges(data=True)),
                    'nodes': list(group_pattern_graph.nodes()),
                    'graph': group_pattern_graph
                }
            else:
                print(f"No nodes found in averaged graph for group {group}")
    
    return group_patterns


def get_pattern_values_for_subjects(all_kinectomes, group_patterns, full, marker_list, result_base_path, pattern_length, start_node, save_csv):
    """
    Calculate path values for each subject using group-specific patterns.
    Returns individual edge weights, sum, and average rather than product.
    """
    pattern_values_data = []
    
    # For each group pattern, evaluate it on all subjects from all groups
    for pattern_group, group_pattern in group_patterns.items():
        group_pattern_edges = [(u, v) for u, v, _ in group_pattern['edges']]
        pattern_name = "_".join(group_pattern['nodes'])
        
        # Now check this pattern in all groups (including the pattern group itself)
        for subject_group in all_kinectomes.keys():
            for sub_id in all_kinectomes[subject_group].keys():
                for task in all_kinectomes[subject_group][sub_id].keys():
                    for kinematics in all_kinectomes[subject_group][sub_id][task].keys():
                        directions_list = list(all_kinectomes[subject_group][sub_id][task][kinematics].keys())
                        
                        # Get kinectome data based on available directions
                        kinectome_data = None
                        if 'AP' in directions_list and 'ML' in directions_list and 'V' in directions_list and all_kinectomes[subject_group][sub_id][task][kinematics]['AP'] is not None:

                            # Extract the avg arrays for each direction
                            ap_kinectome = all_kinectomes[subject_group][sub_id][task][kinematics]['AP']['avg']
                            ml_kinectome = all_kinectomes[subject_group][sub_id][task][kinematics]['ML']['avg']
                            v_kinectome = all_kinectomes[subject_group][sub_id][task][kinematics]['V']['avg']
                            # Stack the matrices along a new axis to create a 22x22x3 array
                            kinectome_data = np.stack([ap_kinectome, ml_kinectome, v_kinectome], axis=2)
                        elif 'full' in directions_list and full and all_kinectomes[subject_group][sub_id][task][kinematics]['full'] is not None:
                            kinectome_data = all_kinectomes[subject_group][sub_id][task][kinematics]['full']['avg']
                        
                        if kinectome_data is None:
                            continue
                        
                        # Build graph from kinectome data
                        graphs = build_graph(kinectome_data, marker_list)
                        
                        # Process each graph (direction)
                        for idx, graph in enumerate(graphs):
                            # Determine direction name
                            if kinectome_data.ndim == 3 and kinectome_data.shape[2] == 3:
                                direction = ['AP', 'ML', 'V'][idx]
                            elif 'full' in directions_list:
                                direction = 'full'
                            else:
                                direction = f"direction_{idx}"
                            
                            # Extract edge weights for the pattern
                            edge_weights = []
                            missing_edges = []
                            edge_weights_with_nodes = []  # Store (weight, (u, v)) tuples
                            
                            for u, v in group_pattern_edges:
                                if graph.has_edge(u, v):
                                    weight = graph[u][v]['weight']
                                    edge_weights.append(weight)
                                    edge_weights_with_nodes.append((weight, (u, v)))
                                else:
                                    missing_edges.append((u, v))
                            
                            # Calculate different metrics for the path
                            if len(missing_edges) == 0:  # All edges exist
                                path_sum = sum(edge_weights)
                                # path_product = np.prod(edge_weights)  # Keep this if you want to compare
                                path_min = min(edge_weights)
                                path_max = max(edge_weights)

                                # Find weakest link (minimum weight edge)
                                min_weight, weakest_edge = min(edge_weights_with_nodes, key=lambda x: x[0])
                                weakest_link = f"{weakest_edge[0]}-{weakest_edge[1]}"

                                valid_pattern = True
                            else:
                                path_sum = np.nan
                                path_product = np.nan
                                path_min = np.nan
                                path_max = np.nan
                                weakest_link = None
                                valid_pattern = False
                            
                            # Add the pattern values for this subject
                            pattern_values_data.append({
                                'Pattern_Group': pattern_group,  # Group where pattern was found
                                'Subject_Group': subject_group,  # Group of the subject
                                'Subject': sub_id,
                                'Task': task,
                                'Kinematics': kinematics,
                                'Direction': direction,
                                'Pattern': pattern_name,
                                'Path_Sum': path_sum,
                                # 'Path_Product': path_product,  # Original calculation for reference
                                # 'Path_Min': path_min,
                                # 'Path_Max': path_max,
                                # 'Valid_Pattern': valid_pattern,
                                # 'Num_Edges': len(group_pattern_edges),
                                # 'Missing_Edges': len(missing_edges),
                                'Weakest_Link': weakest_link,
                                # 'Edge_Weights': edge_weights if valid_pattern else None
                            })
    
    # Create DataFrame from the pattern values data
    pattern_values_df = pd.DataFrame(pattern_values_data)
    
    # Save the pattern values to CSV
    csv_filename = f"path_values_length_{pattern_length}_start_{start_node}.csv"

    # Define the save path
    save_path = Path(result_base_path, 'patterns')

    # Check if the 'patterns' directory exists, if not create it
    save_path.mkdir(parents=True, exist_ok=True)

    # Save the csv file
    if save_csv:
        pattern_values_df.to_csv(save_path / csv_filename, index=False)
        
    print(f"Path values saved to {csv_filename}")
    print(f"Shape of results: {pattern_values_df.shape}")
    
    # # Print summary statistics
    # if not pattern_values_df.empty:
    #     print("\nSummary statistics for path averages:")
    #     for pattern_group in pattern_values_df['Pattern_Group'].unique():
    #         for subject_group in pattern_values_df['Subject_Group'].unique():
    #             subset = pattern_values_df[
    #                 (pattern_values_df['Pattern_Group'] == pattern_group) & 
    #                 (pattern_values_df['Subject_Group'] == subject_group)
    #             ]
    #             if not subset.empty:
    #                 avg_values = subset['Path_Average'].dropna()
    #                 if len(avg_values) > 0:
    #                     print(f"  {pattern_group} pattern in {subject_group} subjects: "
    #                           f"mean={avg_values.mean():.3f}, std={avg_values.std():.3f}, n={len(avg_values)}")
    
    return pattern_values_df


def compare_groups_statistical(pattern_values_df, pattern_group, 
                              subject_group1, subject_group2, kinematics=None, direction=None, task=None):
    """
    Perform statistical comparison between two subject groups for a specific pattern.
    
    Parameters:
    -----------
    pattern_values_df : pd.DataFrame
        DataFrame containing pattern values
    pattern_group : str
        The pattern group to analyze (e.g., 'Parkinson' - the group where pattern was identified)
    subject_group1 : str
        First subject group to compare (e.g., 'Parkinson')
    subject_group2 : str
        Second subject group to compare (e.g., 'Control')
    kinematics : str, optional
        Specific kinematics/speed to filter by (e.g., 'acc', 'preferred')
    direction : str, optional
        Specific direction to filter by (e.g., 'AP', 'ML', 'V')
    """
    from scipy import stats
    import numpy as np
    
    # Filter data for the first subject group with the specified pattern
    mask1 = (pattern_values_df['Pattern_Group'] == pattern_group) & \
            (pattern_values_df['Subject_Group'] == subject_group1)
    
    group1_wekest_link = pattern_values_df[mask1]['Weakest_Link'].value_counts()
    weakest_link_name1 = group1_wekest_link.index[0]

    if kinematics is not None:
        mask1 = mask1 & (pattern_values_df['Kinematics'] == kinematics)
    if direction is not None:
        mask1 = mask1 & (pattern_values_df['Direction'] == direction)
    if task is not None:
        mask1 = mask1 & (pattern_values_df['Task'] == task)
    
    group1_data = pattern_values_df[mask1]['Path_Sum'].dropna()
    
    # Filter data for the second subject group with the same pattern
    mask2 = (pattern_values_df['Pattern_Group'] == pattern_group) & \
            (pattern_values_df['Subject_Group'] == subject_group2)
    
    group2_wekest_link = pattern_values_df[mask2]['Weakest_Link'].value_counts()
    weakest_link_name2 = group2_wekest_link.index[0]

    if kinematics is not None:
        mask2 = mask2 & (pattern_values_df['Kinematics'] == kinematics)
    if direction is not None:
        mask2 = mask2 & (pattern_values_df['Direction'] == direction)
    if task is not None:
        mask2 = mask2 & (pattern_values_df['Task'] == task)
    
    group2_data = pattern_values_df[mask2]['Path_Sum'].dropna()
    
    pattern = pattern_values_df[mask2]['Pattern'].iloc[0]
    # Check if we have enough data
    if len(group1_data) < 3 or len(group2_data) < 3:
        print(f"Insufficient data for comparison:")
        print(f"  {pattern_group} pattern in {subject_group1}: n={len(group1_data)}")
        print(f"  {pattern_group} pattern in {subject_group2}: n={len(group2_data)}")
        return None
    
    # Test for normality using Shapiro-Wilk test
    _, p_norm1 = stats.shapiro(group1_data)
    _, p_norm2 = stats.shapiro(group2_data)
    
    # Determine if data is normally distributed (p > 0.05)
    normal_distribution = (p_norm1 > 0.05) and (p_norm2 > 0.05)
    
    # Perform appropriate statistical test
    if normal_distribution and len(group1_data) > 5 and len(group2_data) > 5:
        # Check for equal variances using Levene's test
        _, p_levene = stats.levene(group1_data, group2_data)
        equal_var = p_levene > 0.05
        
        # Perform t-test
        statistic, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
        test_type = f"Independent t-test (equal_var={equal_var})"
    else:
        # Perform Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        test_type = "Mann-Whitney U test"
    
    # Calculate descriptive statistics
    group1_stats = {
        'mean': np.round(np.mean(group1_data), 1),
        'std': np.round(np.std(group1_data, ddof=1), 1),
        'median': np.round(np.median(group1_data), 1),
        'n': len(group1_data)
    }
    
    group2_stats = {
        'mean': np.round(np.mean(group2_data), 1),
        'std': np.round(np.std(group2_data, ddof=1), 1),
        'median': np.round(np.median(group2_data), 1),
        'n': len(group2_data)
    }
    
    # Print results
    print(f"\n{'='*80}")
    print(f"STATISTICAL COMPARISON")
    print(f"{'='*80}")
    print(f"Pattern: {pattern_group} pattern")
    print(f'Patterns nodes: {pattern}')
    print(f"Comparing: {subject_group1} vs {subject_group2} subjects")
    if kinematics:
        print(f"Speed/Kinematics: {kinematics}")
    if direction:
        print(f"Direction: {direction}")    
    if task:
        print(f"Task: {task}")
    print(f"{'-'*80}")

    print(f'Weakest link in {subject_group1} group is {weakest_link_name1}')
    print(f'Weakest link in {subject_group2} group is {weakest_link_name2}')

    print(f"{'-'*80}")

    print(f"Normality tests (Shapiro-Wilk):")
    print(f"  {subject_group1} p-value: {p_norm1:.4f} {'(Normal)' if p_norm1 > 0.05 else '(Non-normal)'}")
    print(f"  {subject_group2} p-value: {p_norm2:.4f} {'(Normal)' if p_norm2 > 0.05 else '(Non-normal)'}")
    print(f"  Overall assessment: {'Normal distribution' if normal_distribution else 'Non-normal distribution'}")
    print(f"{'-'*80}")
    
    print(f"Descriptive Statistics:")
    print(f"  {subject_group1}:")
    print(f"    Mean ± SD: {group1_stats['mean']:.1f} ± {group1_stats['std']:.1f}")
    print(f"    Median: {group1_stats['median']:.1f}")
    print(f"    n = {group1_stats['n']}")
    print(f"  {subject_group2}:")
    print(f"    Mean ± SD: {group2_stats['mean']:.1f} ± {group2_stats['std']:.1f}")
    print(f"    Median: {group2_stats['median']:.1f}")
    print(f"    n = {group2_stats['n']}")
    print(f"{'-'*80}")
    
    print(f"Statistical Test: {test_type}")
    print(f"Test statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.3f}")
    print(f"Significance: {'Significant' if p_value < 0.05 else 'Not significant'} (α = 0.05)")
    if p_value < 0.001:
        print(f"Effect: Highly significant (p < 0.001)")
    elif p_value < 0.01:
        print(f"Effect: Very significant (p < 0.01)")
    elif p_value < 0.05:
        print(f"Effect: Significant (p < 0.05)")
    
    # Calculate effect size
    if normal_distribution:
        # Cohen's d for t-test
        pooled_std = np.sqrt(((group1_stats['n'] - 1) * group1_stats['std']**2 + 
                             (group2_stats['n'] - 1) * group2_stats['std']**2) / 
                            (group1_stats['n'] + group2_stats['n'] - 2))
        effect_size = abs(group1_stats['mean'] - group2_stats['mean']) / pooled_std
        print(f"Effect size (Cohen's d): {effect_size:.2f}")
    else:
        # Calculate rank-biserial correlation for Mann-Whitney U
        effect_size = 1 - (2 * statistic) / (group1_stats['n'] * group2_stats['n'])
        print(f"Effect size (rank-biserial correlation): {effect_size:.2f}")
    
    print(f"{'='*80}\n")
    
    # Return results as dictionary for further analysis
    return {
        'pattern_group': pattern_group,
        'subject_group1': subject_group1,
        'subject_group2': subject_group2,
        'task' : task,
        'kinematics': kinematics,
        'direction': direction,
        'pattern': pattern,
        'weakest_link_group1': weakest_link_name1,
        'weakest_link_group2': weakest_link_name2,
        'test_type': test_type,
        'statistic': statistic,
        'p_value': np.round(p_value, 3),
        'normal_distribution': normal_distribution,
        'group1_stats': group1_stats,
        'group2_stats': group2_stats,
        'effect_size': effect_size
    }

def multiple_corrections(comparison, use_two_stage):
    os.chdir(Path('C:\\Users\\Karolina\\Desktop\\pykinectome\\results\\patterns\\stats_dicts'))
 
    with open(comparison, 'rb') as file:
        all_results = pickle.load(file)

    # Group results by pattern length
    pattern_lengths = set([key[0] for key in all_results.keys()])


    all_significant_patterns = {}
    total_comparisons = 0
    total_significant = 0

    for pattern_length in sorted(pattern_lengths):
        length_keys = [(pl, sn) for (pl, sn) in all_results.keys() if pl == pattern_length]
        
        if not length_keys:
            continue
        
        p_values = [all_results[key]['p_value'] for key in length_keys]
        
        if use_two_stage:
            # STAGE 1: Liberal screening (FDR at 10% or uncorrected p < 0.01)
             # This identifies "promising" patterns
            promising_indices = [i for i, p in enumerate(p_values) if p < 0.01]  # Uncorrected screening
        
            if promising_indices:
                # STAGE 2: Apply correction only to promising patterns
                promising_p_values = [p_values[i] for i in promising_indices]
                promising_keys = [length_keys[i] for i in promising_indices]
            
                # Apply FDR correction to the promising subset
                rejected, p_adjusted, _, _ = multipletests(promising_p_values, alpha=0.05, method='bonferroni')
            
                print(f"Pattern length {pattern_length}:")
                print(f"  Total comparisons: {len(p_values)}")
                print(f"  Promising patterns (p < 0.01): {len(promising_indices)}")
                print(f"  Significant after Bonferroni correction: {sum(rejected)}")
            
                total_comparisons += len(p_values)
                total_significant += sum(rejected)
            
                # Add significant results
                for i, key in enumerate(promising_keys):
                    if rejected[i]:
                        all_significant_patterns[key] = all_results[key].copy()
                        all_significant_patterns[key]['p_adjusted'] = p_adjusted[i]
                        all_significant_patterns[key]['two_stage_significant'] = True
                        all_significant_patterns[key]['original_p_value'] = all_results[key]['p_value']
                        all_significant_patterns[key]['stage1_threshold'] = 0.01
                        all_significant_patterns[key]['stage2_method'] = 'fdr_bh'
        else:
            # Classic Bonferroni correction on all p-values for this pattern length
            rejected, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
        
            print(f"Pattern length {pattern_length}:")
            print(f"  Total comparisons: {len(p_values)}")
            print(f"  Significant after Bonferroni correction: {sum(rejected)}")
        
            total_comparisons += len(p_values)
            total_significant += sum(rejected)
        
            # Add significant results
            for i, key in enumerate(length_keys):
                if rejected[i]:
                    all_significant_patterns[key] = all_results[key].copy()
                    all_significant_patterns[key]['p_adjusted'] = p_adjusted[i]
                    all_significant_patterns[key]['classic_significant'] = True
                    all_significant_patterns[key]['original_p_value'] = all_results[key]['p_value']
                    all_significant_patterns[key]['correction_method'] = 'bonferroni'
    
    return all_significant_patterns

def patterns_stat_analysis(marker_list_affect, diagnosis, kinematics, task_names, tracking_systems, run, pd_on, base_path, result_base_path, full, correlation, pickle_name, interpol):
    all_results = {}
    for pattern_length in range(2, 21):
        for start_node in marker_list_affect:
            stat_analysis_dict = patterns_main(diagnosis, kinematics, task_names, tracking_systems, run, pd_on, base_path, 
                                               marker_list_affect, result_base_path, full, correlation, pattern_length, start_node, interpol, save_csv=False)
            
            if stat_analysis_dict is not None:
                # all_results[(pattern_length, start_node)] = stat_analysis_dict
                if (pattern_length, start_node) not in all_results:
                    all_results[(pattern_length, start_node)] = {}
                # Now we can update it
                all_results[(pattern_length, start_node)].update(stat_analysis_dict)## faster???

    import pickle
    with open(pickle_name, 'wb') as f:
        pickle.dump(all_results, f)



def patterns_main(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list_affect, 
                  result_base_path, full, correlation_method,
                  pattern_length, start_node, interpol, save_csv):
    
    ''' The main function to run the pattern analysis (defining patterns of n length)
    '''

    all_kinectomes = calc_std_avg_matrices(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, full, correlation_method, interpol)

    # get strongest patterns of a given length and starting a given node for each subject
    subject_patterns = get_pattern_for_subject(all_kinectomes, marker_list_affect, full, pattern_length, start_node)

    # get average group patterns of a given length and starting a given node
    group_patterns = get_avg_group_patterns(subject_patterns, pattern_length, start_node) 


    pattern_values_df = get_pattern_values_for_subjects(all_kinectomes, group_patterns, full, marker_list_affect, result_base_path,
                                                        pattern_length, start_node, save_csv)

    # input for statistical comparisons

    pattern_group1 = 'Control' # which group pattern to take for comparison
    subject_group1 = 'Parkinson' # group #1 for comparison
    subject_group2 = 'Control' # group #2 for comparison
    kinematics = 'acc'
    direction = 'AP'
    task = 'walkFast'


    analysis_dict = compare_groups_statistical(pattern_values_df, pattern_group1,
                              subject_group1, subject_group2, kinematics, direction, task)

    return analysis_dict