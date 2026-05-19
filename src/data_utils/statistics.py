import numpy as np
import pandas as pd
import os
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu, kruskal, friedmanchisquare
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#used
def check_normality(data, alpha=0.05):
    """
    Check normality using Shapiro-Wilk test.
    Returns True if normal, False if not normal.
    """
    if len(data) < 3:
        return True  # Assume normal for very small samples
    
    try:
        _, p_value = stats.shapiro(data)
        return p_value > alpha
    except:
        return True  # Assume normal if test fails

#used
def check_equal_variances(groups, alpha=0.05):
    """
    Check equal variances using Levene's test.
    Returns True if equal variances, False if not.
    """
    if len(groups) < 2:
        return True
    
    try:
        _, p_value = stats.levene(*groups)
        return p_value > alpha
    except:
        return True  # Assume equal variances if test fails

#used
def remove_outliers_iqr(data, k=1.5):
    """Remove outliers using IQR method"""
    if len(data) == 0:
        return data
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

#used
def perform_pairwise_posthoc(data_long, dependent_var='value', subject_var='subject_id', 
                            factor_var='speed', alpha=0.05):
    """
    Perform pairwise post-hoc comparisons after significant repeated measures test.
    Uses Wilcoxon signed-rank tests with Bonferroni correction.
    """
    # Get unique factor levels
    factor_levels = sorted(data_long[factor_var].unique())
    
    if len(factor_levels) < 2:
        return None
    
    # Perform pairwise comparisons
    pairwise_results = []
    
    for i, level1 in enumerate(factor_levels):
        for j, level2 in enumerate(factor_levels):
            if i >= j:  # Only do each comparison once
                continue
            
            # Get data for this pair
            data1 = data_long[data_long[factor_var] == level1]
            data2 = data_long[data_long[factor_var] == level2]
            
            # Merge on subject to get paired data
            merged = data1.merge(data2, on=subject_var, suffixes=('_1', '_2'))
            
            if len(merged) < 3:
                continue
            
            values1 = merged[f'{dependent_var}_1'].dropna()
            values2 = merged[f'{dependent_var}_2'].dropna()
            
            if len(values1) != len(values2) or len(values1) < 3:
                continue
            
            # Perform Wilcoxon signed-rank test
            try:
                stat, p_value = stats.wilcoxon(values1, values2)
                
                pairwise_results.append({
                    'comparison': f"{level1} vs {level2}",
                    'level1': level1,
                    'level2': level2,
                    'test_statistic': stat,
                    'p_value': p_value,
                    'n_pairs': len(values1),
                    'mean1': np.mean(values1),
                    'mean2': np.mean(values2),
                    'median1': np.median(values1),
                    'median2': np.median(values2)
                })
            except:
                continue
    
    if not pairwise_results:
        return None
    
    # Apply Bonferroni correction to pairwise comparisons
    p_values = [result['p_value'] for result in pairwise_results]
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
    
    for i, result in enumerate(pairwise_results):
        result['p_corrected'] = p_corrected[i]
        result['significant'] = rejected[i]
    
    return pairwise_results

#used
def perform_between_group_comparison(group_data, group_labels, alpha=0.05):
    """
    Perform appropriate statistical test between groups.
    Returns test results dictionary.
    """
    n_groups = len(group_data)
    
    if n_groups < 2:
        return None
    
    # Check normality for each group
    normality_results = []
    for data in group_data:
        if len(data) > 2:
            normality_results.append(check_normality(data, alpha))
        else:
            normality_results.append(True)  # Assume normal for very small samples
    
    all_normal = all(normality_results)
    
    if n_groups == 2:
        # Two groups: t-test vs Mann-Whitney U
        if all_normal and check_equal_variances(group_data, alpha):
            # Independent t-test
            t_stat, p_value = ttest_ind(group_data[0], group_data[1])
            test_name = "Independent t-test"
            test_statistic = t_stat
        else:
            # Mann-Whitney U test
            u_stat, p_value = mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')
            test_name = "Mann-Whitney U test"
            test_statistic = u_stat
    
    else:
        # Multiple groups: ANOVA vs Kruskal-Wallis
        if all_normal and check_equal_variances(group_data, alpha):
            # One-way ANOVA
            f_stat, p_value = f_oneway(*group_data)
            test_name = "One-way ANOVA"
            test_statistic = f_stat
        else:
            # Kruskal-Wallis test
            h_stat, p_value = kruskal(*group_data)
            test_name = "Kruskal-Wallis test"
            test_statistic = h_stat
    
    return {
        'test_name': test_name,
        'test_statistic': test_statistic,
        'p_value': p_value,
        'groups': group_labels,
        'n_per_group': [len(data) for data in group_data],
        'means': [np.mean(data) for data in group_data],
        'medians': [np.median(data) for data in group_data],
        'stds': [np.std(data, ddof=1) for data in group_data],
        'normality_per_group': normality_results,
        'all_normal': all_normal
    }

#used
def analyze_centrality_statistics_by_community(df, consensus_communities, alpha=0.05, 
                                               correction_methods=['fdr_bh'], 
                                               n_bootstrap=1000, use_bootstrap=True):
    """
    Perform statistical analysis on centrality data with multiple comparison corrections
    applied separately within each community.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with centrality data (output from export_centrality_to_single_csv)
    consensus_communities : list of sets
        List of sets containing body segments for each community
    alpha : float
        Significance level (default: 0.05)
    correction_methods : list
        Multiple comparison correction methods to apply
    n_bootstrap : int
        Number of bootstrap iterations
    use_bootstrap : bool
        Whether to use bootstrap analysis
    
    Returns:
    --------
    dict : Dictionary containing all statistical results organized by community
    """
    
    # Extract segment names and create combinations
    data_columns = [col for col in df.columns if col not in ['group', 'subject_id']]
    segments = set()
    for col in data_columns:
        parts = col.split('_')
        if len(parts) >= 3:
            segment = '_'.join(parts[:-2])
            segments.add(segment)
    
    segments = sorted(segments)
    speeds = ['pref', 'slow', 'fast']
    directions = ['AP', 'ML', 'V']
    groups = df['group'].unique()
    
    # Create mapping of segments to communities
    segment_to_community = {}
    for i, community in enumerate(consensus_communities):
        for segment in community:
            segment_to_community[segment] = i
    
    results = {
        'between_groups_by_community': {},
        'within_groups_by_community': {},
        'summary_by_community': {}
    }
    
    print("=" * 80)
    print("CENTRALITY STATISTICAL ANALYSIS BY COMMUNITY")
    print("=" * 80)
    

    # Remove outliers from centrality data
    print("Removing outliers from centrality data...")
    data_columns = [col for col in df.columns if col not in ['group', 'subject_id']]
    df_clean = df.copy()

    for col in data_columns:
        if df_clean[col].dtype in ['float64', 'int64']:
            original_count = df_clean[col].notna().sum()
            df_clean[col] = remove_outliers_iqr(df_clean[col])
            final_count = df_clean[col].notna().sum()
            removed_count = original_count - final_count
            if removed_count > 0:
                print(f"  {col}: removed {removed_count} outliers ({removed_count/original_count*100:.1f}%)")

    # Use cleaned dataframe for analysis
    df = df_clean
    print(f"Outlier removal complete. Original shape: {df.shape}")

    # Process each community separately
    for community_idx, community in enumerate(consensus_communities):
        print(f"\n{'='*20} COMMUNITY {community_idx + 1} {'='*20}")
        print(f"Segments: {', '.join(sorted(community))}")
        
        results['between_groups_by_community'][community_idx] = {}
        results['within_groups_by_community'][community_idx] = {}
        
        # 1. BETWEEN-GROUP COMPARISONS for this community
        print(f"\n1. BETWEEN-GROUP COMPARISONS - COMMUNITY {community_idx + 1}")
        print("-" * 60)
        
        between_group_tests = []
        
        for segment in segments:
            if segment not in community:
                continue
                
            results['between_groups_by_community'][community_idx][segment] = {}
            
            for speed in speeds:
                results['between_groups_by_community'][community_idx][segment][speed] = {}
                
                for direction in directions:
                    col_name = f"{segment}_{speed}_{direction}"
                    
                    if col_name not in df.columns:
                        continue
                    
                    # Prepare data for comparison
                    group_data = []
                    group_labels = []
                    
                    for group in groups:
                        group_df = df[df['group'] == group]
                        values = group_df[col_name].dropna()
                        
                        if len(values) > 0:
                            group_data.append(values)
                            group_labels.append(group)
                    
                    if len(group_data) < 2:
                        continue
                    
                    # Perform statistical test
                    if use_bootstrap:
                        test_result = perform_between_group_comparison_with_iterative_bootstrap(
                            group_data, group_labels, alpha, n_bootstrap, col_name)
                    else:
                        test_result = perform_between_group_comparison(group_data, group_labels, alpha)
                    
                    if test_result:
                        test_result['variable'] = col_name
                        test_result['community'] = community_idx
                        results['between_groups_by_community'][community_idx][segment][speed][direction] = test_result
                        between_group_tests.append(test_result)
        
        # Apply multiple comparison corrections within this community
        if between_group_tests:
            p_values = [test['p_value'] for test in between_group_tests]
            
            for method in correction_methods:
                try:
                    rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
                    
                    print(f"\nBetween-group results for Community {community_idx + 1} (corrected with {method.upper()}):")
                    print(f"{'Variable':<25} {'Test':<20} {'Statistic':<12} {'p-value':<12} {'p-corrected':<12} {'Significant':<12}")
                    print("-" * 105)
                    
                    for i, test in enumerate(between_group_tests):
                        sig = "Yes" if rejected[i] else "No"
                        bootstrap_info = ""
                        if 'iterative_bootstrap' in test:
                            bootstrap_info = f" | Bootstrap p: {test['bootstrap_p_value']:.4f}"
                        
                        print(f"{test['variable']:<25} {test['test_name']:<20} {test['test_statistic']:<12.3f} "
                              f"{test['p_value']:<12.4f} {p_corrected[i]:<12.4f} {sig:<12}{bootstrap_info}")
                        
                        # Add corrected results to the test info
                        test[f'p_corrected_{method}'] = p_corrected[i]
                        test[f'significant_{method}'] = rejected[i]
                
                except Exception as e:
                    print(f"Error applying {method} correction: {e}")
        
        # 2. WITHIN-GROUP COMPARISONS for this community
        print(f"\n2. WITHIN-GROUP COMPARISONS - COMMUNITY {community_idx + 1}")
        print("-" * 60)
        
        within_group_tests = []
        
        for group in groups:
            results['within_groups_by_community'][community_idx][group] = {}
            group_df = df[df['group'] == group].copy()
            
            for segment in segments:
                if segment not in community:
                    continue
                    
                results['within_groups_by_community'][community_idx][group][segment] = {}
                
                for direction in directions:
                    # Prepare data for repeated measures analysis
                    speed_columns = [f"{segment}_{speed}_{direction}" for speed in speeds]
                    available_columns = [col for col in speed_columns if col in df.columns]
                    
                    if len(available_columns) < 2:
                        continue
                    
                    # Create long format for repeated measures analysis
                    rm_data = []
                    
                    for idx, row in group_df.iterrows():
                        subject_id = row['subject_id']
                        
                        for speed_col in available_columns:
                            if not pd.isna(row[speed_col]):
                                speed = speed_col.split('_')[-2]
                                rm_data.append({
                                    'subject_id': subject_id,
                                    'speed': speed,
                                    'value': row[speed_col]
                                })
                    
                    if len(rm_data) == 0:
                        continue
                    
                    rm_df = pd.DataFrame(rm_data)
                    
                    # Filter to complete subjects
                    available_speeds = rm_df['speed'].unique()
                    subject_counts = rm_df.groupby('subject_id')['speed'].count()
                    complete_subjects = subject_counts[subject_counts == len(available_speeds)].index
                    
                    if len(complete_subjects) < 3:
                        continue
                    
                    rm_df = rm_df[rm_df['subject_id'].isin(complete_subjects)]
                    
                    # Perform statistical test (same as before)
                    try:
                        # Check normality and choose appropriate test
                        speed_groups = rm_df.groupby('speed')['value'].apply(list).to_dict()
                        use_parametric = True
                        
                        if len(speed_groups) == 3 and all(len(vals) >= 3 for vals in speed_groups.values()):
                            for speed_vals in speed_groups.values():
                                if not check_normality(speed_vals, alpha):
                                    use_parametric = False
                                    break
                        
                        if use_parametric and len(speed_groups) >= 2:
                            rm_anova = AnovaRM(rm_df, 'value', 'subject_id', within=['speed'])
                            rm_result = rm_anova.fit()
                            f_stat = rm_result.anova_table.loc['speed', 'F Value']
                            p_value = rm_result.anova_table.loc['speed', 'Pr > F']
                            test_name = "Repeated Measures ANOVA"
                            test_statistic = f_stat
                        else:
                            pivot_df = rm_df.pivot(index='subject_id', columns='speed', values='value')
                            complete_data = pivot_df.dropna()
                            if len(complete_data) < 3 or len(complete_data.columns) < 2:
                                continue
                            test_statistic, p_value = friedmanchisquare(*[complete_data[col] for col in complete_data.columns])
                            test_name = "Friedman test"
                        
                        # Calculate descriptive statistics
                        speed_means = rm_df.groupby('speed')['value'].mean()
                        speed_stds = rm_df.groupby('speed')['value'].std()
                        speed_medians = rm_df.groupby('speed')['value'].median()
                        
                        test_info = {
                            'variable': f"{segment}_{direction}",
                            'group': group,
                            'community': community_idx,
                            'test_name': test_name,
                            'test_statistic': test_statistic,
                            'p_value': p_value,
                            'n_subjects': len(complete_subjects),
                            'speeds_tested': list(speed_means.index),
                            'means': speed_means.to_dict(),
                            'medians': speed_medians.to_dict(),
                            'stds': speed_stds.to_dict(),
                            'parametric_test': use_parametric
                        }
                        
                        results['within_groups_by_community'][community_idx][group][segment][direction] = test_info
                        within_group_tests.append(test_info)
                        
                        # Add bootstrap if requested
                        if use_bootstrap:
                            bootstrap_within_result = perform_within_group_iterative_bootstrap(rm_df, n_bootstrap, alpha)
                            if bootstrap_within_result is not None:
                                test_info['bootstrap_within'] = bootstrap_within_result
                                test_info['bootstrap_p_value'] = bootstrap_within_result['p_values'][-1]
                                test_info['bootstrap_significant'] = bootstrap_within_result['p_values'][-1] < alpha
                    
                    except Exception as e:
                        print(f"Error in repeated measures analysis for {group} - {segment}_{direction}: {e}")
                        continue
        
        # Apply multiple comparison corrections within this community for within-group tests
        if within_group_tests:
            p_values = [test['p_value'] for test in within_group_tests]
            
            for method in correction_methods:
                try:
                    rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
                    
                    print(f"\nWithin-group results for Community {community_idx + 1} (corrected with {method.upper()}):")
                    print(f"{'Group':<12} {'Variable':<18} {'Test':<20} {'Statistic':<12} {'p-value':<12} {'p-corrected':<12} {'Significant':<12}")
                    print("-" * 115)
                    
                    for i, test in enumerate(within_group_tests):
                        sig = "Yes" if rejected[i] else "No"
                        bootstrap_info = ""
                        if 'bootstrap_within' in test:
                            bootstrap_info = f" | Bootstrap p: {test['bootstrap_p_value']:.4f}"
                        
                        print(f"{test['group']:<12} {test['variable']:<18} {test['test_name']:<20} "
                              f"{test['test_statistic']:<12.3f} {test['p_value']:<12.4f} {p_corrected[i]:<12.4f} {sig:<12}{bootstrap_info}")
                        
                        # Add corrected results
                        test[f'p_corrected_{method}'] = p_corrected[i]
                        test[f'significant_{method}'] = rejected[i]
                        
                        # Perform post-hoc tests if significant
                        if rejected[i] and len(test['speeds_tested']) > 2:
                            try:
                                # Better variable name parsing
                                variable_parts = test['variable'].split('_')
                                if len(variable_parts) >= 2:
                                    direction = variable_parts[-1]  # Last part is direction
                                    segment = '_'.join(variable_parts[:-1])  # Everything else is segment
                                else:
                                    continue
                                
                                # Get the original data for post-hoc analysis
                                group_df = df[df['group'] == test['group']].copy()
                                speed_columns = [f"{segment}_{speed}_{direction}" for speed in test['speeds_tested']]
                                available_columns = [col for col in speed_columns if col in df.columns]
                                
                                if not available_columns:
                                    continue
                                
                                # Reconstruct rm_data for post-hoc
                                rm_data_posthoc = []
                                for idx, row in group_df.iterrows():
                                    subject_id = row['subject_id']
                                    for speed_col in available_columns:
                                        if not pd.isna(row[speed_col]):
                                            speed = speed_col.split('_')[-2]
                                            rm_data_posthoc.append({
                                                'subject_id': subject_id,
                                                'speed': speed,
                                                'value': row[speed_col]
                                            })
                                
                                if not rm_data_posthoc:
                                    continue
                                
                                rm_df_posthoc = pd.DataFrame(rm_data_posthoc)
                                
                                # Filter to complete subjects
                                subject_counts = rm_df_posthoc.groupby('subject_id')['speed'].count()
                                complete_subjects = subject_counts[subject_counts >= 2].index
                                rm_df_posthoc = rm_df_posthoc[rm_df_posthoc['subject_id'].isin(complete_subjects)]
                                
                                if rm_df_posthoc.empty:
                                    continue
                                
                                posthoc_results = perform_pairwise_posthoc(rm_df_posthoc)
                                
                                if posthoc_results:
                                    test['posthoc_results'] = posthoc_results
                                    print(f"    Post-hoc comparisons:")
                                    print(f"    {'Comparison':<20} {'p-value':<12} {'p-corrected':<12} {'Significant':<12}")
                                    print(f"    {'-'*60}")
                                    
                                    for ph_result in posthoc_results:
                                        ph_sig = "Yes" if ph_result['significant'] else "No"
                                        print(f"    {ph_result['comparison']:<20} {ph_result['p_value']:<12.4f} "
                                              f"{ph_result['p_corrected']:<12.4f} {ph_sig:<12}")
                                        
                            except Exception as e:
                                print(f"    Error in post-hoc analysis: {e}")
                                continue
                
                except Exception as e:
                    print(f"Error applying {method} correction: {e}")
        
        # Summary for this community
        results['summary_by_community'][community_idx] = {
            'n_between_group_tests': len([test for test in between_group_tests if test['community'] == community_idx]),
            'n_within_group_tests': len([test for test in within_group_tests if test['community'] == community_idx]),
            'segments': list(community),
            'correction_methods': correction_methods,
            'alpha': alpha
        }
    
    return results

#used in centrality
def perform_within_group_iterative_bootstrap(rm_df, n_bootstrap=1000, alpha=0.05):
    """
    Perform iterative bootstrap analysis for within-group repeated measures data.
    
    Parameters:
    -----------
    rm_df : pandas.DataFrame
        Long-format dataframe with columns: subject_id, speed, value
    n_bootstrap : int
        Number of bootstrap iterations per sample size
    alpha : float
        Significance level
    
    Returns:
    --------
    dict : Bootstrap results across all sample sizes
    """
    
    # Get unique speeds and calculate original statistics
    speeds = sorted(rm_df['speed'].unique())
    if len(speeds) < 2:
        return None
    
    # Calculate original means for each speed
    original_means = rm_df.groupby('speed')['value'].mean().to_dict()
    
    # Get subjects with complete data (all speeds)
    subject_counts = rm_df.groupby('subject_id')['speed'].count()
    complete_subjects = subject_counts[subject_counts == len(speeds)].index.tolist()
    rm_df_complete = rm_df[rm_df['subject_id'].isin(complete_subjects)].copy()
    
    if len(complete_subjects) < 3:
        return None
    
    # Sample size fractions to test (10% to 90%)
    sample_fractions = np.arange(0.1, 1.0, 0.1)
    
    results = {
        'original_means': original_means,
        'sample_fractions': sample_fractions,
        'bootstrap_means_by_speed': {speed: [] for speed in speeds},
        'bootstrap_stds_by_speed': {speed: [] for speed in speeds},
        'bootstrap_diffs_all': [],  # For pairwise differences if 2 speeds
        'f_statistics': [],  # For ANOVA if >2 speeds
        'p_values': [],
        'speeds': speeds,
        'complete_subjects': complete_subjects,
        'n_complete_subjects': len(complete_subjects),
        'n_bootstrap': n_bootstrap
    }
    
    for i, fraction in enumerate(sample_fractions):
        print(f"      Within-group bootstrapping at {int(fraction*100)}% sample size ({i+1}/{len(sample_fractions)})...")
        
        n_subjects_sample = max(2, int(len(complete_subjects) * fraction))
        
        bootstrap_means_per_speed = {speed: [] for speed in speeds}
        bootstrap_f_stats = []
        bootstrap_diffs = []  # For pairwise comparison if only 2 speeds
        
        for boot_iter in range(n_bootstrap):
            # Sample subjects with replacement
            sampled_subjects = np.random.choice(complete_subjects, size=n_subjects_sample, replace=True)
            
            # Get data for sampled subjects
            boot_data = rm_df_complete[rm_df_complete['subject_id'].isin(sampled_subjects)]
            
            # Calculate means for each speed in this bootstrap sample
            boot_means = boot_data.groupby('speed')['value'].mean()
            
            for speed in speeds:
                bootstrap_means_per_speed[speed].append(boot_means[speed])
            
            # If only 2 speeds, calculate pairwise difference
            if len(speeds) == 2:
                diff = boot_means[speeds[0]] - boot_means[speeds[1]]
                bootstrap_diffs.append(diff)
            
            # If more than 2 speeds, calculate F-statistic (simplified ANOVA)
            elif len(speeds) > 2:
                try:
                    # Simple one-way ANOVA F-statistic calculation
                    speed_groups = [boot_data[boot_data['speed'] == speed]['value'].values for speed in speeds]
                    f_stat, _ = stats.f_oneway(*speed_groups)
                    if not np.isnan(f_stat):
                        bootstrap_f_stats.append(f_stat)
                except:
                    pass
        
        # Store bootstrap means and stds for each speed
        for speed in speeds:
            results['bootstrap_means_by_speed'][speed].append(np.mean(bootstrap_means_per_speed[speed]))
            results['bootstrap_stds_by_speed'][speed].append(np.std(bootstrap_means_per_speed[speed]))
        
        # Calculate p-value based on test type
        if len(speeds) == 2:
            # For pairwise comparison
            original_diff = original_means[speeds[0]] - original_means[speeds[1]]
            bootstrap_diffs = np.array(bootstrap_diffs)
            
            # Two-tailed test
            if original_diff >= 0:
                p_value = 2 * np.mean(bootstrap_diffs <= -abs(original_diff))
            else:
                p_value = 2 * np.mean(bootstrap_diffs >= abs(original_diff))
            p_value = min(p_value, 1.0)
            
            results['bootstrap_diffs_all'].append(bootstrap_diffs)
            
        else:
            # For ANOVA-like comparison
            if bootstrap_f_stats:
                # Calculate original F-statistic
                original_speed_groups = [rm_df_complete[rm_df_complete['speed'] == speed]['value'].values for speed in speeds]
                original_f, original_p = stats.f_oneway(*original_speed_groups)
                
                # Bootstrap p-value: proportion of bootstrap F-stats >= original F
                p_value = np.mean(np.array(bootstrap_f_stats) >= original_f)
                results['f_statistics'].append(np.mean(bootstrap_f_stats))
            else:
                p_value = 1.0
                results['f_statistics'].append(0.0)
        
        results['p_values'].append(p_value)
    
    return results

#used in centrality
def create_within_group_bootstrap_plot(bootstrap_result, group_name, variable_name, save_path, correction_method=''):
    """
    Create comprehensive within-group bootstrap analysis plots.
    
    Parameters:
    -----------
    bootstrap_result : dict
        Result from perform_within_group_iterative_bootstrap
    group_name : str
        Name of the group
    variable_name : str
        Name of the variable being tested
    save_path : str
        Path to save the plot
    correction_method : str
        Correction method used
    """
    
    fig = plt.figure(figsize=(20, 12))
    
    sample_fractions = bootstrap_result['sample_fractions']
    sample_sizes_pct = sample_fractions * 100
    speeds = bootstrap_result['speeds']
    colors = plt.cm.Set1(np.linspace(0, 1, len(speeds)))
    
    # Plot 1: Bootstrap Means Stability for Each Speed
    ax1 = plt.subplot(3, 3, 1)
    for speed, color in zip(speeds, colors):
        means = bootstrap_result['bootstrap_means_by_speed'][speed]
        stds = bootstrap_result['bootstrap_stds_by_speed'][speed]
        
        ax1.plot(sample_sizes_pct, means, 'o-', color=color, linewidth=2, markersize=6, label=f'{speed}')
        ax1.fill_between(sample_sizes_pct, 
                         np.array(means) - np.array(stds), 
                         np.array(means) + np.array(stds), 
                         alpha=0.2, color=color)
        
        # Add original mean line
        original_mean = bootstrap_result['original_means'][speed]
        ax1.axhline(y=original_mean, color=color, linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Sample Size (%)')
    ax1.set_ylabel('Bootstrap Mean')
    ax1.set_title(f'Speed Means Stability vs Sample Size\n{group_name} - {variable_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: P-value Stability
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(sample_sizes_pct, bootstrap_result['p_values'], 'ro-', linewidth=2, markersize=6)
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    ax2.set_xlabel('Sample Size (%)')
    ax2.set_ylabel('P-value')
    ax2.set_title('P-value Stability vs Sample Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, min(1.0, max(bootstrap_result['p_values']) * 1.1))
    
    # Plot 3: Standard Deviation by Speed
    ax3 = plt.subplot(3, 3, 3)
    for speed, color in zip(speeds, colors):
        stds = bootstrap_result['bootstrap_stds_by_speed'][speed]
        ax3.plot(sample_sizes_pct, stds, 'o-', color=color, linewidth=2, markersize=6, label=f'{speed}')
    
    ax3.set_xlabel('Sample Size (%)')
    ax3.set_ylabel('Bootstrap Standard Deviation')
    ax3.set_title('Bootstrap Variability by Speed')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plots 4-6: Distribution plots for pairwise differences (if 2 speeds) at 10%, 50%, 90%
    if len(speeds) == 2 and bootstrap_result['bootstrap_diffs_all']:
        sample_indices = [0, 4, 8]  # 10%, 50%, 90%
        sample_labels = ['10%', '50%', '90%']
        plot_colors = ['lightcoral', 'lightblue', 'lightgreen']
        
        original_diff = bootstrap_result['original_means'][speeds[0]] - bootstrap_result['original_means'][speeds[1]]
        
        for i, (idx, label, color) in enumerate(zip(sample_indices, sample_labels, plot_colors)):
            if idx < len(bootstrap_result['bootstrap_diffs_all']):
                ax = plt.subplot(3, 3, 4 + i)
                bootstrap_diffs = bootstrap_result['bootstrap_diffs_all'][idx]
                
                ax.hist(bootstrap_diffs, bins=50, alpha=0.7, density=True, color=color, edgecolor='black')
                ax.axvline(original_diff, color='red', linestyle='--', linewidth=2, 
                          label=f'Observed: {original_diff:.3f}')
                ax.axvline(np.mean(bootstrap_diffs), color='blue', linestyle='-', linewidth=2,
                          label=f'Bootstrap Mean: {np.mean(bootstrap_diffs):.3f}')
                
                ax.set_xlabel(f'Difference ({speeds[0]} - {speeds[1]})')
                ax.set_ylabel('Density')
                n_subjects = int(sample_fractions[idx] * bootstrap_result['n_complete_subjects'])
                ax.set_title(f'Bootstrap Distribution ({label} sample size)\nn={n_subjects} | p={bootstrap_result["p_values"][idx]:.4f}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
    
    # Plot 7: Bar plot of final means with error bars
    ax7 = plt.subplot(3, 3, 7)
    final_means = [bootstrap_result['bootstrap_means_by_speed'][speed][-1] for speed in speeds]
    final_stds = [bootstrap_result['bootstrap_stds_by_speed'][speed][-1] for speed in speeds]
    original_means_list = [bootstrap_result['original_means'][speed] for speed in speeds]
    
    x_pos = np.arange(len(speeds))
    width = 0.35
    
    ax7.bar(x_pos - width/2, original_means_list, width, label='Original', alpha=0.7, color='gray')
    ax7.bar(x_pos + width/2, final_means, width, yerr=final_stds, label='Bootstrap (90%)', alpha=0.7, capsize=5)
    
    ax7.set_xlabel('Speed Condition')
    ax7.set_ylabel('Mean Value')
    ax7.set_title('Original vs Bootstrap Means (90% sample)')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(speeds)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Summary statistics
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    correction_info = f" ({correction_method.upper()} corrected)" if correction_method else ""
    summary_text = f"""
    WITHIN-GROUP BOOTSTRAP ANALYSIS{correction_info}
    
    Group: {group_name}
    Variable: {variable_name}
    Speed Conditions: {', '.join(speeds)}
    Complete Subjects: {bootstrap_result['n_complete_subjects']}
    Bootstrap Iterations per Size: {bootstrap_result['n_bootstrap']}
    
    FINAL RESULTS (90% sample):
    • P-value: {bootstrap_result['p_values'][-1]:.4f}
    • Significant: {'Yes' if bootstrap_result['p_values'][-1] < 0.05 else 'No'}
    
    ORIGINAL MEANS:
    """
    
    for speed in speeds:
        summary_text += f"• {speed}: {bootstrap_result['original_means'][speed]:.3f}\n    "
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def perform_iterative_bootstrap_comparison(group_data, group_labels, n_bootstrap=1000, alpha=0.05):
    """
    Perform iterative bootstrapping comparison between groups from 10% to 90% sample sizes.
    
    Parameters:
    -----------
    group_data : list of arrays
        Data for each group
    group_labels : list of str
        Labels for each group
    n_bootstrap : int
        Number of bootstrap iterations per sample size
    alpha : float
        Significance level
    
    Returns:
    --------
    dict : Bootstrap results across all sample sizes
    """
    
    if len(group_data) != 2:
        return None  # Currently only supports two-group comparisons
    
    group1_data = np.array(group_data[0])
    group2_data = np.array(group_data[1])
    
    # Original difference in means
    original_diff = np.mean(group1_data) - np.mean(group2_data)
    
    # Sample size fractions to test (10% to 90%)
    sample_fractions = np.arange(0.1, 1.0, 0.1)
    
    results = {
        'original_difference': original_diff,
        'sample_fractions': sample_fractions,
        'bootstrap_means': [],
        'bootstrap_stds': [],
        'bootstrap_diffs_all': [],
        'p_values': [],
        'ci_lowers': [],
        'ci_uppers': [],
        'correlations_with_original': [],
        'group_labels': group_labels,
        'group1_original_size': len(group1_data),
        'group2_original_size': len(group2_data),
        'n_bootstrap': n_bootstrap
    }
    
    for i, fraction in enumerate(sample_fractions):
        print(f"    Bootstrapping at {int(fraction*100)}% sample size ({group_data[0].name}))...")
        bootstrap_diffs = []
        
        for i in range(n_bootstrap):
            # Sample with replacement
            n1_sample = max(1, int(len(group1_data) * fraction))
            n2_sample = max(1, int(len(group2_data) * fraction))
            
            bootstrap_sample1 = np.random.choice(group1_data, size=n1_sample, replace=True)
            bootstrap_sample2 = np.random.choice(group2_data, size=n2_sample, replace=True)
            
            bootstrap_diff = np.mean(bootstrap_sample1) - np.mean(bootstrap_sample2)
            bootstrap_diffs.append(bootstrap_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate statistics for this sample size
        bootstrap_mean = np.mean(bootstrap_diffs)
        bootstrap_std = np.std(bootstrap_diffs)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_diffs, (alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
        
        # P-value: proportion of bootstrap samples where difference is as extreme as observed
        if original_diff >= 0:
            p_value = 2 * np.mean(bootstrap_diffs <= -abs(original_diff))
        else:
            p_value = 2 * np.mean(bootstrap_diffs >= abs(original_diff))
        p_value = min(p_value, 1.0)
        

        # Correlation with original difference (stability measure)
        if len(bootstrap_diffs) > 1 and np.std(bootstrap_diffs) > 1e-10:
            correlation = np.corrcoef([original_diff] * len(bootstrap_diffs), bootstrap_diffs)[0, 1]
            if np.isnan(correlation):
                correlation = 0
        else:
            correlation = 0

        # Store results
        results['bootstrap_means'].append(bootstrap_mean)
        results['bootstrap_stds'].append(bootstrap_std)
        results['bootstrap_diffs_all'].append(bootstrap_diffs)
        results['p_values'].append(p_value)
        results['ci_lowers'].append(ci_lower)
        results['ci_uppers'].append(ci_upper)
        results['correlations_with_original'].append(correlation)
    
    return results

def create_comprehensive_bootstrap_plot(bootstrap_result, variable_name, save_path, correction_method=''):
    """
    Create comprehensive bootstrap analysis plots.
    
    Parameters:
    -----------
    bootstrap_result : dict
        Result from perform_iterative_bootstrap_comparison
    variable_name : str
        Name of the variable being tested
    save_path : str
        Path to save the plot
    """
    
    fig = plt.figure(figsize=(20, 15))
    
    sample_fractions = bootstrap_result['sample_fractions']
    sample_sizes_pct = sample_fractions * 100
    original_diff = bootstrap_result['original_difference']
    
    # Plot 1: Bootstrap Mean Stability vs Sample Size (similar to your fig 2)
    ax1 = plt.subplot(3, 3, 1)
    bootstrap_means = np.array(bootstrap_result['bootstrap_means'])
    bootstrap_stds = np.array(bootstrap_result['bootstrap_stds'])
    
    ax1.plot(sample_sizes_pct, bootstrap_means, 'bo-', linewidth=2, markersize=6, label='Bootstrap Mean')
    ax1.fill_between(sample_sizes_pct, 
                     bootstrap_means - bootstrap_stds, 
                     bootstrap_means + bootstrap_stds, 
                     alpha=0.2, color='blue', label='±1 SD')
    ax1.axhline(y=original_diff, color='red', linestyle='--', linewidth=2, label=f'Observed: {original_diff:.3f}')
    ax1.set_xlabel('Sample Size (%)')
    ax1.set_ylabel('Bootstrap Mean Difference')
    ax1.set_title('Bootstrap Mean Stability vs Sample Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: P-value Stability vs Sample Size
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(sample_sizes_pct, bootstrap_result['p_values'], 'go-', linewidth=2, markersize=6)
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    ax2.set_xlabel('Sample Size (%)')
    ax2.set_ylabel('P-value')
    ax2.set_title('P-value Stability vs Sample Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, min(1.0, max(bootstrap_result['p_values']) * 1.1))
    
    # Plot 3: Confidence Interval Width vs Sample Size
    ax3 = plt.subplot(3, 3, 3)
    ci_widths = np.array(bootstrap_result['ci_uppers']) - np.array(bootstrap_result['ci_lowers'])
    ax3.plot(sample_sizes_pct, ci_widths, 'mo-', linewidth=2, markersize=6)
    ax3.set_xlabel('Sample Size (%)')
    ax3.set_ylabel('CI Width (95%)')
    ax3.set_title('Confidence Interval Width vs Sample Size')
    ax3.grid(True, alpha=0.3)
    if len(set(ci_widths)) > 1:  # Only set ylim if there's variation
        ax3.set_ylim(min(ci_widths) * 0.9, max(ci_widths) * 1.1)
    
    # Plots 4-6: Distribution plots for 10%, 50%, and 90% (similar to your fig 3)
    sample_indices = [0, 4, 8]  # 10%, 50%, 90%
    sample_labels = ['10%', '50%', '90%']
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    for i, (idx, label, color) in enumerate(zip(sample_indices, sample_labels, colors)):
        ax = plt.subplot(3, 3, 4 + i)
        bootstrap_diffs = bootstrap_result['bootstrap_diffs_all'][idx]
        
        ax.hist(bootstrap_diffs, bins=50, alpha=0.7, density=True, color=color, edgecolor='black')
        ax.axvline(original_diff, color='red', linestyle='--', linewidth=2, 
                  label=f'Observed: {original_diff:.3f}')
        ax.axvline(bootstrap_result['bootstrap_means'][idx], color='blue', linestyle='-', linewidth=2,
                  label=f'Bootstrap Mean: {bootstrap_result["bootstrap_means"][idx]:.3f}')
        ax.axvline(bootstrap_result['ci_lowers'][idx], color='orange', linestyle=':', alpha=0.8)
        ax.axvline(bootstrap_result['ci_uppers'][idx], color='orange', linestyle=':', alpha=0.8,
                  label=f'95% CI: [{bootstrap_result["ci_lowers"][idx]:.3f}, {bootstrap_result["ci_uppers"][idx]:.3f}]')
        
        ax.set_xlabel('Difference in Means')
        ax.set_ylabel('Density')
        ax.set_title(f'Bootstrap Distribution ({label} sample size)\nn={int(sample_fractions[idx] * bootstrap_result["group1_original_size"])}, {int(sample_fractions[idx] * bootstrap_result["group2_original_size"])} | p={bootstrap_result["p_values"][idx]:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot 7: Standard Deviation vs Sample Size
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(sample_sizes_pct, bootstrap_result['bootstrap_stds'], 'co-', linewidth=2, markersize=6)
    ax7.set_xlabel('Sample Size (%)')
    ax7.set_ylabel('Bootstrap Standard Deviation')
    ax7.set_title('Bootstrap Variability vs Sample Size')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Correlation Stability (measure of how consistent results are)
    ax8 = plt.subplot(3, 3, 8)
    correlations = [abs(corr) if not np.isnan(corr) else 0 for corr in bootstrap_result['correlations_with_original']]
    ax8.plot(sample_sizes_pct, correlations, 'ro-', linewidth=2, markersize=6)
    ax8.set_xlabel('Sample Size (%)')
    ax8.set_ylabel('Absolute Correlation with Full Sample')
    ax8.set_title('Result Consistency vs Sample Size')
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 1)
    
    # Plot 9: Summary statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Create summary table    
    
    correction_info = f" ({correction_method.upper()} corrected)" if correction_method else ""
    summary_text = f"""
    BOOTSTRAP ANALYSIS SUMMARY{correction_info}
        
    Variable: {variable_name}


    Groups: {' vs '.join(bootstrap_result['group_labels'])}
    Original Sample Sizes: n₁={bootstrap_result['group1_original_size']}, n₂={bootstrap_result['group2_original_size']}
    Bootstrap Iterations per Size: {bootstrap_result['n_bootstrap']}
    
    OBSERVED DIFFERENCE: {original_diff:.4f}
    
    STABILITY ANALYSIS:
    • Most stable at 90% sample size
    • Bootstrap Mean (90%): {bootstrap_result['bootstrap_means'][-1]:.4f}
    • Bootstrap Std (90%): {bootstrap_result['bootstrap_stds'][-1]:.4f}
    • P-value (90%): {bootstrap_result['p_values'][-1]:.4f}
    • CI Width (90%): {ci_widths[-1]:.4f}
    
    MINIMUM STABLE SAMPLE SIZE:
    """
    
    # Find minimum stable sample size (where p-value stabilizes)
    p_vals = np.array(bootstrap_result['p_values'])
    final_p = p_vals[-1]
    stable_threshold = 0.1 * final_p  # 10% variation threshold
    
    for i, p_val in enumerate(p_vals):
        if abs(p_val - final_p) <= stable_threshold:
            min_stable_pct = int(sample_sizes_pct[i])
            summary_text += f"~{min_stable_pct}% (p-value stabilized)"
            break
    else:
        summary_text += "Not achieved in tested range"
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

#used
def perform_between_group_comparison_with_iterative_bootstrap(group_data, group_labels, alpha, n_bootstrap=1000, variable_name=""):
    """
    Enhanced version that performs iterative bootstrap analysis.
    
    Parameters:
    -----------
    group_data : list of arrays
        Data for each group
    group_labels : list of str
        Labels for each group  
    alpha : float
        Significance level
    n_bootstrap : int
        Number of bootstrap iterations per sample size
    variable_name : str
        Name of the variable (for plot saving)
    
    Returns:
    --------
    dict : Test results including iterative bootstrap results
    """
    
    # First perform the original statistical test
    original_result = perform_between_group_comparison(group_data, group_labels, alpha)
    
    # Then perform iterative bootstrap comparison
    bootstrap_result = perform_iterative_bootstrap_comparison(group_data, group_labels, n_bootstrap, alpha)
    
    if bootstrap_result is not None:
        # Create and save comprehensive bootstrap plot
        safe_variable_name = variable_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        plot_filename = f"iterative_bootstrap_{safe_variable_name}_{'_vs_'.join(group_labels)}.png"
        plot_path = os.path.join("C:\\Users\\Karolina\\Desktop\\pykinectome\\results\\centrality\\bootstrapping", plot_filename)
        
        create_comprehensive_bootstrap_plot(bootstrap_result, variable_name, plot_path, "")
        
        # Use 90% bootstrap results as the main bootstrap statistics
        final_idx = -1  # 90% sample size
        
        # Combine original and bootstrap results
        if original_result:
            original_result['iterative_bootstrap'] = bootstrap_result
            original_result['bootstrap_p_value'] = bootstrap_result['p_values'][final_idx]
            original_result['bootstrap_significant'] = bootstrap_result['p_values'][final_idx] < alpha
            original_result['bootstrap_ci'] = (bootstrap_result['ci_lowers'][final_idx], bootstrap_result['ci_uppers'][final_idx])
            original_result['bootstrap_mean'] = bootstrap_result['bootstrap_means'][final_idx]
            original_result['bootstrap_std'] = bootstrap_result['bootstrap_stds'][final_idx]
        else:
            # If original test failed, return bootstrap results as main results
            original_result = {
                'test_name': 'Iterative Bootstrap test',
                'test_statistic': bootstrap_result['original_difference'],
                'p_value': bootstrap_result['p_values'][final_idx],
                'groups': group_labels,
                'n_groups': len(group_labels),
                'group_sizes': [bootstrap_result['group1_original_size'], bootstrap_result['group2_original_size']],
                'iterative_bootstrap': bootstrap_result,
                'bootstrap_p_value': bootstrap_result['p_values'][final_idx],
                'bootstrap_significant': bootstrap_result['p_values'][final_idx] < alpha,
                'bootstrap_ci': (bootstrap_result['ci_lowers'][final_idx], bootstrap_result['ci_uppers'][final_idx]),
                'bootstrap_mean': bootstrap_result['bootstrap_means'][final_idx],
                'bootstrap_std': bootstrap_result['bootstrap_stds'][final_idx]
            }
    
    return original_result

