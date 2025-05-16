from src.data_utils.data_loader import load_kinectomes
from src.data_utils import plotting, groups, permutation
from src.graph_utils import kinectome2pattern
from src.modularity import build_graph
from src.kinectome_characteristics import calc_std_avg_matrices
import numpy as np


def get_pattern_for_subject(all_kinectomes, marker_list, full):


    for group in all_kinectomes.keys():
        for sub_id in all_kinectomes[group].keys():
            for task in all_kinectomes[group][sub_id].keys():
                for kinematics in all_kinectomes[group][sub_id][task].keys():
                    directions_list = list(all_kinectomes[group][sub_id][task][kinematics].keys())

                    if 'AP' in directions_list and 'ML' in directions_list and 'V' in directions_list:
                        # Extract the avg arrays for each direction
                        ap_kinectome = all_kinectomes[group][sub_id][task][kinematics]['AP']['avg']
                        ml_kinectome = all_kinectomes[group][sub_id][task][kinematics]['ML']['avg']
                        v_kinectome = all_kinectomes[group][sub_id][task][kinematics]['V']['avg']

                        # Stack the matrices along a new axis to create a 22x22x3 array
                        # The third dimension will have AP at index 0, ML at index 1, and V at index 2
                        combined_kinectome = np.stack([ap_kinectome, ml_kinectome, v_kinectome], axis=2)
                    
                    elif 'full' in directions_list:
                        full_kinectome = all_kinectomes[group][sub_id][task][kinematics]['full']['avg']

                    

                    graphs = build_graph(full_kinectome if full else combined_kinectome, marker_list)

                    for idx, graph in enumerate(graphs):
                        min_patern_graph = kinectome2pattern.min_pattern_subgraph(graph, length = 3, start_node = 'head')
                        min_pattern_nodes = list(min_patern_graph.nodes())
                        direction = directions_list[idx]




                    



    
    print()

def patterns_main(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list_affect, result_base_path, full, correlation_method):
    ''' The main function to run the pattern analysis (defining patterns of n length)
    '''

    all_kinectomes = calc_std_avg_matrices(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, full, correlation_method)

    get_pattern_for_subject(all_kinectomes, marker_list_affect, full)


    print()