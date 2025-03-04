import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import networkx as nx
from src.data_utils import data_loader
from src.kinectome import *
from src.preprocessing import preprocessing
from src.data_utils.kinecome2graph import build_graph, weighted_degree_centrality




if __name__ == "__main__":
    ##
    # LOADING CONFIG CONSTANTS
    ##
    # DATA-PATH
    DATA_PATH = "/home/prdc/Dokumente/Projects-KI/uksh"
    # Seleted tasks
    TASK_NAMES = [
    "walkFast", "walkSlow", "walkPreferred" 
    ]
    TRACKING_SYSTEMS = ["omc"]
    # values to be extracted 
    KINEMATICS = ['vel', 'pos', 'acc']
    # ordered list of markers
    MARKER_LIST = ['head', 'ster', 'l_sho', 'r_sho',  
                'l_elbl', 'r_elbl','l_wrist', 'r_wrist', 'l_hand', 'r_hand', 
                'l_asis', 'l_psis', 'r_asis', 'r_psis', 
                'l_th', 'r_th', 'l_sk', 'r_sk', 
                'l_ank', 'r_ank', 'l_toe', 'r_toe']
    run = None # dont have id "run" in sample data that I have 
    FS = 200 # sampling rate
    sub_ids = ["pp002"] # I have only this data 

    GEN_KINECTOMES = False # generate the kinectomes in first run 

    if GEN_KINECTOMES:
        for sub_id in sub_ids:
            for kinematics in KINEMATICS:
                for task_name in TASK_NAMES:
                    for tracksys in TRACKING_SYSTEMS: 
                        file_path = f"{DATA_PATH}/rawdata/sub-{sub_id}/motion/sub-{sub_id}_task-{task_name}_tracksys-{tracksys}_motion.tsv"

                        try:
                            data = data_loader.load_file(file_path)
                            # trim, reduce dimensions, interpolate, rotate, differentiate
                            preprocessed_data = preprocessing.all_preprocessing(data, sub_id, task_name, run, tracksys, kinematics, FS)

                            calculate_kinectome(preprocessed_data, sub_id, task_name, run, tracksys, kinematics, DATA_PATH, MARKER_LIST,linux=True)


                        except Exception as e:
                            print(f"Error loading file {file_path}: {e}")

    # load the kinectomes and calculate the centrality measures

    for sub_id in sub_ids:
        for kinematics in KINEMATICS:
            for task_name in TASK_NAMES:
                for tracksys in TRACKING_SYSTEMS:
                        # load the kinectomes
                        kinectomes = data_loader.load_kinectomes(DATA_PATH, sub_id, task_name,tracksys,run,kinematics)
                        print(f"{kinematics},{task_name}: \n\n Number of events is {len(kinectomes)}")

                        # direction dict, as order of build_graph
                        directions_dict = {"AP": [], "ML": [], "V":[]} 
                        for k in kinectomes:
                            graphs = build_graph(k,MARKER_LIST)
                            for idx, direction in enumerate(["AP", "ML", "V"]):
                                G = graphs[idx]
                                # calculate the centrality measures
                                centrality_dict = weighted_degree_centrality(G)
                                directions_dict[direction].append(centrality_dict)

                        for idx in ["AP", "ML", "V"]:
                            merged_ = data_loader.merge_dicts(directions_dict[idx])
                            fig, ax = plt.subplots()
                            ax.boxplot(merged_.values(), vert=False, showfliers=False)
                            ax.set_yticklabels(merged_.keys())
                            ax.set_xlabel("Centrality")
                            ax.set_ylabel("Markers")
                            ax.set_title(f"{kinematics} {task_name} {idx}: Centrality")
                            save_path = f"{DATA_PATH}/plots"
                            # Ensure directory exists
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            fig.savefig(f"{save_path}/{sub_id}_{kinematics}_{task_name}_{idx}_centrality.png")
