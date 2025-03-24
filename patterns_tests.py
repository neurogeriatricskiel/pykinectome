
# Description: This script is used to test pattern extraction in the kinectome data
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import networkx as nx
from src.data_utils import data_loader
from src.kinectome import *
from src.preprocessing import preprocessing
from src.graph_utils.kinectome2graph import build_graph, clustering_coef, cc_connected_components
from src.graph_utils.kinectome2pattern import full_subgraph, path_subgraph, cycle_subgraph
from tqdm import tqdm



if __name__ == "__main__":
    ##
    # LOADING CONFIG CONSTANTS
    ##
    # DATA-PATH
    DATA_PATH = "/home/prdc/Dokumente/Projects-KI/uksh"
    # Seleted tasks
    TASK_NAMES = [
    "walkSlow"
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
    MARKER_SUB_LIST = ['head', 'ster',  'r_sho',  
                 'r_elbl', 'r_wrist', 'r_hand']
    run = None # dont have id "run" in sample data that I have 
    FS = 200 # sampling rate
    sub_ids = ["pp002"] 

    for sub_id in sub_ids:
        for kinematics in KINEMATICS:
            for task_name in TASK_NAMES:
                for tracksys in TRACKING_SYSTEMS:
                    kinectomes = data_loader.load_kinectomes(DATA_PATH, sub_id, task_name,tracksys,run,kinematics)   
                    for k in kinectomes:
                        G_directions = build_graph(k, MARKER_LIST) # list of graphs
                        directions_iterator = tqdm(G_directions, desc=f"---Subject: {sub_id}, Task: {task_name}---")
                        for i,g in enumerate(directions_iterator):
                            # full subgraph
                            # subgraph = full_subgraph(g, MARKER_SUB_LIST)
                            # path subgraph
                            # subgraph = path_subgraph(g, MARKER_SUB_LIST)
                            # cycle subgraph
                            subgraph = cycle_subgraph(g, MARKER_SUB_LIST)

