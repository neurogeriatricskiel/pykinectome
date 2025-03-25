# Description: This script is used to calculate the clustering coefficient of the kinectome
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import networkx as nx
from src.data_utils import data_loader
from src.kinectome import *
from src.preprocessing import preprocessing
from src.graph_utils.kinectome2graph import build_graph, clustering_coef, cc_connected_components
from src.data_utils.plotting import plot_cc, event_plot_cc, event_plot_components
from tqdm import tqdm





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
    sub_ids = ["pp002"] 

    threshold_list = [0.2,0.4,0.6,0.8]

    for sub_id in sub_ids:
        for kinematics in KINEMATICS:
            for task_name in TASK_NAMES:
                for tracksys in TRACKING_SYSTEMS:
                    # plot_cc(DATA_PATH,sub_id,task_name,tracksys,run,kinematics,MARKER_LIST,threshold_list)
                    for direction in [0,1,2]:
                        event_plot_cc(DATA_PATH,sub_id,task_name,tracksys,run,kinematics,MARKER_LIST,threshold_list,direction)
                        # event_plot_components(DATA_PATH,sub_id,task_name,tracksys,run,kinematics,MARKER_LIST,threshold_list,direction)

