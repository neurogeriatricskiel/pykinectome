# Description: This script is used to calculate the clustering coefficient of the kinectome
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import networkx as nx
from src.data_utils import data_loader
from src.kinectome import *
from src.preprocessing import preprocessing
from src.data_utils.kinecome2graph import build_graph, clustering_coef, cc_connected_components
from tqdm import tqdm


def plot_cc(DATA_PATH,sub_id,task_name,tracksys,run,kinematics,MARKER_LIST,threshold_list=[0.2,0.4,0.6,0.8]):
    fig, axs = plt.subplots(3,len(threshold_list), figsize=(15, 15))
    # load the kinectomes
    kinectomes = data_loader.load_kinectomes(DATA_PATH, sub_id, task_name,tracksys,run,kinematics)
    print(f"{kinematics},{task_name}: \n\n Number of events is {len(kinectomes)}")

    for i, limit in enumerate(threshold_list):
        # direction dict, as order of build_graph
        directions_dict = {"AP": [], "ML": [], "V":[]} 
        for k in kinectomes:
            graphs = build_graph(k,MARKER_LIST,limit)
            for idx, direction in enumerate(["AP", "ML", "V"]):
                G = graphs[idx]
                # calculate the clustering coef 
                cc_dict = clustering_coef(G)
                directions_dict[direction].append(cc_dict)

        for j,idx in enumerate(["AP", "ML", "V"]):
            merged_ = data_loader.merge_dicts(directions_dict[idx])
            axs[j,i].boxplot(merged_.values(), vert=False, showfliers=False)
            axs[j,i].set_yticklabels(merged_.keys())
            axs[j,i].set_xlabel("clustering coefficient")
            axs[j,i].set_ylabel("Markers")
            axs[j,i].set_title(f"{idx}_{limit}") 
    for ax in axs.flat:
        ax.label_outer()
    save_path = f"{DATA_PATH}/plots"
    # Ensure directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.suptitle(f"{kinematics} {task_name} {sub_id} ") 

    fig.savefig(f"{save_path}/{sub_id}_{kinematics}_{task_name}-cc.png")



def event_plot_cc(DATA_PATH,sub_id,task_name,tracksys,run,kinematics,MARKER_LIST,threshold_list=[0.2,0.4,0.6,0.8],direction=0):
    if direction == 0:
        idx = "AP"
    elif direction == 1:
        idx = "ML"
    elif direction == 2:
        idx = "V"
    # load the kinectomes
    kinectomes = data_loader.load_kinectomes(DATA_PATH, sub_id, task_name,tracksys,run,kinematics)
    # print(f"{kinematics},{task_name}: \n\n Number of events is {len(kinectomes)}")
    fig = plt.figure(figsize=(8, 8))
    cmap = mpl.cm.get_cmap("Spectral")
    events_iterator = tqdm(threshold_list, desc=f"---Subject: {kinematics}, Direction: {idx}, Task: {task_name}---")
    ax = None
    for n, limit in enumerate(events_iterator):
        ax = plt.subplot(1,len(threshold_list),n+1, frameon=False, sharex=ax)
        directions_dict = {idx: []}
        for k in kinectomes:
            graphs = build_graph(k,MARKER_LIST,limit)
            G = graphs[direction]
            # calculate the clustering coef 
            cc_dict = clustering_coef(G)
            directions_dict[idx].append(cc_dict)
        
        merged_ = data_loader.merge_dicts(directions_dict[idx])
        for i, k in enumerate(merged_.keys()):
            Y = np.array(merged_[k])
            X = np.arange(len(Y))
            ax.plot(X,Y+i,color="k",zorder=100-i)
            color = cmap(i / 22)
            ax.fill_between(X,Y + i, i, color=color, zorder=100 - i)
        
        if n == 0:
            ax.yaxis.set_tick_params(labelleft=True)
            ax.set_yticks(np.arange(len(merged_.keys())))
            ax.set_yticklabels([f"{k}" for k in merged_.keys()],verticalalignment="bottom")
        else:
            ax.yaxis.set_tick_params(labelleft=False)

        ax.text(
        0.0,
        1.0,
        f"Threshold {limit}",
        ha="left",
        va="top",
        weight="bold",
        transform=ax.transAxes,
        )
    plt.tight_layout()
    save_path = f"{DATA_PATH}/plots"
    # Ensure directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig.savefig(f"{save_path}/{sub_id}_{kinematics}_{idx}_{task_name}-curve_event.png")


def event_plot_components(DATA_PATH,sub_id,task_name,tracksys,run,kinematics,MARKER_LIST,threshold_list=[0.2,0.4,0.6,0.8],direction=1):
    if direction == 0:
        idx = "AP"
    elif direction == 1:
        idx = "ML"
    elif direction == 2:
        idx = "V"
    # load the kinectomes
    kinectomes = data_loader.load_kinectomes(DATA_PATH, sub_id, task_name,tracksys,run,kinematics)
    # print(f"{kinematics},{task_name}: \n\n Number of events is {len(kinectomes)}")
    fig = plt.figure(figsize=(8, 8))
    cmap = mpl.cm.get_cmap("Spectral")
    events_iterator = tqdm(kinectomes, desc=f"---Subject: {kinematics}, Direction: {idx}, Task: {task_name}---")
    ax = None
    for n, k in enumerate(events_iterator):
        ax = plt.subplot(1,len(kinectomes),n+1, frameon=False, sharex=ax)
        directions_dict = {idx: []}
        for j, limit in enumerate(threshold_list):
            graphs = build_graph(k,MARKER_LIST,limit)
            G = graphs[direction]
            # calculate the clustering coef 
            cc = cc_connected_components(G)
            directions_dict[idx].append(len(cc))
        
        # merged_ = data_loader.merge_dicts(directions_dict[idx])
        for i, k in enumerate(directions_dict.keys()):
            Y = np.array(directions_dict[k])
            X = np.array(threshold_list)
            ax.plot(X,Y+i,color="k",zorder=100-i)
            color = cmap(i / 22)
            ax.fill_between(X,Y + i, i, color=color, zorder=100 - i)
        
        if n == 0:
            ax.yaxis.set_tick_params(labelleft=True)
            # ax.set_yticks(np.arange(len(kinectomes)))
            ax.set_yticks(np.arange(10))
            # ax.set_yticklabels([f"Event {n}" for n in range(1,len(kinectomes) + 1 )],verticalalignment="bottom")
        else:
            ax.yaxis.set_tick_params(labelleft=False)

        ax.text(
        0.0,
        1.0,
        f"Event {n}",
        ha="left",
        va="top",
        weight="bold",
        transform=ax.transAxes,
        )
    plt.tight_layout()
    save_path = f"{DATA_PATH}/plots"
    # Ensure directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig.savefig(f"{save_path}/{sub_id}_{kinematics}_{idx}_{task_name}-connected-components.png")


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
                    plot_cc(DATA_PATH,sub_id,task_name,tracksys,run,kinematics,MARKER_LIST,threshold_list)
                    for direction in [0,1,2]:
                        event_plot_cc(DATA_PATH,sub_id,task_name,tracksys,run,kinematics,MARKER_LIST,threshold_list,direction)
                        event_plot_components(DATA_PATH,sub_id,task_name,tracksys,run,kinematics,MARKER_LIST,threshold_list,direction)
