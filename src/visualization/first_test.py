"""
Simple vizualization of the motion of a person using pyvista.
The script reads the motion data of a person and creates a 3D visualization of the person's motion.
The script also uses the IMU data to create a weighted graph of the person's body parts and label the points according to the community they belong to.
TODO- The labels are not being so nicely plotted, they are overlapping as time of Task goes on 
The script saves the visualization as a video file.
"""
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib import colormaps
import networkx as nx


def create_body_part_mesh(x, y, z, radius=5):
    return pv.Sphere(radius=radius, center=(x, y, z))


def create_skeleton_line(x1, y1, z1, x2, y2, z2):
    return pv.Line((x1, y1, z1), (x2, y2, z2))


PATH_MOTION = "./sub-pp002/motion/sub-pp002_task-choreo_tracksys-omc_motion.tsv"
PATH_CHANNELS = "./sub-pp002/motion/sub-pp002_task-choreo_tracksys-omc_channels.tsv"

df = pd.read_csv(
    PATH_MOTION, sep='\t')

# TODO Make it more fleixble and create a checker if the number is valid. 
num_frames = len(df) # either the entire task or number of wanted frames.  

tracked_points = pd.read_csv(
    PATH_CHANNELS, sep='\t')

tracked_points = tracked_points['tracked_point'].unique()


sensors_ts = {}

tracked_points = [
    p for p in tracked_points if "start" not in p and "end" not in p]

for t in tracked_points:
    cols_ = [col for col in df.columns if t in col and "_err" not in col]
    ts_ = df[cols_].to_numpy()
    sensors_ts[t] = ts_



skeleton = [
    # head
    ["lf_hd", "rf_hd"],
    ["lb_hd", "rb_hd"],
    ["lf_hd", "lb_hd"],
    ["rf_hd", "rb_hd"],

    # head to sternum
    ["lf_hd", "m_ster1"],
    ["rf_hd", "m_ster1"],
    ["lb_hd", "m_ster1"],
    ["rb_hd", "m_ster1"],

    ["lf_hd", "m_ster2"],
    ["rf_hd", "m_ster2"],
    ["lb_hd", "m_ster2"],
    ["rb_hd", "m_ster2"],

    ["lf_hd", "m_ster3"],
    ["rf_hd", "m_ster3"],
    ["lb_hd", "m_ster3"],
    ["rb_hd", "m_ster3"],

    # triangle in the sternum
    ["m_ster1", "m_ster2"],
    ["m_ster2", "m_ster3"],
    ["m_ster3", "m_ster1"],

    # sternum to shoulders and arms
    # left
    ["m_ster1", "l_sho"],
    ["m_ster2", "l_sho"],
    ["m_ster3", "l_sho"],
    ["l_sho", "l_ua"],
    ["l_ua", "l_elbl"],
    ["l_elbl", "l_frm"],
    ["l_frm", "l_wrr"],
    ["l_wrr", "l_wru"],
    ["l_wru", "l_hand"],

    # right
    ["m_ster1", "r_sho"],
    ["m_ster2", "r_sho"],
    ["m_ster3", "r_sho"],
    ["r_sho", "r_ua"],
    ["r_ua", "r_elbl"],
    ["r_elbl", "r_frm"],
    ["r_frm", "r_wrr"],
    ["r_wrr", "r_wru"],
    ["r_wru", "r_hand"],

    # sternum to pelvis
    ["m_ster2", "l_asis"],
    ["m_ster2", "l_psis"],


    ["m_ster2", "r_asis"],
    ["m_ster2", "r_psis"],

    ["m_ster3", "l_asis"],
    ["m_ster3", "l_psis"],

    ["m_ster3", "r_asis"],
    ["m_ster3", "r_psis"],

    ["l_asis", "l_psis"],
    ["r_asis", "r_psis"],
    ["l_asis", "r_asis"],
    ["l_psis", "r_psis"],

    # left leg
    ["l_asis", "l_th1"],
    ["l_psis", "l_th1"],
    ["l_asis", "l_th2"],
    ["l_psis", "l_th2"],

    ["l_th1", "l_th2"],
    ["l_th1", "l_th3"],
    ["l_th2", "l_th3"],
    ["l_th2", "l_th4"],
    ["l_th3", "l_th4"],
    ["l_th1", "l_th4"],

    ["l_th4", "l_sk2"],
    ["l_th3", "l_sk1"],
    ["l_th4", "l_sk2"],
    ["l_th3", "l_sk1"],

    ["l_sk1", "l_sk2"],
    ["l_sk1", "l_sk3"],
    ["l_sk1", "l_sk4"],
    ["l_sk2", "l_sk3"],
    ["l_sk2", "l_sk4"],
    ["l_sk3", "l_sk4"],
    ["l_sk2", "l_sk4"],
    ["l_sk3", "l_ank"],
    ["l_sk4", "l_ank"],
    ["l_ank", "l_heel"],
    ["l_heel", "l_toe"],


    # right leg
    ["r_asis", "r_th1"],
    ["r_psis", "r_th1"],
    ["r_asis", "r_th2"],
    ["r_psis", "r_th2"],

    ["r_th1", "r_th2"],
    ["r_th1", "r_th3"],
    ["r_th1", "r_th4"],
    ["r_th2", "r_th3"],
    ["r_th2", "r_th4"],
    ["r_th3", "r_th4"],


    ["r_th4", "r_sk2"],
    ["r_th3", "r_sk1"],
    ["r_th4", "r_sk2"],
    ["r_th3", "r_sk1"],

    ["r_sk1", "r_sk2"],
    ["r_sk1", "r_sk3"],
    ["r_sk1", "r_sk4"],
    ["r_sk2", "r_sk3"],
    ["r_sk2", "r_sk4"],
    ["r_sk3", "r_sk4"],

    ["r_sk3", "r_ank"],
    ["r_sk4", "r_ank"],
    ["r_ank", "r_heel"],
    ["r_heel", "r_toe"],

]


####
# INERTIA DATAAA
##########
df_iner = pd.read_csv(
    "./sub-pp002/motion/sub-pp002_task-choreo_tracksys-imu_motion.tsv", sep='\t')

tracked_points_iner = pd.read_csv(
    './sub-pp002/motion/sub-pp002_task-choreo_tracksys-imu_channels.tsv', sep='\t')

tracked_points_iner = tracked_points_iner['tracked_point'].unique()


acc_mag = {}

for point in tracked_points_iner:
    acc_mag[f"{point}_ACC_mag"] = np.sqrt(
        df_iner[f'{point}_ACC_x']**2 + df_iner[f'{point}_ACC_y']**2 + df_iner[f'{point}_ACC_z']**2)


def weighted_graph(acc_mag, t_stamp):
    G = nx.Graph()
    keys = list(acc_mag.keys())
    w_ = []
    for name in keys:
        G.add_node(name)
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            w = acc_mag[keys[i]].iloc[t_stamp] - acc_mag[keys[j]].iloc[t_stamp]
            w_.append(w)
            G.add_edge(keys[i], keys[j], weight=w)
    w_ = np.mean(w_)
    return G, w_


initial_graph, _ = weighted_graph(acc_mag, 0)


######################
# end of inertia
#################


# Initialize the plotter
plotter = pv.Plotter(notebook=False, off_screen=True)
plotter.open_movie("task_backwards_with_imu_2.mp4")

body_part_meshes = []
skeleton_line_meshes = []


colormap = colormaps.get_cmap("viridis")


def normalize(val): return (val - 0) / (1 - 0)


body_part_colors = [colormap(i) for i in range(len(tracked_points))]

body_parts = tracked_points

# Create initial meshes
for body_part, color in zip(tracked_points, body_part_colors):
    x = df.loc[0, f'{body_part}_POS_x']
    y = df.loc[0, f'{body_part}_POS_y']
    z = df.loc[0, f'{body_part}_POS_z']
    mesh = create_body_part_mesh(x, y, z)
    plotter.add_mesh(mesh, color=color)
    body_part_meshes.append(mesh)

for connection in skeleton:
    part1 = connection[0]
    part2 = connection[1]
    x1, y1, z1 = df.loc[0, f'{part1}_POS_x'], df.loc[0,
                                                     f'{part1}_POS_y'], df.loc[0, f'{part1}_POS_z']
    x2, y2, z2 = df.loc[0, f'{part2}_POS_x'], df.loc[0,
                                                     f'{part2}_POS_y'], df.loc[0, f'{part2}_POS_z']
    skl_mesh = create_skeleton_line(x1, y1, z1, x2, y2, z2)
    plotter.add_mesh(skl_mesh, color="black", line_width=2)
    skeleton_line_meshes.append(skl_mesh)

# Loop over each frame
for num in range(14500, num_frames):  # start from 1 as we already plotted the first frame
    print(f"Processing frame {num}/{num_frames}")
    # Update each body part
    mean_position = np.array([0.0, 0.0, 0.0])

    ###
    # graphhhh
    ####

    initial_graph, weight = weighted_graph(acc_mag, num)

    communities = nx.community.louvain_communities(
        initial_graph, weight="weight", threshold=weight)

    mod_group = {}

    for i, n in enumerate(communities):
        for el in n:
            mod_group[el] = f"C_{i}"

    ######
    # end of grpah
    #####

    for body_part, mesh in zip(body_parts, body_part_meshes):
        x = df.loc[num, f'{body_part}_POS_x']
        y = df.loc[num, f'{body_part}_POS_y']
        z = df.loc[num, f'{body_part}_POS_z']
        # Update points of existing mesh
        mesh.points = pv.Sphere(radius=5, center=(x, y, z)).points
        ###
        ## TODO - Ajust the labels problems.
        ####
        if body_part == "rf_hd":  # head_ACC_mag
            label_point = mesh.points[0]
            label = pv.Label(mod_group['head_ACC_mag'],
                             position=label_point, size=9)
            plotter.add_actor(label)

        elif body_part == "l_frm":
            label_point = mesh.points[0]
            label = pv.Label(
                mod_group['left_fore_arm_ACC_mag'], position=label_point, size=9)
            plotter.add_actor(label)
        # 'left_fore_arm_ACC_mag': 'C_0',

        elif body_part == "l_th1":
            label_point = mesh.points[0]
            label = pv.Label(
                mod_group['left_thigh_ACC_mag'], position=label_point, size=9)
            plotter.add_actor(label)
        # 'left_thigh_ACC_mag': 'C_0',

        elif body_part == "m_ster2":
            label_point = mesh.points[0]
            label = pv.Label(mod_group['sternum_ACC_mag'],
                             position=label_point, size=9)
            plotter.add_actor(label)
        # 'sternum_ACC_mag': 'C_0',

        elif body_part == "l_asis" or body_part == "r_asis":
            label_point = mesh.points[0]
            label = pv.Label(mod_group['pelvis_ACC_mag'],
                             position=label_point, size=9)
            plotter.add_actor(label)
        # 'pelvis_ACC_mag': 'C_0',

        elif body_part == "r_heel":
            label_point = mesh.points[0]
            label = pv.Label(
                mod_group['right_foot_ACC_mag'], position=label_point, size=9)
            plotter.add_actor(label)
        # 'right_foot_ACC_mag': 'C_0',

        elif body_part == "r_ua":
            label_point = mesh.points[0]
            label = pv.Label(
                mod_group['right_upper_arm_ACC_mag'], position=label_point, size=9)
            plotter.add_actor(label)
        # 'right_upper_arm_ACC_mag': 'C_0',

        elif body_part == "l_sk2":
            label_point = mesh.points[0]
            label = pv.Label(
                mod_group['left_shank_ACC_mag'], position=label_point, size=9)
            plotter.add_actor(label)
        # 'left_shank_ACC_mag': 'C_0',

        elif body_part == "r_frm":
            label_point = mesh.points[0]
            label = pv.Label(
                mod_group['right_fore_arm_ACC_mag'], position=label_point, size=9)
            plotter.add_actor(label)
        # 'right_fore_arm_ACC_mag': 'C_0',

        elif body_part == "l_ank":
            label_point = mesh.points[0]
            label = pv.Label(
                mod_group['left_ankle_ACC_mag'], position=label_point, size=9)
            plotter.add_actor(label)
        # 'left_ankle_ACC_mag': 'C_1',

        elif body_part == "l_heel":
            label_point = mesh.points[0]
            label = pv.Label(
                mod_group['left_foot_ACC_mag'], position=label_point, size=9)
            plotter.add_actor(label)
        # 'left_foot_ACC_mag': 'C_1',

        elif body_part == "r_ank":
            label_point = mesh.points[0]
            label = pv.Label(
                mod_group['right_ankle_ACC_mag'], position=label_point, size=9)
            plotter.add_actor(label)
        # 'right_ankle_ACC_mag': 'C_1',

        elif body_part == "l_ua":
            label_point = mesh.points[0]
            label = pv.Label(
                mod_group['left_upper_arm_ACC_mag'], position=label_point, size=9)
            plotter.add_actor(label)
        # 'left_upper_arm_ACC_mag': 'C_1',

        elif body_part == "r_th1":
            label_point = mesh.points[0]
            label = pv.Label(
                mod_group['right_thigh_ACC_mag'], position=label_point, size=9)
            plotter.add_actor(label)
        # 'right_thigh_ACC_mag': 'C_1',

        elif body_part == "r_sk2":
            label_point = mesh.points[0]
            label = pv.Label(
                mod_group['right_shank_ACC_mag'], position=label_point, size=9)
            plotter.add_actor(label)
        # 'right_shank_ACC_mag': 'C_1'
        #####
        mean_position += np.array([x, y, z])

    # # Calculate mean position of all body parts
    mean_position /= len(body_parts)
    plotter.camera.focal_point = mean_position.tolist()  # Update the camera focal point

    # Update each skeleton connection
    for connection, mesh in zip(skeleton, skeleton_line_meshes):
        part1 = connection[0]
        part2 = connection[1]
        x1, y1, z1 = df.loc[num, f'{part1}_POS_x'], df.loc[num,
                                                           f'{part1}_POS_y'], df.loc[num, f'{part1}_POS_z']
        x2, y2, z2 = df.loc[num, f'{part2}_POS_x'], df.loc[num,
                                                           f'{part2}_POS_y'], df.loc[num, f'{part2}_POS_z']
        # Update points of existing mesh
        mesh.points = pv.Line((x1, y1, z1), (x2, y2, z2)).points

    plotter.write_frame()  # Write the current frame
plotter.close()
