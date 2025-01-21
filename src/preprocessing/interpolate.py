import kineticstoolkit as ktk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fill_gaps(data):

    # for col in data.columns():
        # column = data[col]

    markers = ktk.read_c3d(
                ktk.doc.download("kinematics_basket_sprint.c3d")
            )["Points"]


    return data