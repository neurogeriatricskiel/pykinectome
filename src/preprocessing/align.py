from sklearn.decomposition import PCA
import pandas as pd

# Use PCA for aligning marker data with main walking direction

def pca(data: pd.DataFrame):
    ''' Runs principal component analysis.
    Returns a rotation matrix'''
    # apply principal component analysis
    pca = PCA(n_components=2)
    pca.fit(X=data)

    # get the rotation matrix
    rotation_matrix = pca.components_

    return rotation_matrix