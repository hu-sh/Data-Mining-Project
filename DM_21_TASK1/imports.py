import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from IPython.display import display

from collections import defaultdict
from scipy.stats import pearsonr

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

from sklearn.cluster import SpectralClustering, AgglomerativeClustering, OPTICS, DBSCAN, KMeans
from sklearn.cluster import ward_tree
import hdbscan

from scipy.cluster.hierarchy import dendrogram
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

from sklearn.model_selection import ParameterSampler

import pickle

audio_cols = [
            'bpm',
            'centroid',
            'rolloff',
            'flux',
            'zcr',
            'flatness',
            'spectral_complexity',
            'pitch',
            'loudness'
            ]
# we already remove rms, it wil be removed anyways

lyrical_cols = ["n_tokens", 
                "n_sentences", 
                "swear_IT", 
                "swear_EN", 
                "tokens_per_sent", 
                "char_per_tok", 
                "avg_token_per_clause",
                "lexical_density",
                "swear_density_IT",
                "swear_density_EN",
                "wps"
               ]

clust_cols = audio_cols + ["wps", "lexical_density", "avg_token_per_clause", "tokens_per_sent", "swear_density_IT"]
# "char_per_tok"?

cols_umap3 = [f"umap3_{i+1}" for i in range(3)] 
cols_umap5 = [f"umap5_{i+1}" for i in range(5)]
cols_umap8 = [f"umap8_{i+1}" for i in range(8)] 

def plot_xy(df, col1, col2):
    x = df[col1]
    y = df[col2]
    plt.figure(figsize=(8,5))
    plt.scatter(x, y)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.grid(True)
    plt.show()

def plot_hist(df, col, bins='fd'):
    plt.hist(df[col], bins, edgecolor='black')  
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def calc_pearson(df, col1, col2):
    x = df[col1]
    y = df[col2]
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]

    r, p = stats.pearsonr(x_clean, y_clean)
    return [r, p]


def find_elbow_point(distances):
    n_points = len(distances)
    
    P_start = np.array([0, distances[0]])
    P_end = np.array([n_points - 1, distances[-1]])
    
    chord_vector = P_end - P_start
    
    chord_norm = np.linalg.norm(chord_vector)
    
    max_distance = -1
    elbow_index = -1

    for i in range(1, n_points - 1):
        P_i = np.array([i, distances[i]])
        
        point_vector = P_i - P_start
        
        projection_length = np.dot(point_vector, chord_vector) / chord_norm
        
        projected_vector = projection_length * (chord_vector / chord_norm)
        
        perpendicular_vector = point_vector - projected_vector
        
        d_i = np.linalg.norm(perpendicular_vector)
        
        if d_i > max_distance:
            max_distance = d_i
            elbow_index = i

    return distances[elbow_index], elbow_index

__all__ = [
    # Librerie e Moduli (alias)
    "math", "np", "pd", "stats", "plt", "px", "sns", "display", "hdbscan", "pickle",
    
    # Utilità e Statistica
    "defaultdict", "pearsonr", "dendrogram", "ward_tree",
    
    # Preprocessing e Trasformazioni
    "StandardScaler", "MinMaxScaler", "PowerTransformer", "PCA",
    
    # Modelli di Clustering e Regressione
    "LinearRegression", "KMeans", "DBSCAN", "OPTICS", "SpectralClustering", 
    "AgglomerativeClustering", "NearestNeighbors", "ParameterSampler", "metrics",
    
    # Costanti e Liste di Colonne
    "audio_cols", "lyrical_cols", "clust_cols", "cols_umap3", "cols_umap5", "cols_umap8",
    
    # Funzioni personalizzate
    "plot_xy", "plot_hist", "calc_pearson", "find_elbow_point"
]