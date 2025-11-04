import sys
import os
import json
import tifffile
import importlib
import numpy as np

import seaborn as sns
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from collections import defaultdict

from sklearn.cluster import k_means
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from matplotlib.patches import Polygon
from sknetwork.clustering import Leiden
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances

# widget libraries
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QEventLoop

from anno_class import annotation_loop


matplotlib.use('QtAgg')


class core_data:
    def __init__(self, expression_data, image, segments, core, marker_info):

        # expression_subset = expression_data[expression_data['core'] == core]
        self.expression_data = expression_data
        self.image = image
        self.segments = segments 
        self.marker_info = marker_info
        self.pc_data = []
        self.features = []
        self.scaled_features = []
        self.sampled_cells = []
        self.annotations = []
        self.features_selected = False
        self.plot_df = self.expression_data[['X_coord', 'Y_coord']]
        self.manual_samples = defaultdict()

    def select_features(self, feats):
        valid_columns = [col for col in feats if col in self.expression_data.columns]
        self.features = valid_columns
        if len(self.features) > 0:
            self.features_selected = True
        
        scaled_features = self.expression_data[self.features]
        self.scaled_features = StandardScaler().fit_transform(scaled_features)
        # print(f"Selected features: {self.features}") # for debugging

    def run_pca(self):
        if not self.features_selected:
            print("Please run select_features() first.")
            return

        print("Running PCA...")
        pca = PCA(n_components = 10)
        pca.fit(self.scaled_features)
        x = pca.transform(self.scaled_features)

        self.pc_data = x
        self.plot_df.loc[:, ['PC1', 'PC2']] = x[:, :2]
        
    def run_kmeans(self, num_clust, random_state, pre_split = False):
        centroid, label, intertia = k_means(X = self.scaled_features,
                                            n_clusters = num_clust,
                                            random_state = random_state)

        self.plot_df.loc[:,[f'kmeans_{num_clust}']] = label.tolist()
       
    def run_gmm(self, n_components, random_state):
        
        gmm = GaussianMixture(n_components = n_components, 
                              random_state = random_state)
        
        gmm.fit(self.scaled_features)
        gmm_labels = gmm.predict(self.scaled_features)
        probabilities = gmm.predict_proba(self.scaled_features)
        
        self.plot_df.loc[:, ['gmm']] = gmm_labels

    def run_leiden(self, resolution, random_state, knn = 15):

        k = knn  # Choose an appropriate number of neighbors
        adjacency_matrix = kneighbors_graph(self.scaled_features, k, 
                                            mode = 'connectivity', include_self = False)
        leiden = Leiden(resolution = resolution, random_state = random_state)
        labels = leiden.fit_predict(adjacency_matrix)
        
        self.plot_df.loc[:,[f'leiden_{resolution}']] = labels


    def filter_border_polygons(self, tolerance = 1):
        """
        Filter out polygons that have more than one point touching or near the image borders.
        
        :param polygons: List of polygon coordinates [(x1, y1), (x2, y2), ...]
        :param image_shape: Tuple of (height, width) of the image
        :param tolerance: Distance from border to consider as touching (default 1 pixel)
        :return: List of filtered polygons and list of indices of kept polygons
        """
        height, width = self.image.shape[1:]
        
        filtered_segments = {}
        kept_keys = []
        
        for key, segment in self.segments.items():
            x_values, y_values = segment
            border_points = 0
            for x, y in zip(x_values, y_values):
                y_inverted = np.abs(y)
                if (x <= tolerance or x >= width - tolerance or 
                    y_inverted <= tolerance or y_inverted >= height - tolerance):
                    border_points += 1
            
            if border_points <= 1:
                filtered_segments[key] = segment
                kept_keys.append(key)
        
        return filtered_segments, kept_keys

    
    def farthest_point_sampling(self, data, n_samples):
        """
        Perform Farthest Point Sampling on the given data.
        """
        data_matrix = data if isinstance(data, np.ndarray) else data.values
        n_points = data_matrix.shape[0]

        if n_samples >= n_points:
            return list(range(n_points))

        first_idx = np.random.randint(n_points)
        indices = [first_idx]
        distances = np.full(n_points, np.inf)

        for _ in range(1, n_samples):
            last_idx = indices[-1]
            dist = distance.cdist([data_matrix[last_idx]], data_matrix)[0]
            distances = np.minimum(distances, dist)
            indices.append(np.argmax(distances))

        return indices

    def cell_sampler(self, cluster_col, max_sample=15, random_state=5, keep_oob=False, tolerance=1, use_fps=False):
        sample_q = []
        np.random.seed(random_state)

        if not keep_oob:
            filtered_segments, kept_keys = self.filter_border_polygons(tolerance)
            filtered_plot_df = self.plot_df.loc[self.plot_df.index.isin(kept_keys)].sort_index()
        else:
            filtered_plot_df = self.plot_df
            
        for cluster in filtered_plot_df[cluster_col].unique():
            this_sset = filtered_plot_df[filtered_plot_df[cluster_col] == cluster]
            sample_size = min(this_sset.shape[0], max_sample)
            
            if use_fps:
                # Correctly access the scaled features
                # Get indices from this_sset
                index_list = this_sset.index.tolist()

                # Find which indices to take from self.scaled_features
                indices = [self.plot_df.index.get_loc(idx) for idx in index_list]

                # Apply fps to the indices within this_sset
                fps_data = self.scaled_features[indices]

                # Call fps to get the proper subset of the data
                sampled_indices = self.farthest_point_sampling(fps_data, sample_size)

                # Get the OBJECT IDS
                ids = this_sset.iloc[sampled_indices].index.tolist()

            else:
                ids = this_sset.sample(sample_size, random_state=random_state).index.tolist()
            
            sample_q.extend(ids)

        self.sampled_cells = sample_q
            
    def approximate_bounds(self):
        vmin_vmax_dict = {}
        for marker, (index, _, _) in self.marker_info.items():
            marker_channel = self.image[index]
            vmin = float(np.percentile(marker_channel, 5))
            vmax = float(np.percentile(marker_channel, 95))

            vmax = max(20, vmax) # offsets extreme low expression (rare cell types)
            vmin_vmax_dict[marker] = [index, vmin, vmax]

        self.marker_info = vmin_vmax_dict

    def plot_marker(self, marker):
        plt.figure(figsize = (3.5,3.5))
        values = self.marker_info[marker]
        plt.imshow(self.image[values[0], :, :], 
                   vmin = values[1], 
                   vmax = values[2],
                   cmap = "magma")
        plt.title(marker)
        plt.show()

    def annotate(self, show_markers, cell_types = None):
        annotation_loop(self, show_markers, cell_types = cell_types)

    def manual_sample(self, marker, lower_percentile, upper_percentile, 
                      n, showplot = True, random_state = 5):
        
        # Calculate the actual values corresponding to the percentiles
        lower_bound = np.percentile(self.expression_data[marker], lower_percentile)
        upper_bound = np.percentile(self.expression_data[marker], upper_percentile)
            
        # Filter the DataFrame based on the calculated bounds
        filtered_df = self.expression_data[(self.expression_data[marker] >= lower_bound) & 
        (self.expression_data[marker] <= upper_bound)]

        if showplot:
            plt.figure(figsize = (4, 2))
            plt.hist(self.expression_data[marker], 
                     color = 'lightblue', 
                     edgecolor = 'black',
                     linewidth = 0.5,
                     bins = 50)
            plt.axvline(x = lower_bound, linestyle = '--', color = 'red')
            plt.axvline(x = upper_bound, linestyle = '--', color = 'red')
            plt.title(f'{filtered_df.shape[0]} cells')
            plt.show()
        
        # Sample n rows if there are more than n rows after filtering
        if len(filtered_df) > n:
            ids = filtered_df.sample(n, random_state = random_state).index.to_list()
        else:
            ids = filtered_df.index.to_list()

        self.manual_samples[marker] = ids

    
