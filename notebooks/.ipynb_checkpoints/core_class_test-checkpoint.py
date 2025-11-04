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
from annotation_utils import softmax, minmax

matplotlib.use('QtAgg')

class core_data:
    def __init__(self, expression_data, image, segments, core, marker_info):
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
        self.plot_df = self.expression_data[['X_coord', 'Y_coord']].copy()
        self.manual_samples = defaultdict()

        # Initialize new attributes for lineage and clustering
        self.lineage_assignments = pd.Series(dtype=str)
        self.lineage_subsets = {}
        self.subclusters = {}
        self.lineage_softmax = pd.DataFrame()

    def select_features(self, feats):
        valid_columns = [col for col in feats if col in self.expression_data.columns]
        self.features = valid_columns
        if len(self.features) > 0:
            self.features_selected = True

        scaled_features = self.expression_data[self.features]
        self.scaled_features = StandardScaler().fit_transform(scaled_features)

    def run_pca(self, n_dim = 10):
        if not self.features_selected:
            print("Please run select_features() first.")
            return

        print("Running PCA...")
        pca = PCA(n_components = n_dim)
        pca.fit(self.scaled_features)
        x = pca.transform(self.scaled_features)

        self.pc_data = x
        self.plot_df.loc[:, ['PC1', 'PC2']] = x[:, :2]

    def run_kmeans(self, num_clust, random_state, pre_split = False):
        centroid, label, intertia = k_means(X = self.scaled_features,
                                            n_clusters = num_clust,
                                            random_state = random_state)

        labels = [str(i) for i in label]  # Convert labels to strings
        if not pre_split:
            self.plot_df.loc[:, [f'kmeans_{num_clust}']] = labels
        else:
            pass

    def run_gmm(self, n_components, random_state):
        gmm = GaussianMixture(n_components = n_components,
                              random_state = random_state)

        gmm.fit(self.scaled_features)
        gmm_labels = gmm.predict(self.scaled_features)
        gmm_labels = [str(i) for i in gmm_labels]  # Convert labels to strings

        probabilities = gmm.predict_proba(self.scaled_features)

        self.plot_df.loc[:, ['gmm']] = gmm_labels

    def run_leiden(self, resolution, random_state, knn = 15):
        k = knn
        adjacency_matrix = kneighbors_graph(self.scaled_features, k,
                                            mode = 'connectivity', include_self = False)
        leiden = Leiden(resolution = resolution, random_state = random_state)
        labels = leiden.fit_predict(adjacency_matrix)
        labels = [str(i) for i in labels]  # Convert labels to strings

        self.plot_df.loc[:, [f'leiden_{resolution}']] = labels

    def filter_border_polygons(self, tolerance = 1):
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
                index_list = this_sset.index.tolist()
                indices = [self.plot_df.index.get_loc(idx) for idx in index_list]
                fps_data = self.scaled_features[indices]
                sampled_indices = self.farthest_point_sampling(fps_data, sample_size)
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

            vmax = max(20, vmax)
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

        lower_bound = np.percentile(self.expression_data[marker], lower_percentile)
        upper_bound = np.percentile(self.expression_data[marker], upper_percentile)

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

        if len(filtered_df) > n:
            ids = filtered_df.sample(n, random_state = random_state).index.to_list()
        else:
            ids = filtered_df.index.to_list()

        self.manual_samples[marker] = ids

    def lineage_probs(self, marker_dictionary):
        exprs = self.expression_data.loc[:, list(marker_dictionary.values())]
        exprs_minmax = exprs.apply(minmax, axis=1)
        exprs_softmax = exprs_minmax.apply(softmax, axis=1)

        new_columns = [col.replace('_Cell_Mean', '_softmax') for col in exprs_softmax.columns]
        exprs_softmax.columns = new_columns

        return exprs_softmax

    def lineage_split(self, marker_dictionary, random_state = 5):
        self.lineage_softmax = self.lineage_probs(marker_dictionary)
        self.plot_df = pd.concat([self.plot_df, self.lineage_softmax], axis=1)

        num_clust = len(marker_dictionary.keys())
        kmeans_results = k_means(X=self.lineage_softmax, n_clusters=num_clust, random_state=random_state)
        kmeans_labels = [str(i) for i in kmeans_results[1]]  # Convert to strings

        self.lineage_assignments = pd.Series(kmeans_labels, index=self.lineage_softmax.index, dtype=str)  # Specify dtype=str
        self.plot_df.loc[self.plot_df.index, 'kmeans_lineage'] = self.lineage_assignments

        self.lineage_subsets = {}
        for i in range(num_clust):
            mask = self.lineage_assignments == str(i)
            subset_data = self.scaled_features[mask]

            self.lineage_subsets[str(i)] = { #String for everything
                'data': subset_data,
                'indices': self.plot_df[mask].index.tolist()
            }

    def run_stratified_clustering(self, resolution_dict = None, cluster_method = 'leiden', **kwargs):
        self.subclusters = {}

        if resolution_dict is None:
            resolution_dict = {}

        for lineage, subset in self.lineage_subsets.items():
            subset_indices = self.plot_df[self.plot_df['kmeans_lineage'] == lineage].index.tolist()
            subset_data = subset['data']

            if not subset_indices:
                print(f"Warning: No data for lineage {lineage}. Skipping clustering.")
                self.subclusters[lineage] = {}
                continue

            resolution = resolution_dict.get(lineage, kwargs.get('resolution', 1.0))
            labels = None

            if cluster_method == 'leiden':
                adjacency = kneighbors_graph(subset_data,
                                           kwargs.get('knn', 15),
                                           mode='connectivity',
                                           include_self = False)
                leiden = Leiden(resolution = resolution)
                labels = leiden.fit_predict(adjacency)
                labels = [str(i) for i in labels]
            elif cluster_method == 'kmeans':
                _, labels, _ = k_means(subset_data,
                                     n_clusters=kwargs.get('n_clusters', 5),
                                     random_state=kwargs.get('random_state', 5))
                labels = [str(i) for i in labels]

            # This is where the labels should be converted to strings, must pass these to the dataframe
            self.subclusters[lineage] = {idx: str(label) for idx, label in zip(subset_indices, labels)}

        # But, we must save this to the plot DF as well!
        for lineage, cluster_map in self.subclusters.items():
            for idx, label in cluster_map.items():
                self.plot_df.loc[idx, 'stratified_cluster'] = f"{lineage}_{label}" #Must pass the labels as strings as well!

