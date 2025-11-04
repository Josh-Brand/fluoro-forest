import sys
import os
import json
import tifffile
import importlib
import numpy as np
import math

import seaborn as sns
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from sklearn.cluster import k_means
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Polygon

# widget libraries
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QEventLoop

matplotlib.use('QtAgg')


#################################

def read_ome_tiff(file_path):
    with tifffile.TiffFile(file_path) as tif:
        return tif.asarray()

#################################

def read_geom_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

##################################

def process_cell_polygon(segment_data):
    # Extract x and y coordinates
    x = segment_data[0]
    y = segment_data[1]
    
    # Convert y-coordinates to absolute values
    y_inv = [float(np.abs(yi)) for yi in y]
    
    # Combine x and fixed y into coordinates
    coords = np.column_stack((x, y_inv))  # Create (x, y) pairs
    
    # Calculate centroid using fixed coordinates
    centroid = (np.mean(coords[:, 0]), np.mean(coords[:, 1]))
    
    return centroid, coords

#################################

def display_cell_multi_marker(image, marker_info, centroid, coords, show_markers, figure, padding=50):
    figure.clf()
    figure.clear()
    figure.patch.set_facecolor('black')
    
    centroid_x, centroid_y = centroid
    
    n_markers = len(show_markers[0]) if isinstance(show_markers[0], list) else len(show_markers)
    n_cols = min(4, n_markers)
    n_rows = math.ceil(n_markers / n_cols)
    markers_to_show = show_markers[0] if isinstance(show_markers[0], list) else show_markers

    gs = figure.add_gridspec(n_rows, n_cols)
    
    for i, marker in enumerate(markers_to_show):
        ax = figure.add_subplot(gs[i // n_cols, i % n_cols])
        
        ax.set_facecolor('black')
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)
            
        slice_index, vmin, vmax = marker_info[marker]

        # Calculate initial view limits
        x_start = int(centroid_x) - padding
        x_end = int(centroid_x) + padding
        y_start = int(centroid_y) - padding
        y_end = int(centroid_y) + padding

        # Ensure square shape and handle edge cases
        size = max(x_end - x_start, y_end - y_start)
        x_start = int(centroid_x) - size // 2
        x_end = x_start + size
        y_start = int(centroid_y) - size // 2
        y_end = y_start + size

        # Adjust for image boundaries
        if x_start < 0:
            x_end -= x_start
            x_start = 0
        elif x_end > image.shape[2]:
            x_start -= (x_end - image.shape[2])
            x_end = image.shape[2]
        
        if y_start < 0:
            y_end -= y_start
            y_start = 0
        elif y_end > image.shape[1]:
            y_start -= (y_end - image.shape[1])
            y_end = image.shape[1]

        # Extract region of interest
        marker_image = image[slice_index, y_start:y_end, x_start:x_end]
        
        im = ax.imshow(marker_image, cmap='magma', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
        
        # Adjust polygon coordinates
        adjusted_coords = [(x - x_start, y - y_start) for x, y in coords]
        poly = plt.Polygon(adjusted_coords, fill = False, edgecolor = 'cyan', linewidth = 1, alpha = 0.5)
        ax.add_patch(poly)

        ax.set_title(marker, fontsize=10, color='white')
        ax.set_xticks([])
        ax.set_yticks([])

    figure.tight_layout()
    figure.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.3, hspace=0.3)

#################################

def minmax(x):
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val == min_val:
        return np.zeros_like(x, dtype = float)
    return (x - min_val) / (max_val - min_val)

#################################

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 0)

#################################

def transform_geometry(feature):
    # Extract coordinates
    coordinates = feature['geometry']['coordinates'][0] # from geojson qupath export
    
    # Separate x and y coordinates
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]
    
    # Use the existing ID from the feature
    feature_id = feature['id']
    
    # Create the new structure
    return {feature_id: [x_coords, y_coords]}


########

def merge_annotations(dfs, id_col = 'Object.ID', 
    celltype_col = 'CellType', exclude_pattern = None):

    # Standardize columns
    for df in dfs:
        df.columns = [id_col, celltype_col]
    
    # Concatenate all, keeping only the latest annotation for each Object.ID
    merged = pd.concat(dfs, axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=[id_col], keep = 'last')
    
    # Optionally filter out unwanted cell types
    if exclude_pattern:
        merged = merged[~merged[celltype_col].str.contains(exclude_pattern, na=False)]
    
    # Create dictionary: {Object.ID: CellType}
    merged_dict = merged.set_index(id_col)[celltype_col].to_dict()
    
    # Count cell types
    print(merged[celltype_col].value_counts())
    
    return merged_dict
