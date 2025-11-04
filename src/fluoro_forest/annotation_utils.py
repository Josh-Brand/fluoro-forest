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

# read in image file as tiff, each slice is one marker
def read_ome_tiff(file_path):
    with tifffile.TiffFile(file_path) as tif:
        return tif.asarray()

#################################

# used to read in cell segmentations from QuPath workflow
def read_geom_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

##################################

# converts .geomjson / segmentations to polygons for visualization
def process_cell_polygon(segment_data):
    # extract x and y coordinates
    x = segment_data[0]
    y = segment_data[1]
    
    # set to absolute values, segmentations inverted (as with image rendering)
    y_inv = [float(np.abs(yi)) for yi in y]
    
    # combine x and fixed y 
    coords = np.column_stack((x, y_inv))  # Create (x, y) pairs
    
    # Calculate centroid using fixed coordinates
    centroid = (np.mean(coords[:, 0]), np.mean(coords[:, 1]))
    
    return centroid, coords

#################################

# main visualization during the annotation loop
def display_cell_multi_marker(image, marker_info, centroid, coords, show_markers, figure, padding = 50): # adjust padding to see more around the cell
    figure.clf()
    figure.clear()
    figure.patch.set_facecolor('black') # background color of plots
    
    centroid_x, centroid_y = centroid
    
    n_markers = len(show_markers[0]) if isinstance(show_markers[0], list) else len(show_markers)
    n_cols = min(4, n_markers) # number of columns set to 4 as maximum
    n_rows = math.ceil(n_markers / n_cols)
    
    # defined by user
    markers_to_show = show_markers[0] if isinstance(show_markers[0], list) else show_markers

    gs = figure.add_gridspec(n_rows, n_cols)

    # arrange plots
    for i, marker in enumerate(markers_to_show):
        ax = figure.add_subplot(gs[i // n_cols, i % n_cols])
        
        ax.set_facecolor('black')
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)
            
        slice_index, vmin, vmax = marker_info[marker]

        # include padding for window around cell centroid
        x_start = int(centroid_x) - padding
        x_end = int(centroid_x) + padding
        y_start = int(centroid_y) - padding
        y_end = int(centroid_y) + padding

        # sets window size prevents issue if cell goes beyond image dimensions
        size = max(x_end - x_start, y_end - y_start)
        x_start = int(centroid_x) - size // 2
        x_end = x_start + size
        y_start = int(centroid_y) - size // 2
        y_end = y_start + size

        # image boundary handling
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

        # extract region of interest in the image slice
        marker_image = image[slice_index, y_start:y_end, x_start:x_end]

        # image for plotting
        im = ax.imshow(marker_image, 
                       cmap = 'magma', 
                       vmin = vmin, 
                       vmax = vmax, 
                       aspect = 'equal',
                       origin = 'lower')
        
        # include polygon coordinates to show user what cell is being annotated
        adjusted_coords = [(x - x_start, y - y_start) for x, y in coords]
        poly = plt.Polygon(adjusted_coords, fill = False, edgecolor = 'cyan', linewidth = 1, alpha = 0.5)
        ax.add_patch(poly)

        ax.set_title(marker, fontsize = 10, color = 'white')
        ax.set_xticks([])
        ax.set_yticks([])

    figure.tight_layout()
    figure.subplots_adjust(left = 0.05, right = 0.95, top = 0.9, bottom = 0.05, wspace = 0.3, hspace = 0.3)

#################################

def minmax(x):
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val == min_val:
        return np.zeros_like(x, dtype = float)
    return (x - min_val) / (max_val - min_val)

#################################

# used for lineage level clustering with kmeans
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 0)

#################################

# for each cell id, contains coordinates (geometry) that is extracted from geojson
# qupath specific workflow
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


#################################

# used in case annotations from multiple sessions are mereged, prioritizes latest if duplicated
# cells were annotated

def merge_annotations(dfs, id_col = 'Object.ID', 
    celltype_col = 'CellType', exclude_pattern = None):
    
    for df in dfs:
        df.columns = [id_col, celltype_col]
    
    # concatenate all
    merged = pd.concat(dfs, axis = 0, ignore_index=True)
    merged = merged.drop_duplicates(subset=[id_col], keep = 'last')
    
    # filter out cell type patterns that may be ambiguous (goal-specific)
    if exclude_pattern:
        merged = merged[~merged[celltype_col].str.contains(exclude_pattern, na = False)]
    
    # dictionary that goes into core.annotations slot
    merged_dict = merged.set_index(id_col)[celltype_col].to_dict()
    
    print(merged[celltype_col].value_counts()) # cell type counts
    
    return merged_dict
