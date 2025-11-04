# acceess in segmentations environment (cellpose)
import pandas as pd
from skimage import measure
import numpy as np
import string
import random
import matplotlib.pyplot as plt
import skimage
from skimage.segmentation import expand_labels
from scipy.ndimage import distance_transform_edt

from scipy.spatial import Voronoi
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Rectangle


def average_expression(image, masks, outlines, marker_info, log = True, clip_low = 0., clip_high = 1.):
    num_rows = len(outlines) # preparing expression matrix
    df = pd.DataFrame({f'{marker}_Cell_Mean': [0] * num_rows for marker in marker_info})

    for i in range(len(marker_info)):
        exprs = measure.regionprops_table(masks, 
                                          intensity_image = image[i,:,:], 
                                          properties = ['label', 'mean_intensity'])
        df.iloc[:,i] = exprs['mean_intensity']

    if log:
        df = pd.DataFrame(np.log(df + 1))

    if clip_low != 0. or clip_high != 1.:
        print(f'clipping values between: {clip_low} - {clip_high} %-iles')

        lower_thresholds = df.quantile(clip_low)
        upper_thresholds = df.quantile(clip_high)
    
        # Clip values in the DataFrame to these thresholds
        clipped_df = df.clip(lower = lower_thresholds, 
                             upper = upper_thresholds, axis = 1)

    return df

def expand_areas(nuclear_masks, expansion_distance = 5):

    distance = distance_transform_edt(nuclear_masks > 0)    
    expanded_masks = expand_labels(nuclear_masks, distance = expansion_distance)
    
    return expanded_masks

def calc_cell_area(polygon):
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def qc_plots(expression_data, plot_show = 1):
    qc_plots = {'1': {'x': 'DAPI_Cell_Mean'},
                '2': {'x': 'Unexpanded_cell_area'}}

    params = qc_plots[str(plot_show)]

    plt.figure(figsize = (3,2))
    plt.hist(expression_data[params['x']], bins = 50, edgecolor = 'black', alpha = 0.8)
    plt.title(params['x'])
    plt.show()


def calculate_centroid(polygon, invert_y = True):
    if invert_y:
        num = -1
    else:
        num = 1
        
    x = polygon[:, 0]
    y = polygon[:, 1]
    
    centroid_x, centroid_y = np.mean(x), num*np.mean(y)

    return (centroid_x, centroid_y)

# for barcoding of cells
def generate_key():
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(15))


def calculate_mean_ranks(df, suffix='_Cell_Mean'):
    """
    Calculates the mean rank for each cell across specified marker columns.

    This function first ranks cells within each marker column (features ending
    with the specified suffix) and then calculates the average rank for each
    cell across all those markers.

    Assumes preprocessing (e.g., log/arcsinh transform) has ideally been
    applied to the input DataFrame *before* calling this function.

    Args:
        df (pd.DataFrame): DataFrame where rows are cells (e.g., indexed by
                           'Object.ID') and columns include marker intensity data.
        suffix (str, optional): The suffix identifying the marker columns to use
                                for ranking. Defaults to '_Cell_Mean'.

    Returns:
        pd.Series: A Series containing the mean rank for each cell, indexed by
                   the cell identifiers from the input DataFrame's index.
                   Returns an empty Series with the original index if no columns
                   with the specified suffix are found.
    """

    # 1. Select columns based on the specified suffix
    marker_cols = df.filter(like=suffix)

    # Check if any marker columns were found
    
    # 2. Calculate ranks for each marker column (column-wise ranking)
    # axis=0 means rank *within* each column independently.
    # method='average' handles ties by assigning the average rank.
    ranks_df = marker_cols.rank(axis=0, method='average')

    # 3. Calculate the mean rank across markers for each cell (row-wise mean)
    # axis=1 means calculate the mean *across* the columns for each row.
    mean_ranks = ranks_df.mean(axis=1)

    # Name the resulting series for clarity
    mean_ranks.name = 'Mean_Rank'

    return mean_ranks


def normalize_image(image_stack):
    normalized_stack = np.zeros_like(image_stack, dtype = np.uint8)
    
    for i in range(image_stack.shape[0]):
        layer = image_stack[i]
        layer_min, layer_max = np.min(layer), np.max(layer)
        
        # Normalize to 0-255 range
        normalized_layer = ((layer - layer_min) / (layer_max - layer_min) * 255).astype(np.uint8)
        normalized_stack[i] = normalized_layer

    return normalized_stack



    
