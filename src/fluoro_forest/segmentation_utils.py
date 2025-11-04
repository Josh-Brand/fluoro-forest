# acceess in cellpose environment
import pandas as pd
from skimage import measure
import numpy as np
import string
import random
import matplotlib.pyplot as plt
import skimage
from skimage.segmentation import expand_labels
from scipy.ndimage import distance_transform_edt

from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Rectangle

#################################

# calculates cell averages to be used for model training, prediction, and visualizations based on mask
def average_expression(image, masks, outlines, marker_info, log = True, clip_low = 0., clip_high = 1.):
    num_rows = len(outlines) # preparing expression matrix
    df = pd.DataFrame({f'{marker}_Cell_Mean': [0] * num_rows for marker in marker_info})

    for i in range(len(marker_info)):
        exprs = measure.regionprops_table(masks, 
                                          intensity_image = image[i,:,:], 
                                          properties = ['label', 'mean_intensity'])
        df.iloc[:,i] = exprs['mean_intensity']

    if log:
        df = pd.DataFrame(np.log(df + 0.1))

    if clip_low != 0. or clip_high != 1.:
        print(f'clipping values between: {100*clip_low} - {100*clip_high} %-iles')

        lower_thresholds = df.quantile(clip_low)
        upper_thresholds = df.quantile(clip_high)
    
        # Clip values in the DataFrame to these thresholds
        clipped_df = df.clip(lower = lower_thresholds, 
                             upper = upper_thresholds, axis = 1)

    return df

#################################

# nuclear expansion - expand segmentation to fit cell body around nucleus
def expand_areas(nuclear_masks, expansion_distance = 5):

    distance = distance_transform_edt(nuclear_masks > 0)    
    expanded_masks = expand_labels(nuclear_masks, distance = expansion_distance)
    
    return expanded_masks

#################################


def calc_cell_area(polygon): # can be used as QC metrics
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

#################################


def qc_plots(expression_data, plot_show = 1): # simple QC plots that can guide segmentation pruning
    qc_plots = {'1': {'x': 'DAPI_Cell_Mean'},
                '2': {'x': 'Cell_area'}}

    params = qc_plots[str(plot_show)]

    plt.figure(figsize = (3,2))
    plt.hist(expression_data[params['x']], bins = 50, edgecolor = 'black', alpha = 0.8)
    plt.title(params['x'])
    plt.show()

#################################

def calculate_centroid(polygon, invert_y = True):
    if invert_y:
        num = -1
    else:
        num = 1
        
    x = polygon[:, 0]
    y = polygon[:, 1]
    
    centroid_x, centroid_y = np.mean(x), num*np.mean(y)

    return (centroid_x, centroid_y)

#################################

# for barcoding of cells, random numbers work fine too
def generate_key():
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(15))

#################################

# possible QC tool - highlights cell types with low expression across all markers
def calculate_mean_ranks(df, suffix = '_Cell_Mean'):

    marker_cols = df.filter(like = suffix)

    # ranky by average markers expression in each
    ranks_df = marker_cols.rank(axis = 0, method = 'average')
    mean_ranks = ranks_df.mean(axis = 1)

    # save results
    mean_ranks.name = 'Mean_Rank'

    return mean_ranks

#################################

# if image is outside of range fit to standardized 0-255 (then later fixed)
def normalize_image(image_stack):
    normalized_stack = np.zeros_like(image_stack, dtype = np.uint8)
    
    for i in range(image_stack.shape[0]):
        layer = image_stack[i]
        layer_min, layer_max = np.min(layer), np.max(layer)
        
        # Normalize to 0-255 range
        normalized_layer = ((layer - layer_min) / (layer_max - layer_min) * 255).astype(np.uint8)
        normalized_stack[i] = normalized_layer

    return normalized_stack

#################################


    
