# requires cell_annotation environment
import numpy as np
import collections

import seaborn as sns
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.preprocessing import StandardScaler # for heatmap expression data
from scipy.optimize import linear_sum_assignment # for diagnolization in contingency plot


#################################

def cell_plot(core, plot_type = 'cell', figsize = (8, 6), col = None, size = 8, 
              coloring_type = 'continuous', palette = 'viridis', alpha = 1, color_map = False):
    
    # generates scatter plot, can be colored continuously or categorically, careful not to overload

        # core: core_data object 
        # plot_types: 'cell' for X_coord vs Y_coord and 'PC' for PC1 vs PC2
        # figsize: where width, height = inches
        # color: column in dataframe
        # size: sactter point size
        # coloring_type: 'continuous' or 'categorical', continouous as deafult for protection
        # palette: used for consistent plotting
        # alpha: transparency of points
    
    fig, ax = plt.subplots(figsize = figsize)

    # pick coordinates
    if plot_type == 'cell':
        x_coord, y_coord = 'X_coord', 'Y_coord'
    elif plot_type == 'PC':
        x_coord, y_coord = 'PC1', 'PC2'
    else:
        raise ValueError("plot_type must be 'cell' or 'PC'")

    # copy data to not overwrite
    plot_df = core.plot_df.copy()

    # color handling by data types
    if col is not None:
        is_continuous = pd.api.types.is_numeric_dtype(plot_df[col]) and coloring_type == 'continuous'

        if is_continuous:
            scatter = ax.scatter(plot_df[x_coord], plot_df[y_coord], 
                                 c=plot_df[col], cmap=palette, s=size)
            plt.colorbar(scatter, ax = ax, label = col)
        else:
            # categorical data
            plot_df[col] = plot_df[col].fillna('not_annotated').astype(str) # defaults to gray in sampling plots

            # fixed ordering in stratitifed clustering (0-1, 0-2, 1-1, 1-2, 1-3, ...)
            if col == 'stratified_cluster':
                def stratified_sort_key(x):
                    try:
                        parts = x.split('_')
                        return [int(p) for p in parts]
                    except ValueError:
                        return [float('inf')]

                categories = sorted(plot_df[col].unique(), key = stratified_sort_key)
            else:
                # categorical can be converted? clustering interpreted to numbers to sort, then converted back
                def sort_key(x):
                    try:
                        return int(x)  # string to integer for sorting levels
                    except ValueError:
                        return float('inf') 

                categories = sorted(plot_df[col].unique(), key = sort_key)

            # color map selection
            if not color_map:
                color_palette = sns.color_palette(palette, n_colors = len(categories)) # number of clusters, etc
                color_map = dict(zip(categories, color_palette))
                color_map['not_annotated'] = 'lightgray'

            # Scatter plot for each category
            for category in categories:
                mask = plot_df[col] == category
                ax.scatter(plot_df.loc[mask, x_coord], plot_df.loc[mask, y_coord], 
                           color = color_map[category], label = category, s = size, alpha = alpha)

            ax.legend(title = col, bbox_to_anchor = (1.05, 1), loc = 'upper left', markerscale = 3) # legend properties
    else:
        ax.scatter(plot_df[x_coord], plot_df[y_coord], s = size)

    ax.set_xlabel(x_coord)
    ax.set_ylabel(y_coord)
    plt.tight_layout()
    plt.show()

#################################

# creates contingency heatmap for classificatoin errors, aligns by max value 

def contingency_plot(core, column1, column2, figsize=(6, 4), annot_fontsize = 6):

        # core: should contain plot_df for comparison of clusters / classificaiton of choosing
        # column1 / column2 classifications to compare 1 to 1
    
    # check if columns exist
    if core is not None:
        if column1 not in core.plot_df.columns or column2 not in core.plot_df.columns:
            raise ValueError(f"One or both columns missing: {column1}, {column2}")
        df_copy = core.plot_df[[column1, column2]].copy() # copy for plot, overwrite protection
    else: 
        df_copy = pd.concat([column1, column2], axis = 1)
        column1, column2 = df_copy.columns[0], df_copy.columns[1]
    
    # sorting of values (numerically)
    def custom_sort_key(x):
        try:
            # for subclustered examples
            parts = x.split('_')
            return [int(p) for p in parts]
        except ValueError:
            return [float('inf')]

    # convert data to string 
    df_copy[column1] = df_copy[column1].fillna('NaN').astype(str)
    df_copy[column2] = df_copy[column2].fillna('NaN').astype(str)

    # sort rows/columns (archive?, now uses dagonalization)
    all_rows_categories = sorted(df_copy[column1].unique(), key = custom_sort_key)
    all_cols_categories = sorted(df_copy[column2].unique(), key = custom_sort_key)

    # crosstab for contingency then osrt
    crosstab = pd.crosstab(df_copy[column1], df_copy[column2], dropna = False)
    crosstab_full = crosstab.reindex(index = all_rows_categories, 
                                     columns = all_cols_categories, 
                                     fill_value = 0)

    # hungarian for diagonalization) - linear sum
    M, N = crosstab_full.shape
    k = max(M, N)
    cost_matrix = np.zeros((k, k))
    cost_matrix[:M, :N] = -crosstab_full.values

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # reordering
    final_ordered_rows = [all_rows_categories[i] for i in row_ind if i < M]
    final_ordered_cols = [all_cols_categories[i] for i in col_ind if i < N]

    # reindex and plot
    aligned_crosstab = crosstab_full.reindex(index = final_ordered_rows, columns = final_ordered_cols)

    plt.figure(figsize = figsize)
    sns.heatmap(aligned_crosstab,
                annot = True,
                fmt = "d",
                cmap = "viridis",
                cbar = True,
                linewidths = .5,
                linecolor = 'lightgray',
                alpha = 0.7,
                annot_kws={'size': annot_fontsize}
               )

    plt.xlabel(column2)
    plt.ylabel(column1)
    plt.title(f"{column1} vs {column2}")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


#################################


# heatmap of expression values in a classification or cluster within plot_df (saved in core class)


def expression_heatmap(core, cluster_col = 'stratified_cluster', cell_mean_substring = 'Cell_Mean',
                            cmap='vlag', figsize=(5, 8)):

        # plot_df in core contains expression values
        # cluster_col, which to plot from (leiden, predicted, etc)
        # cell_mean_substring, identifies plotting columns by subseting - qupath specific, generalize ?
        # cmap = cell colors 
       
    clst_df = core.plot_df.copy()
    exprs_df = core.expression_data.copy() # raw expression, will z score for plotting

    plot_df = pd.concat([clst_df, exprs_df], axis = 1)

    # convert clusters to number to sort
    def is_numeric_convertible(series):
        try:
            pd.to_numeric(series, errors = 'raise')
            return True
        except ValueError:
            return False
    
    # order
    if is_numeric_convertible(plot_df[cluster_col]):
        plot_df['_numeric_sort'] = pd.to_numeric(plot_df[cluster_col])
        plot_df = plot_df.sort_values('_numeric_sort')
        numeric_labels = True
    else:
        plot_df = plot_df.sort_values(cluster_col)
        numeric_labels = False
    
    # expression columns (_Cell_Mean)
    cell_mean_cols = [col for col in plot_df.columns if cell_mean_substring in col]

    # dataframe containing only cluster and expressino values of interest
    df_subset = plot_df.loc[:, cell_mean_cols + [cluster_col]]
    df_subset.columns = df_subset.columns.str.replace('_Cell_Mean', '')

    # cluster sorting
    plot_df['sort_key'] = pd.to_numeric(plot_df[cluster_col], errors='coerce')
    plot_df = plot_df.sort_values('sort_key').drop('sort_key', axis=1)
    
    # calculates mean within each group
    grouped_df = df_subset.groupby(cluster_col, sort = False).mean()  # Prevent re-sorting

    # z scaled data across clusters for interpretation of markers
    scaler = StandardScaler()
    zscored_df = pd.DataFrame(
        scaler.fit_transform(grouped_df),
        index=grouped_df.index,
        columns=grouped_df.columns
    )
    
    # heatmap
    plt.figure(figsize = figsize)
    ax = sns.heatmap(
        zscored_df.T,
        cmap = cmap,
        center = 0,
        linewidths = .5,
        cbar_kws = {'label': 'Z-Score'},
        yticklabels = True
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation = 0,
        fontsize = 8
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation = 45,
        ha = 'right',
        fontsize = 8
    )

    # plot titles
    plt.title(f'Expression by {cluster_col}', pad  =10)
    plt.xlabel(cluster_col, labelpad = 10)
    plt.ylabel('Markers', labelpad = 10)
    plt.tight_layout()
    plt.show()


#################################

# visualization of segmentation data (geomjson from Qupath)
# if segmentations in annotation window seem off, could be need to flip segmentations

def plot_segmentation(core, figsize = (5, 5), invert = True):
    fig, ax = plt.subplots(figsize = figsize)
    ax.set_facecolor('white')
    
    for polygon_id, coords in core.segments.items():
        x_coords = coords[0]

        # somtimes necessary if image is flipped (how images are rendered in plt.imshhow
        if invert:
            y_coords = [-1*val for val in coords[1]] # inverted for image
        else: 
            y_coords = coords[1]
            
        vertices = list(zip(x_coords, y_coords))

        # add patches one at a time
        polygon_patch = matplotlib.patches.Polygon(vertices, 
                                                  closed = True, 
                                                  fill = False, 
                                                  edgecolor = 'red', 
                                                  linewidth = 0.5)
        ax.add_patch(polygon_patch)
    
    ax.autoscale_view()
    ax.tick_params(axis = 'both', which = 'both', length = 0)
    ax.tick_params(labelbottom = False, labelleft = False)
    ax.set_aspect('equal', adjustable = 'box')
    ax.grid(False)
    
    plt.show()

#################################

# visualization of cross validation after fitting models via precision, recall, and f1 scores 

def plot_metrics(data, metrics):
    # metrics = f1, precision, recall

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), sharey = True)

    # list if only one metric
    if n_metrics == 1:
        axes = [axes]

    # color palette for cell types
    unique_classes = data['class'].unique()
    palette = sns.color_palette(n_colors=len(unique_classes))
    
    for i, metric in enumerate(metrics):
        sns.barplot(
            data = data,
            x = 'class',
            y = metric,
            alpha = 0.4,
            hue = 'class',
            palette = palette,
            edgecolor = 'black',
            dodge = False,
            ax = axes[i],
            legend = False,
            width = 0.7,
            errorbar = "se" # standard error used on bar
        )
        
        # overlay individual points for transparency
        sns.stripplot(
            data = data,
            x = 'class',
            y = metric,
            hue = 'class',
            palette = palette,
            dodge = False,
            ax = axes[i],
            jitter = True,
            size = 4,  
            alpha = 0.8,
            legend = False,
            linewidth = 0.5,
            edgecolor = 'black'
        )
        

        axes[i].set_title(f'{metric.capitalize()}')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('')
        axes[i].tick_params(labelrotation = 45, axis = 'x')
    

    plt.subplots_adjust(wspace = 0.4, top = 0.85)
    plt.ylim(0, 1.2)
    plt.tight_layout(rect = [0, 0, 1, 0.95], pad = 1)
    plt.show()

#################################