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


def cell_plot(core, plot_type = 'cell', figsize = (8, 6), col = None, size = 8, 
              coloring_type = 'continuous', palette = 'viridis', alpha = 1, color_map = False):
    """
    Generates a scatter plot, colored by a specified column.

    Parameters:
        core (core_data): The core_data object containing the data.
        plot_type (str, optional): Type of plot to generate. 'cell' for X_coord vs Y_coord, 'PC' for PC1 vs PC2. Defaults to 'cell'.
        figsize (tuple, optional): The figure size (width, height) in inches. Defaults to (8, 6).
        col (str, optional): The column to color the points by. Defaults to None.
        size (int, optional): The size of the scatter points. Defaults to 8.
        coloring_type (str, optional): 'continuous' or 'categorical'. Defaults to 'continuous'.
        palette (str, optional): The color palette to use. Defaults to 'viridis'.
        alpha (float, optional): Transparency level of scatter points. Defaults to 1.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 1. Define x and y coordinates
    if plot_type == 'cell':
        x_coord, y_coord = 'X_coord', 'Y_coord'
    elif plot_type == 'PC':
        x_coord, y_coord = 'PC1', 'PC2'
    else:
        raise ValueError("plot_type must be 'cell' or 'PC'")

    # 2. Copy the data
    plot_df = core.plot_df.copy()

    # 3. Handle coloring based on the data type
    if col is not None:
        is_continuous = pd.api.types.is_numeric_dtype(plot_df[col]) and coloring_type == 'continuous'

        if is_continuous:
            scatter = ax.scatter(plot_df[x_coord], plot_df[y_coord], 
                                 c=plot_df[col], cmap=palette, s=size)
            plt.colorbar(scatter, ax=ax, label=col)
        else:
            # Handle categorical data
            plot_df[col] = plot_df[col].fillna('not_annotated').astype(str)

            # Custom ordering for stratified_cluster
            if col == 'stratified_cluster':
                def stratified_sort_key(x):
                    try:
                        parts = x.split('_')
                        return [int(p) for p in parts]
                    except ValueError:
                        return [float('inf')]  # Push invalid values to the end

                categories = sorted(plot_df[col].unique(), key=stratified_sort_key)
            else:
                # Check if categorical values can be converted into numbers and sort accordingly
                def sort_key(x):
                    try:
                        return int(x)  # Convert to integer for sorting
                    except ValueError:
                        return float('inf')  # Push non-numeric values to the end

                categories = sorted(plot_df[col].unique(), key=sort_key)

            # Define color mapping
            if not color_map:
                color_palette = sns.color_palette(palette, n_colors=len(categories))
                color_map = dict(zip(categories, color_palette))
                color_map['not_annotated'] = 'lightgray'

            # Scatter plot for each category
            for category in categories:
                mask = plot_df[col] == category
                ax.scatter(plot_df.loc[mask, x_coord], plot_df.loc[mask, y_coord], 
                           color=color_map[category], label=category, s=size, alpha=alpha)

            ax.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)
    else:
        ax.scatter(plot_df[x_coord], plot_df[y_coord], s=size)

    # 4. Set labels and title
    ax.set_xlabel(x_coord)
    ax.set_ylabel(y_coord)

    # 5. Adjust layout
    plt.tight_layout()
    plt.show()

#####

def contingency_plot(core, column1, column2, figsize=(6, 4), annot_fontsize = 6):
    """
    Generates a simplified contingency heatmap plot, aligning rows and columns
    to maximize diagonal values using the Hungarian algorithm, while ensuring
    ALL unique values from both original columns are included. Uses a custom
    sort key for category labels containing numbers/underscores.

    Args:
        core: An object assumed to have a pandas DataFrame attribute named 'plot_df'.
        column1 (str): The name of the first column (rows of heatmap).
        column2 (str): The name of the second column (columns of heatmap).
        figsize (tuple): The size of the plot figure.
        annot_fontsize (int): Font size for the annotations inside the heatmap cells.
    """
    # 1. Ensure columns exist and create a working copy
    if core is not None:
        if column1 not in core.plot_df.columns or column2 not in core.plot_df.columns:
            raise ValueError(f"One or both columns missing: {column1}, {column2}")
        df_copy = core.plot_df[[column1, column2]].copy()
    else: 
        df_copy = pd.concat([column1, column2], axis = 1)
        column1, column2 = df_copy.columns[0], df_copy.columns[1]
    
    # --- Define the custom sorting key function ---
    def custom_sort_key(x):
        """
        Sorts strings numerically. Handles simple numbers ('10') and
        underscore-separated numbers ('1_10'). Non-numeric strings
        (like 'NaN' or others) are sorted last.
        """
        try:
            # Split by underscore and convert parts to integers
            parts = x.split('_')
            return [int(p) for p in parts]
        except ValueError:
            # If conversion fails (e.g., for 'NaN' or other non-numeric strings),
            # return a value that sorts these items last.
            # Using [float('inf')] groups all non-numeric items together at the end.
            return [float('inf')]
    # --------------------------------------------

    # 2. Convert data to string, handle NaNs, get *all* unique custom-sorted categories
    df_copy[column1] = df_copy[column1].fillna('NaN').astype(str)
    df_copy[column2] = df_copy[column2].fillna('NaN').astype(str)

    # --- Use custom sort_key with sorted() ---
    all_rows_categories = sorted(df_copy[column1].unique(), key=custom_sort_key)
    all_cols_categories = sorted(df_copy[column2].unique(), key=custom_sort_key)
    # -----------------------------------------

    # 3. Create the contingency table and reindex fully
    crosstab = pd.crosstab(df_copy[column1], df_copy[column2], dropna=False)
    # Use the custom-sorted categories for consistent indexing
    crosstab_full = crosstab.reindex(index=all_rows_categories, columns=all_cols_categories, fill_value=0)

    # 4. Prepare square cost matrix for Hungarian Algorithm
    M, N = crosstab_full.shape
    k = max(M, N)
    cost_matrix = np.zeros((k, k))
    cost_matrix[:M, :N] = -crosstab_full.values

    # 5. Apply Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 6. Get the optimally ordered category lists
    # Map indices back using the custom-sorted lists established earlier
    final_ordered_rows = [all_rows_categories[i] for i in row_ind if i < M]
    final_ordered_cols = [all_cols_categories[i] for i in col_ind if i < N]

    # 7. Reindex the *full* crosstab with the optimal order determined by alignment
    aligned_crosstab = crosstab_full.reindex(index=final_ordered_rows, columns=final_ordered_cols)

    # 8. Plot the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(aligned_crosstab,
                annot=True,
                fmt="d",
                cmap="viridis",
                cbar=True,
                linewidths=.5,
                linecolor='lightgray',
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


#####

def expression_heatmap(core, cluster_col='stratified_cluster', cell_mean_substring='Cell_Mean',
                            cmap='vlag', figsize=(5, 8)):
    """
    Generates a z-scored heatmap of gene expression by cluster.

    Parameters:
        plot_df (pd.DataFrame): DataFrame containing gene expression data and cluster assignments.
        cluster_col (str, optional): Name of the column containing cluster assignments. Defaults to 'stratified_cluster'.
        cell_mean_substring (str, optional): Substring to identify gene expression columns. Defaults to 'Cell_Mean'.
        cmap (str, optional): Color map for the heatmap. Defaults to 'vlag'.
        figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (5, 8).
    """

    clst_df = core.plot_df.copy()
    exprs_df = core.expression_data.copy()

    plot_df = pd.concat([clst_df, exprs_df], axis = 1)
    # 1. Select gene expression columns
    cell_mean_cols = [col for col in plot_df.columns if cell_mean_substring in col]

    # 2. Create subset DataFrame
    df_subset = plot_df.loc[:, cell_mean_cols + [cluster_col]]
    df_subset.columns = df_subset.columns.str.replace('_Cell_Mean', '')
    
    # 3. Group by cluster and calculate the mean
    grouped_df = df_subset.groupby(cluster_col).mean()

    # 4. Standardize the grouped data
    scaler = StandardScaler()
    zscored_df = pd.DataFrame(
        scaler.fit_transform(grouped_df),
        index=grouped_df.index,
        columns=grouped_df.columns
    )
    
    # 5. Create the heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        zscored_df.T,
        cmap=cmap,
        center=0,
        linewidths=.5,
        cbar_kws={'label': 'Z-Score'},
        yticklabels=True
    )

    # 6. Improve label visibility
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=8
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha='right',
        fontsize=8
    )

    # 7. Customize plot elements
    plt.title(f'Expression by {cluster_col}', pad=10)
    plt.xlabel(cluster_col, labelpad=10)
    plt.ylabel('Markers', labelpad=10)
    plt.tight_layout()
    plt.show()

def plot_segmentation(core, figsize = (5, 5), invert = True):
    fig, ax = plt.subplots(figsize = figsize)
    ax.set_facecolor('white')
    
    for polygon_id, coords in core.segments.items():
        x_coords = coords[0]
        
        if invert:
            y_coords = [-1*val for val in coords[1]] # inverted for image
        else: 
            y_coords = coords[1]
            
        vertices = list(zip(x_coords, y_coords))
    
        polygon_patch = matplotlib.patches.Polygon(vertices, 
                                                  closed = True, 
                                                  fill = False, 
                                                  edgecolor = 'red', 
                                                  linewidth = 0.5)
        ax.add_patch(polygon_patch)
    
    ax.autoscale_view()
    ax.tick_params(axis = 'both', which = 'both', length = 0)
    ax.tick_params(labelbottom = False, labelleft = False)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    
    plt.show()


def plot_metrics(data, metrics):
    """
    Creates side-by-side bar plots with overlaid scatter points for multiple metrics.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), sharey=True)

    # Ensure axes is a list even if only one metric
    if n_metrics == 1:
        axes = [axes]

    # Create a color palette for classes
    unique_classes = data['class'].unique()
    palette = sns.color_palette(n_colors=len(unique_classes))
    
    for i, metric in enumerate(metrics):
        # Plot barplot as background
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
            errorbar = "se"
        )
        
        # Overlay scatterplot
        sns.stripplot(
            data = data,
            x = 'class',
            y = metric,
            hue = 'class',
            palette = palette,
            dodge = False,
            ax = axes[i],
            jitter = True,
            size = 4,  # adjust size
            alpha = 0.8, # transparency
            legend = False,
            linewidth = 0.5,
            edgecolor = 'black'
        )
        

        axes[i].set_title(f'{metric.capitalize()}')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('')
        axes[i].tick_params(labelrotation=45, axis = 'x')
    

    plt.subplots_adjust(wspace = 0.4, top = 0.85)
    plt.ylim(0, 1.2)
    plt.tight_layout(rect = [0, 0, 1, 0.95], pad=1)
    plt.show()

