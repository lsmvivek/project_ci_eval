import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 2025-06-22, modified to allow for causes in columns or rows
def create_adj_mat_figure(adj_matrix: np.array, labels: list, make_lag1=True, causes_in_columns=True):
    
    # Prepare the adjacency matrix
    adj_matrix = adj_matrix.copy().astype(bool)
    if not make_lag1:
        adj_matrix = adj_matrix[:, :adj_matrix.shape[1] // 2]

    # Set font and color map
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    color_code = np.array([(230, 230, 230), (116, 169, 207)]) / 255.0
    cmap = matplotlib.colors.ListedColormap(color_code)

    if causes_in_columns:
        # Create the figure and heatmap
        fig, ax = plt.subplots(figsize=(10, 10), )
        sns.heatmap(adj_matrix, cmap=cmap, cbar=False, linewidths=0.7, 
                    linecolor='grey', zorder=10, ax=ax)

        ax.xaxis.tick_top()
        # Configure x-axis labels
        if make_lag1:
            extended_labels = labels + [f'{label}_lag1' for label in labels]
            ax.set_xticks(np.arange(0.5, len(extended_labels) + 0.5))
            ax.set_xticklabels(extended_labels, fontsize=10, rotation=90)
            for i, label in enumerate(extended_labels):
                if '_lag1' in label:
                    ax.get_xticklabels()[i].set_fontstyle('italic')
        else:
            ax.set_xticks(np.arange(0.5, len(labels) + 0.5))
            ax.set_xticklabels(labels, fontsize=10, rotation=90)

        # Configure y-axis labels
        ax.set_yticks(np.arange(0.5, len(labels) + 0.5))
        ax.set_yticklabels(labels, fontsize=10, rotation=0)
    
    else:
        # Create the figure and heatmap
        fig, ax = plt.subplots(figsize=(10, 10), )
        sns.heatmap(adj_matrix.T, cmap=cmap, cbar=False, linewidths=0.7, 
                    linecolor='grey', zorder=10, ax=ax)

        ax.xaxis.tick_top()
        # Configure x-axis labels
        if make_lag1:
            extended_labels = labels + [f'{label}_lag1' for label in labels]
            ax.set_yticks(np.arange(0.5, len(extended_labels) + 0.5))
            ax.set_yticklabels(extended_labels, fontsize=10, rotation=0)
            for i, label in enumerate(extended_labels):
                if '_lag1' in label:
                    ax.get_yticklabels()[i].set_fontstyle('italic')
        else:
            ax.set_yticks(np.arange(0.5, len(labels) + 0.5))
            ax.set_yticklabels(labels, fontsize=10, rotation=0)
            
        ax.set_xticks(np.arange(0.5, len(labels) + 0.5))
        ax.set_xticklabels(labels, fontsize=10, rotation=90)

    # Set aspect ratio and spines
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)

    # Adjust layout to ensure y-axis labels are visible
    plt.subplots_adjust(left=0.2)
    return fig