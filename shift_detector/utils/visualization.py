import numpy as np
import matplotlib.pyplot as plt

COLOR_1 = 'cornflowerblue'
COLOR_2 = 'seagreen'


def plot_cumulative_step_ratio_histogram(column1, column2, bin_edges):
    bin_counts1, _ = np.histogram(column1, bins=bin_edges)
    bin_counts2, _ = np.histogram(column2, bins=bin_edges)
    cumsum_ratios1 = np.cumsum(bin_counts1 / float(len(column1)))
    cumsum_ratios2 = np.cumsum(bin_counts2 / float(len(column2)))
    bin_width = bin_edges[1] - bin_edges[0]
    x = np.ravel(list(zip(bin_edges[:-1], bin_edges[:-1] + bin_width)))
    y1 = np.ravel(list(zip(cumsum_ratios1, cumsum_ratios1)))
    y2 = np.ravel(list(zip(cumsum_ratios2, cumsum_ratios2)))
    plt.plot(x, y1, color=COLOR_1)
    plt.plot(x, y2, alpha=0.5, color=COLOR_2)
    return cumsum_ratios1, cumsum_ratios2


def plot_ratio_histogram(column1, column2, bin_edges):
    bin_counts1, _ = np.histogram(column1, bins=bin_edges)
    bin_counts2, _ = np.histogram(column2, bins=bin_edges)
    bin_ratios1 = bin_counts1 / float(len(column1))
    bin_ratios2 = bin_counts2 / float(len(column2))
    bin_width = bin_edges[1] - bin_edges[0]
    plt.bar(bin_edges[:-1], bin_ratios1, width=bin_width, facecolor=COLOR_1)
    plt.bar(bin_edges[:-1], bin_ratios2, width=bin_width, alpha=0.5, facecolor=COLOR_2)