import numpy as np
import matplotlib.pyplot as plt


def cumulative_step_ratio_histogram(column, df1, df2, bin_edges):
    bin_counts1, _ = np.histogram(df1[column], bins=bin_edges)
    bin_counts2, _ = np.histogram(df2[column], bins=bin_edges)
    cumsum_ratios1 = np.cumsum(bin_counts1 / float(len(df1[column])))
    cumsum_ratios2 = np.cumsum(bin_counts2 / float(len(df2[column])))
    bin_width = bin_edges[1] - bin_edges[0]
    x = np.ravel(list(zip(bin_edges[:-1], bin_edges[:-1] + bin_width)))
    y1 = np.ravel(list(zip(cumsum_ratios1, cumsum_ratios1)))
    y2 = np.ravel(list(zip(cumsum_ratios2, cumsum_ratios2)))
    plt.plot(x, y1, color='cornflowerblue')
    plt.plot(x, y2, alpha=0.5, color='seagreen')
    return cumsum_ratios1, cumsum_ratios2


def ratio_histogram(column, df1, df2, bin_edges):
    bin_counts1, _ = np.histogram(df1[column], bins=bin_edges)
    bin_counts2, _ = np.histogram(df2[column], bins=bin_edges)
    bin_ratios1 = bin_counts1 / float(len(df1[column]))
    bin_ratios2 = bin_counts2 / float(len(df2[column]))
    bin_width = bin_edges[1] - bin_edges[0]
    plt.bar(bin_edges[:-1], bin_ratios1, width=bin_width, facecolor='cornflowerblue')
    plt.bar(bin_edges[:-1], bin_ratios2, width=bin_width, facecolor='seagreen')