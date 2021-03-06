from collections import namedtuple

import numpy as np
import pandas as pd

COLOR_1 = 'cornflowerblue'
COLOR_2 = 'seagreen'

LEGEND_1 = 'DS 1'
LEGEND_2 = 'DS 2'

PLOT_ROW_HEIGHT = 5.0
PLOT_GRID_WIDTH = 12.0

PlotData = namedtuple('PlotData', ['plot_function', 'required_rows'])


def calculate_bin_counts(bin_edges, columns):
    bin_counts1, _ = np.histogram(columns[0], bins=bin_edges)
    bin_counts2, _ = np.histogram(columns[1], bins=bin_edges)
    return bin_counts1, bin_counts2


def calculate_bin_ratios(bin_counts, columns):
    bin_ratios1 = bin_counts[0] / float(len(columns[0]))
    bin_ratios2 = bin_counts[1] / float(len(columns[1]))
    return bin_ratios1, bin_ratios2


def cumsum_of_ratios(bin_ratios):
    return np.cumsum(bin_ratios[0]), np.cumsum(bin_ratios[1])


def plot_cumulative_step_ratio_histogram(axes, columns, bin_edges):
    bin_counts = calculate_bin_counts(bin_edges, columns)
    bin_ratios = calculate_bin_ratios(bin_counts, columns)
    bin_width = bin_edges[1] - bin_edges[0]
    cumsum_ratios = cumsum_of_ratios(bin_ratios)
    x = np.ravel(list(zip(bin_edges[:-1], bin_edges[:-1] + bin_width)))
    y1 = np.ravel(list(zip(cumsum_ratios[0], cumsum_ratios[0])))
    y2 = np.ravel(list(zip(cumsum_ratios[1], cumsum_ratios[1])))
    axes.plot(x, y1, color=COLOR_1)
    axes.plot(x, y2, alpha=0.5, color=COLOR_2)
    return cumsum_ratios


def plot_binned_ratio_histogram(axes, columns, bin_edges):
    bin_counts = calculate_bin_counts(bin_edges, columns)
    bin_ratios = calculate_bin_ratios(bin_counts, columns)
    bin_width = bin_edges[1] - bin_edges[0]
    axes.bar(bin_edges[:-1], bin_ratios[0], width=bin_width, facecolor=COLOR_1)
    axes.bar(bin_edges[:-1], bin_ratios[1], width=bin_width, alpha=0.5, facecolor=COLOR_2)


def calculate_value_ratios(columns, top_k):
    value_counts1 = columns[0].value_counts().sort_values(ascending=False)
    value_counts1.index = [str(ix) for ix in value_counts1.index]
    value_counts2 = columns[1].value_counts().sort_values(ascending=False)
    value_counts2.index = [str(ix) for ix in value_counts2.index]
    indices = set(value_counts1.head(top_k).index).union(set(value_counts2.head(top_k).index))
    value_counts = pd.concat([value_counts1[indices], value_counts2[indices]], axis=1).sort_index().fillna(0.0)
    return value_counts.fillna(0).apply(axis='columns',
                                        func=lambda row: pd.Series([row.iloc[0] / len(columns[0]),
                                                                    row.iloc[1] / len(columns[1])],
                                                                   index=[LEGEND_1, LEGEND_2]))


def plot_categorical_horizontal_ratio_histogram(axes, columns, top_k):
    value_ratios = calculate_value_ratios(columns, top_k)
    return value_ratios.plot(kind='barh', fontsize='medium', ax=axes)


def plot_title(column):
    return "Column: '{}'".format(column)
