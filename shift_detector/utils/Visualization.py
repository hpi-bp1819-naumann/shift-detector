import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_cumulative_comparison(data1, data2):
    cumsums = []
    for data, color in [(data1, 'blue'), (data2, 'green')]:
        values, base = np.histogram(data, bins=range(min(np.concatenate([data1, data2])),
                                                     max(np.concatenate([data1, data2])) + 1))
        cumulative = np.cumsum(values)
        plt.plot(base[:-1], cumulative, c=color)
        cumsums.append(cumulative)
    max_idx, max_d = max_distance(cumsums[0], cumsums[1])
    plt.axvline(x=base[max_idx])
    print(max_d)
    plt.show()


def max_distance(cumsum1, cumsum2):
    distances = abs(cumsum1 - cumsum2)
    max_idx = [index for index in range(len(distances)) if distances[index] == max(distances)][0]
    return max_idx, max(distances)


def plot_histogram(column, data1, data2, type='numerical'):
    col1 = data1[column]
    col2 = data2[column]
    if type == 'categorical':
        data = pd.concat([col1.value_counts().head(50), col2.value_counts().head(50)], axis=1, sort=False)
        data = data.fillna(0).apply(axis='columns',
                                    func=lambda row: pd.Series([row.iloc[0] / sum(row), row.iloc[1] / sum(row)]))
        data.plot(kind='bar', figsize=(50, 10), fontsize=32, stacked=True)
    elif type == 'numerical':
        for data, color in [(col1, 'blue'), (col2, 'green')]:
            mn, mx = min(np.concatenate([col1, col2])), max(np.concatenate([col1, col2])) + 1
            bins = range(mn, mx, np.int64((mx - mn) / 40))  # TODO: Fix this
            plt.hist(data, bins=bins, color=color)
            plt.show()
    else:
        raise ValueError('Type may only be numerical or categorical')
    plt.show()