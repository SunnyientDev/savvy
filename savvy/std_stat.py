import plotly.express as px
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def simple_mean(values: list) -> float:
    return sum(values) / len(values)


def weighted_mean(values: list, counts: list) -> float:
    """
    Weighted average.
    """
    return sum([value * count for value, count in zip(values, counts)]) / sum(counts)


def get_frequencies(values: list) -> dict:
    """
    Get frequency.
    """
    unique_values = sorted(list(set(values)))
    return {key: values.count(key) for key in unique_values}


def frequency_polygon(values: list, counts: list):
    """
    Polygon for values and their frequency.
    """
    data = {"variant": values,
            "frequency": [count / sum(counts) for count in counts]}
    fig = px.line(data, x="variant", y="frequency", title='Frequency polygon')
    return fig


def variation_range(values: list) -> float | int:
    """
    R = x_max - x_min.
    """
    return max(values) - min(values)


def get_intervals_num(values: list):
    """
    Sturges formula: partial intervals.
    """
    return 1 + 3.222 * math.log(len(values), 10)


def equal_spaced_grouping(values: list) -> dict:
    intervals_num = round(get_intervals_num(values), 0)
    h = (max(values) - min(values)) / intervals_num
    groups = {"x_min": [i for i in np.arange(min(values), max(values), h)],
              "x_max": [i for i in np.arange(min(values) + h, max(values) + h, h)]}
    groups['x_mean'] = [value / 2 for value in list(map(lambda x, y: x + y, groups['x_min'], groups['x_max']))]
    groups['counts'] = [0] * len(groups['x_mean'])
    for i in range(len(groups['x_mean'])):
        groups['counts'][i] = len(list(filter(lambda score: groups['x_min'][i] <= score < groups['x_max'][i], values)))
    return {'groups': groups, 'parameters': {'h': h, 'intervals_count': intervals_num}}


def plot_frequency_histogram(values: list) -> None:
    groups = equal_spaced_grouping(values)
    h = groups['parameters']['h']
    groups = groups['groups']
    groups['n_i/h'] = [i/h for i in groups['counts']]
    plt.bar(groups['x_min'], groups['n_i/h'], width=h)
    xticks = list(set(groups['x_min'] + groups['x_max']))
    plt.xticks(len(xticks), xticks)
    plt.show()


