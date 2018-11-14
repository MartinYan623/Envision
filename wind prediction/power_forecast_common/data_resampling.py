# coding=utf-8
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


def resample_data_ratio(x_df, y_df, min_prob=0.05, seed_num=100):
    np.random.seed(seed_num)
    y_prob = y_df / 12
    y_prob[y_prob < min_prob] = min_prob
    y_prob[y_prob >= (1 - min_prob)] = 1 - min_prob

    choose = np.random.binomial(1, p=y_prob).astype(bool)
    return x_df[choose], y_df[choose]


def resample_data_gaussian(x_df, y_df, mean=9, std=2, max_prob=0.99, seed_num=100):
    np.random.seed(seed_num)
    y_prob = norm.pdf(y_df, mean, std)
    alpha = max_prob / norm.pdf(mean-std, mean, std)
    y_prob = y_prob * alpha
    y_prob[y_prob >= max_prob] = max_prob
    choose = np.random.binomial(1, p=y_prob).astype(bool)
    return x_df[choose], y_df[choose]

# original min_ min_value=4, max_value=12
def resample_data_duplicate(x_df, y_df, min_value=3, max_value=15, seed_num=1914):
    # double the points within min_value and max_value
    # drop half of the points below min_value
    np.random.seed(seed_num)
    sample_prob = 1.0 * np.zeros(y_df.shape)
    keep_flag = (y_df >= min_value) & (y_df <= max_value)
    sample_prob[keep_flag] = 1.0

    small_flag = y_df < min_value
    small_prob = y_df[y_df < min_value] / min_value
    sample_prob[small_flag] = small_prob

    large_flag = y_df > max_value
    large_prob = max_value / y_df[y_df > max_value]
    sample_prob[large_flag] = large_prob

    choose = np.random.binomial(1, p=sample_prob).astype(bool)
    return pd.concat([x_df[choose], x_df[keep_flag]]), pd.concat([y_df[choose], y_df[keep_flag]])


def check_ws_dist(ws_array):
    bins = [0, 4, 6, 8, 10, 12]
    counts = []
    for n in range(len(bins)):
        start = bins[n]
        if n == len(bins) - 1:
            end = 1000
        else:
            end = bins[n+1]
        index = np.where((ws_array >= start) & (ws_array < end))[0]
        counts.append(len(index))
    return dict(zip(bins, counts))


def plot_gaussian_dist(mean, std):
    x = np.linspace(mean - 2 * std, mean + 2 * std, 100)
    prob = norm.pdf(x, mean, std)
    plt.plot(x, prob)
    plt.show()