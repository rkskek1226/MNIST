import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import OrderedDict


def softmax(x):
    # exp_a = np.exp(x)
    # sum_exp_a = np.sum(exp_a)
    # return exp_a / sum_exp_a

    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

    # c = np.max(x)
    # exp_a = np.exp(x - c)
    # sum_exp_a = np.sum(exp_a)
    # return exp_a / sum_exp_a


def mse(y, t):
    return 0.5 * np.sum((y - t)**2)


def cross_entropy(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def one_hot_encoding(x):
    arr = np.zeros((x.size, 10))
    for idx, row in enumerate(arr):
        row[x[idx]] = 1

    return arr


