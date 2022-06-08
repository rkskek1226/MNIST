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


class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, self.t)
        # self.loss = mse(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
