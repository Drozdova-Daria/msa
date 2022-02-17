import math
import numpy as np

def sample_mean(selection):
    return sum(selection) / len(selection)


def variance(selection):
    v = 0
    for s in selection:
        v += math.pow(s, 2)

    return v / len(selection) - sample_mean(selection)


def central_empirical_point(selection, order):
    m = 0
    s_mean = sample_mean(selection)

    for s in selection:
        m += math.pow(s - s_mean, order)

    return m / len(selection)


def asymmetry_coefficient(selection):
    return central_empirical_point(selection, 3) / math.pow(math.sqrt(variance(selection)), 3)


def excess(selection):
    return central_empirical_point(selection, 4) / math.pow(math.sqrt(variance(selection)), 4)


def standard_deviation(selection):
    x_ = sample_mean(selection)
    d = 0

    for s in selection:
        d += math.pow(s - x_, 2)

    return d / (len(selection) - 1)


def method_maximum_likelihood(sel):
    return np.mean(sel), np.var(sel)