import numpy as np
import scipy.stats as sp

from sel_characteristics import *


def fisher(selection, distribution):
    s_1 = standard_deviation(selection)
    s_2 = standard_deviation(distribution)

    if s_1 < s_2:
        F = sample_mean(distribution) / sample_mean(selection)
        F_e = s_2 / s_1
    else:
        F = sample_mean(selection) / sample_mean(distribution)
        F_e = s_1 / s_2

    return F_e, F


def check_distribution(selection):
    s = np.random.normal(0.5, 0.1, 60)
    f_e, f = fisher(selection, s)
    if f_e < f:
        return 'normal'

    e = np.random.exponential(1.5, 60)
    f_e, f = fisher(selection, e)
    if f_e < f:
        return 'exponential'

    b = sp.beta.rvs(2.31, 0.627, size=60)
    f_e, f = fisher(selection, b)
    if f_e < f:
        return 'beta'

    g = sp.gamma.rvs(1.99, size=60)
    f_e, f = fisher(selection, g)
    if f_e < f:
        return 'gamma'

    return 'none'















