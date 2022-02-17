import matplotlib.pyplot as plt
import seaborn
import scipy.stats as sp
import numpy as np

from sel_characteristics import sample_mean


def plot_empirical_cdf(selection):
    seaborn.kdeplot(selection, cumulative=True)
    plt.title('Эмпирическая функция распределения')
    plt.show()


def norm_histogram(selection):
    seaborn.distplot(selection, kde=False, norm_hist=True)
    plt.title('Нормированная гистограмма')
    plt.show()


def confidence_interval(selection, a):
    mean = sample_mean(selection)
    n = len(selection)

    d = np.var(selection, ddof=1) * n / (n - 1)
    step = sp.t.ppf((a + 1) / 2, n - 1) * np.sqrt(d / n)
    interval = [mean - step, mean + step]

    return interval


def confidence_interval_plot(selection, interval, a):
    seaborn.kdeplot(selection, cumulative=True)
    plt.axvline(interval[0], color='r')
    plt.axvline(interval[1], color='r')
    plt.title('Доверительный интервал для теоретической функции распределения (доверительная веротяность '
              + str(a) + ')')
    plt.show()


def plot_teor_empirical(selection, distribution):
    seaborn.distplot(selection, hist=False, label='Исходная выборка')
    seaborn.distplot(distribution, hist=False, label='Гамма-распределение')
    plt.legend()
    plt.title('Эмпирическая и теоретическая оценка плотности')
    plt.show()


def plot_teor_empirical_cdf(selection, distribution):
    seaborn.kdeplot(selection, cumulative=True, label='Исходная выборка')
    seaborn.kdeplot(distribution, cumulative=True, label='Гамма-распределение')
    plt.legend()
    plt.title('Эмпирическая и теоретическая функция распределения')
    plt.show()