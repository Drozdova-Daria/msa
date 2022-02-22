import matplotlib.pyplot as plt
import seaborn
import scipy.stats as sp
import numpy as np
import statsmodels.api as sm

from sel_characteristics import sample_mean


def plot_empirical_cdf(selection):
    ecdf = sm.distributions.ECDF(selection)
    x = np.linspace(min(selection), max(selection))
    y = ecdf(x)
    plt.step(x, y)
    plt.title('Эмпирическая функция распределения')
    plt.show()


def norm_histogram(selection):
    plt.hist(selection, bins=10)
    plt.title('Нормированная гистограмма')
    plt.show()


def confidence_interval(selection, a):
    mean = sample_mean(selection)
    n = len(selection)

    d = np.var(selection, ddof=1) * n / (n - 1)
    print(sp.t.ppf((a + 1) / 2, n - 1))
    step = sp.t.ppf((a + 1) / 2, n - 1) * np.sqrt(d / n)
    interval = [mean - step, mean + step]

    return interval


def confidence_interval_plot(selection, interval, a):
    ecdf = sm.distributions.ECDF(selection)
    x = np.linspace(min(selection), max(selection))
    y = ecdf(x)
    plt.step(x, y, label='Эмпирическая функция распределения')

    interval = confidence_interval(selection, a)

    y1 = [val - interval[0] for val in y]
    y2 = [val + interval[1] for val in y]
    plt.step(x, y1, color='r', label='g='+str(a))
    plt.step(x, y2, color='r')
    plt.legend()
    plt.title('Доверительный интервал для теоретической функции распределения (доверительная веротяность '
              + str(a) + ')')
    plt.show()


def plot_teor_empirical(selection, distribution):
    seaborn.distplot(distribution, hist=False, label='Гамма-распределение')
    seaborn.distplot(selection, hist=True, label='Исходная выборка')
    plt.legend()
    plt.title('Эмпирическая и теоретическая оценка плотности')
    plt.show()


def plot_teor_empirical_cdf(selection, distribution):
    ecdf = sm.distributions.ECDF(selection)
    x = np.linspace(min(selection), max(selection))
    y = ecdf(x)
    plt.step(x, y, label='Исходная выборка')
    seaborn.kdeplot(distribution, cumulative=True, label='Гамма-распределение')
    plt.legend()
    plt.title('Эмпирическая и теоретическая функция распределения')
    plt.show()
