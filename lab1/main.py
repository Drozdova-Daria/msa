from plots import *
from chi_2 import *


def read_data(file_name):
    selection = []

    with open(file_name) as f:
        for line in f:
            selection += [float(x) for x in line.split()]

    return selection


if __name__ == '__main__':
    selection = read_data("Number_4.txt")

    print('Исходная выборка:')
    print(selection, end='\n\n')

    print('Выборочное среднее:')
    print(sample_mean(selection))

    print('Выборочная дисперсия:')
    print(variance(selection))

    print('Выборочный коэффициент ассиметрии:')
    print(asymmetry_coefficient(selection))

    print('Выборочный эксцесс:')
    print(excess(selection), end='\n\n')

    plot_empirical_cdf(selection)

    norm_histogram(selection)

    a = 0.95
    interval95 = confidence_interval(selection, a)
    print('Доверительный интервал (доверительная веротяность ' + str(a) + '):')
    print(interval95)
    confidence_interval_plot(selection, interval95, a)

    a = 0.90
    interval90 = confidence_interval(selection, a)
    print('Доверительный интервал (доверительная веротяность ' + str(a) + '):')
    print(interval90, end='\n\n')
    confidence_interval_plot(selection, interval90, a)

    print('Вид распределения')
    print(check_distribution(selection))

    m_, s_ = method_maximum_likelihood(selection)
    print('Метод максимального правдоподобия')
    print('mu=' + str(m_) + ', sigma=' + str(s_))

    distribution = sp.gamma.rvs(1.03567, size=len(selection))
    plot_teor_empirical(selection, distribution)

    plot_teor_empirical_cdf(selection, distribution)












