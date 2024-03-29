import math as m
import pandas as pd
import numpy as np
from numpy import array

import scipy.stats as ss
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.power import TTestIndPower, tt_ind_solve_power
from statsmodels.stats.power import NormalIndPower

import matplotlib.pyplot as plt
import seaborn as sns


def calc_Cohen_d_samples(sample_1: list, sample_2: list) -> float:
    """
    Finds Cohen distance between means of two samples.

    Arguments:
        sample_1 (list): List of values in the first group.
        sample_2 (list): List of values in the second group.
    Returns:
        Cohen's index (float): distance.
    """
    S_pooled_squared = (((len(sample_1) - 1) * np.std(sample_1) ** 2 + (len(sample_2) - 1) * np.std(sample_2) ** 2) /
                        ((len(sample_1) + len(sample_2)) - 2))
    return round(abs(np.mean(sample_1) - np.mean(sample_2)) / np.sqrt(S_pooled_squared), 5)


def calc_Cohen_d_by_params(mean_control: float, mean_test: float,
                           sd_control: float, sd_test: float) -> float:
    """
    Finds Cohen distance between means of two samples.
    Warning: you can calc d this way, when two samples are the same size.

    Arguments:
        mean_control (float): mean value of a control group (train).
        mean_test (float): mean value of a test group.
        sd_control (float): standard deviation of a control group.
        sd_test (float): standard deviation of a test group.
    Returns:
        Cohen's index (float): distance.
    """
    S_pooled = np.sqrt((sd_control ** 2 + sd_test ** 2) / 2)
    return round(abs(mean_control - mean_test) / S_pooled, 5)


def calc_sample_size_ztest_analysis(p1: float, p2: float, rel_mde: float,
                         power=0.8, alpha=0.05, plot=False) -> int:
    """
    Calcs sample size with ztest power analysis.

    Arguments:
        p1 (float): proportion for old group.
        p2 (float): proportion for new group.
        rel_mde (float): relative increase, relative MDE
        power (float): power of experiment.
        alpha (float): significance level.
        plot (bool): show plot or not.
    Returns:
        sample_size (int): output sample size (depends on power and alpha).
    """
    # effect size standardization
    d = sm.stats.proportion_effectsize(p1, p2)
    n = sms.NormalIndPower().solve_power(
        effect_size=d,
        power=power,
        alpha=alpha,
        ratio=1
    )
    if plot:
        print(f"Alpha={alpha}")
        print(f'Размер выборки для каждой группы, необходимый, чтобы задетектить эффект в {100 * rel_mde}%: ', n)
        ztest_power = sms.NormalIndPower()
        sample_sizes = np.arange(50, round(1.5 * n), 10)
        (ztest_power.plot_power(dep_var='nobs', nobs=sample_sizes,
                                effect_size=[d], alpha=alpha, title=f'Анализ мощности при alpha = {alpha}'))
        plt.axhline(power, linestyle='--', label='выбранная мощность')
        plt.axvline(n, linestyle='--', color='orange', label='требуемая выборка')
        plt.ylabel('Мощность теста')
        plt.xlabel('Размер выборки')
        plt.grid(alpha=0.08)
        plt.legend()
        plt.show()
    return n
