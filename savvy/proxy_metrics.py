from typing import Any
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint


def find_correlation_CQ_vars(quant_value: pd.Series | list, binary_mask: pd.Series | list,
                             plot=False, show_conclusion=False) -> dict:
    """
    Finds correlation between categorical and quantitative variables.

    Arguments:
        quant_value (pd.Series | list): quantitative variable to find correlation.
        binary_mask (pd.Series | list): binary value to find correlation.
        plot (bool): if True - plot the graphic.
        show_conclusion (bool): show conclusion results.
    Returns:
         point_biserial_R (float): point-biserial correlation (parametric).
         kruskal_pval (float): test whether samples from the same distribution (non-parametric).
         ttest_pval (float): finds the difference between the response
                             of two groups is statistically significant or not (parametric).
    """

    # categorise quantitative metrics into groups
    # based on whether there was a targeted conversion or not
    group_1 = quant_value[binary_mask]  # binary = 1
    group_2 = quant_value[~binary_mask]  # binary = 0

    pointbiserialr = ss.pointbiserialr(quant_value, binary_mask)[0]
    kruskal_pval = ss.kruskal(group_1, group_2)[1]
    ttest_pval = ss.ttest_ind(group_1, group_2)[1]

    if show_conclusion:
        print('Mean quant value for group (binary=1) =', group_1.mean())
        print('Mean quant value for group (binary=0) =', group_2.mean())
        print('Median quant value for group (binary=1) =', group_1.median())
        print('Median quant value for group (binary=0) =', group_2.median())
        print('Kruskal-Wallis H Test p-value =', kruskal_pval)
        print('T-test p-value =', ttest_pval)
        print('point biserial correlation =', pointbiserialr)

    if plot:
        plt.figure(figsize=(12, 4))
        sns.kdeplot(group_1, color='red', label='Binary = True')
        sns.kdeplot(group_2, color='blue', label='Binary = False')
        plt.axvline(x=group_1.mean(), linestyle='--', color='red')
        plt.axvline(x=group_2.mean(), linestyle='--', color='blue')
        plt.title('Groups distributions')
        plt.legend()

    return {'point_biserial_R': pointbiserialr,
            'kruskal_pval': kruskal_pval,
            'ttest_pval': ttest_pval}


def _find_correlation_CQ_vars_by_names(data: pd.DataFrame,
                                       quant_column_name: str,
                                       binary_column_name: str, target_value: Any,
                                       plot=False, show_conclusion=False) -> dict:
    """
    Finds correlation between categorical and quantitative variables.

    Arguments:
        data (pd.DataFrame): input dataset.
        quant_column_name (str): quantitative variable to find correlation.
        binary_column_name (str): binary value to find correlation.
        target_value (Any): target value.
        plot (bool): if True - plot the graphic.
        show_conclusion (bool): show conclusion results.
    Returns:
         point_biserial_R (float): point-biserial correlation (parametric).
         kruskal_pval (float): test whether samples from the same distribution (non-parametric).
         ttest_pval (float): finds the difference between the response
                             of two groups is statistically significant or not (parametric).
    """

    # categorise quantitative metrics into groups
    # based on whether there was a targeted conversion or not
    quant_value = data[quant_column_name]
    binary_mask = data[binary_column_name]==target_value
    group_1 = quant_value[binary_mask]  # binary = 1
    group_2 = quant_value[~binary_mask]  # binary = 0

    pointbiserialr = ss.pointbiserialr(quant_value, binary_mask)[0]
    kruskal_pval = ss.kruskal(group_1, group_2)[1]
    ttest_pval = ss.ttest_ind(group_1, group_2)[1]

    if show_conclusion:
        print('Mean quant value for group (binary=1) =', group_1.mean())
        print('Mean quant value for group (binary=0) =', group_2.mean())
        print('Median quant value for group (binary=1) =', group_1.median())
        print('Median quant value for group (binary=0) =', group_2.median())
        print('Kruskal-Wallis H Test p-value =', kruskal_pval)
        print('T-test p-value =', ttest_pval)
        print('point biserial correlation =', pointbiserialr)

    if plot:
        plt.figure(figsize=(12, 4))
        sns.kdeplot(group_1, color='red', label='Binary = True')
        sns.kdeplot(group_2, color='blue', label='Binary = False')
        plt.axvline(x=group_1.mean(), linestyle='--', color='red')
        plt.axvline(x=group_2.mean(), linestyle='--', color='blue')
        plt.title('Groups distributions')
        plt.legend()

    return {'point_biserial_R': pointbiserialr,
            'kruskal_pval': kruskal_pval,
            'ttest_pval': ttest_pval}


def proxy_metrics_bins_analyzer(quant_value: pd.Series | list, binary_mask: pd.Series | list,
                                step: float, plot=True):
    """
    Finds optimal thresholds for making proxy-metric.

    Arguments:
        quant_value (pd.Series | list): quantitative variable to find correlation.
        binary_mask (pd.Series | list): binary value to find correlation.
        step (float): step for quant_value splitting.
        plot (bool): if True - plot the graphic.
    Returns:
        point_biserial_R (float): point_biserial value.
    """
    deciles_bins = np.quantile(quant_value, np.arange(0, 1, step))
    inds = np.digitize(quant_value, deciles_bins, right=False)  # split on bins
    df = pd.DataFrame({'quant_value': quant_value, 'binary': binary_mask, 'bins': inds})
    inference = df.groupby('bins').mean()

    point_biserial_R = find_correlation_CQ_vars(quant_value=quant_value,
                                                binary_mask=binary_mask,
                                                show_conclusion=True, plot=True)['point_biserial_R']
    if plot:
        plt.figure(figsize=(12, 4))
        sns.lineplot(y=inference['quant_value'], x=inference.index * step, color='red', label='quant_value')
        plt.axhline(y=quant_value.median(), linestyle='--', color='red')
        plt.text(x=inds.mean() * step, y=np.quantile(quant_value, 0.95),
                 s='point_biserial_R = {}'.format(point_biserial_R), )
        ax2 = plt.twinx()
        sns.lineplot(y=inference['binary'], x=inference.index * step, color='blue', ax=ax2, label='binary share')
        plt.axhline(y=binary_mask.mean(), linestyle='--', color='blue')
        plt.title("Graph of changes in input parameters by deciles")
        plt.legend()
    return point_biserial_R


def calc_CC_association_Cramer_V(confusion_matrix: pd.DataFrame) -> np.float64:
    """
    Calculate Cramers V statistic for categorial-categorial association.
    It uses correction from Bergsma and Wicher.

    Arguments:
        confusion_matrix (pd.DataFrame): confusion matrix of your data.
    Returns:
        cramers_v (np.float64): Cramers V value.
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def find_max_Cramer_V_threshold(quant_value: pd.Series | list, binary_mask: pd.Series | list,
                                plot=False, step=1) -> dict:
    """
    Finds threshold with max association Cramer V coefficient.

    Arguments:
        quant_value (pd.Series | list): quantitative variable to find correlation.
        binary_mask (pd.Series | list): binary value to find correlation.
        step (float): step for quant_value splitting.
        plot (bool): if True - plot the graphic.
    Returns:
        (dict):
            threshold (float): value for max association Cramer V coefficient.
            cramer_v_coefficient (float): max value of Cramer V coefficient.
    """
    # define vector of input quant values, where we're going to find max correlation
    arange = np.arange(min(quant_value), max(quant_value), step)
    cramers_v_list = []

    for i in arange:
        confusion_matrix = pd.crosstab(binary_mask, quant_value > i)
        cramers_v_i = calc_CC_association_Cramer_V(confusion_matrix.values)
        cramers_v_list.append(cramers_v_i)

    max_corr_quant_value = arange[np.argmax(np.array(cramers_v_list))]

    if plot:
        sns.lineplot(x=arange, y=cramers_v_list)
        plt.axhline(y=np.array(cramers_v_list).max(), linestyle='--', color='red')
        plt.axvline(x=max_corr_quant_value, linestyle='--', color='red')
        plt.xlabel('quant values')
        plt.ylabel('Cramer-V')

    return ({'threshold': max_corr_quant_value,
             'cramer_v_coefficient': max(cramers_v_list)})


def ab_test_simulation_proxy_metrics(data: pd.DataFrame,
                                     categorical_column_name: str, categorical_target_value: Any,
                                     numeric_column_name: str,
                                     effect=0.1, n_sim=10000, alpha=0.01, pvalue_threshold=0.05) -> None:
    """
    Starts A/B simulation for input parameters.

    Arguments:
        data (pd.DataFrame): input data.
        categorical_column_name (str): name of binary categorical column.
        categorical_target_value (Any): filter categorical column by target value.
        numeric_column_name (str): name of quant column.
        effect (float, optional): effect size.
        n_sim (int, optional): number of simulations.
        alpha (float, optional): significance level or probability of first-order error.
        pvalue_threshold (float, optional): to filter results of z-test.
    Returns:
        None, just prints simulations results.
    """
    ab_binary_ztest_pvalue_list = []
    ab_quant_ztest_pvalue_list = []
    a_group_sample_size_prop_list = []
    quant_target_level = 0

    for i in range(n_sim):
        sample_mask = ss.bernoulli.rvs(0.5, size=len(data)) == 1

        binary_mask_group_a = (data[categorical_column_name] == categorical_target_value)[sample_mask]
        binary_mask_group_b = (data[categorical_column_name] == categorical_target_value)[~sample_mask]
        quant_value_group_a = (data[numeric_column_name])[sample_mask]
        quant_value_group_b = (data[numeric_column_name])[~sample_mask]

        binary_nobs_group_a = len(binary_mask_group_a)
        binary_nobs_group_b = len(binary_mask_group_b)
        quant_nobs_group_a = len(quant_value_group_a)
        quant_nobs_group_b = len(quant_value_group_b)

        sample_size_prop = binary_nobs_group_a / len(data)
        a_group_sample_size_prop_list.append(sample_size_prop)

        binary_counts_group_a = binary_mask_group_a.sum()
        binary_counts_group_b = round(binary_mask_group_b.sum() * (1 + effect))  # add effect
        quant_counts_group_a = (quant_value_group_a > quant_target_level).sum()
        quant_counts_group_b = round((quant_value_group_b > quant_target_level).sum() * (1 + effect))  # add effect

        binary_z_score_i, binary_ztest_pvalue_i = sms.proportions_ztest(
            count=[binary_counts_group_a, binary_counts_group_b],
            nobs=[binary_nobs_group_a, binary_nobs_group_b])

        quant_z_score_i, quant_ztest_pvalue_i = sms.proportions_ztest(
            count=[quant_counts_group_a, quant_counts_group_b],
            nobs=[quant_nobs_group_a, quant_nobs_group_b])

        ab_binary_ztest_pvalue_list.append(binary_ztest_pvalue_i)
        ab_quant_ztest_pvalue_list.append(quant_ztest_pvalue_i)

    binary_power_ci = proportion_confint((np.array(ab_binary_ztest_pvalue_list) <= pvalue_threshold).sum(),
                                         n_sim, alpha=0.01)
    quant_power_ci = proportion_confint((np.array(ab_quant_ztest_pvalue_list) <= pvalue_threshold).sum(),
                                        n_sim, alpha=alpha)

    print('Effect =', effect)
    print('AVG sample size A group proportion =', np.mean(a_group_sample_size_prop_list))
    print('AVG sample size A group =', round(np.mean(a_group_sample_size_prop_list) * len(data)))
    print('AVG sample size B group =', round((1 - np.mean(a_group_sample_size_prop_list)) * len(data)))
    print('------------------')
    print('Target conversion A group = ', binary_counts_group_a / binary_nobs_group_a)
    print('Target conversion B group = ', binary_counts_group_b / binary_nobs_group_b)
    print('Proxy conversion A group = ', quant_counts_group_a / quant_nobs_group_a)
    print('Proxy conversion B group = ', quant_counts_group_b / quant_nobs_group_b)
    print('------------------')
    print('Мощность АБ-теста по целевой метрике',
          (np.array(ab_binary_ztest_pvalue_list) <= pvalue_threshold).sum() / n_sim)
    print('99%-процентный доверительный интервал для целевой метрики', binary_power_ci)
    print('Мощность АБ-теста по прокси метрике',
          (np.array(ab_quant_ztest_pvalue_list) <= pvalue_threshold).sum() / n_sim)
    print('99%-процентный доверительный интервал для прокси метрики', quant_power_ci)
