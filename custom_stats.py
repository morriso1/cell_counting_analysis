from scipy import stats
import scikit_posthocs as sp
from statsmodels import formula
from statsmodels import api


def kruskal_scipy_stats_tidy_df_wrapper(
    tidy_df, indep_var="sample_id", dep_var="mean_intensity"
):
    """
    Task
    ----
    Perform kruskal wallis to determine if significant difference between groups.

    Input
    -----
    Takes tidy DataFrame, independent variable (str) and dependent variable (str).

    Returns
    -------
    statistic : float
    The Kruskal-Wallis H statistic, corrected for ties.

    pvalue : float
    The p-value for the test using the assumption that H has a chi square distribution. 
    """

    data = [
        tidy_df.loc[ids, dep_var].values
        for ids in tidy_df.groupby(indep_var).groups.values()
    ]

    return stats.kruskal(*data)
