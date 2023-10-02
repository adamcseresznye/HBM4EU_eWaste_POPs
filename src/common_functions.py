from itertools import combinations
from pathlib import Path
from typing import List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from scipy.stats import kstest, mannwhitneyu, shapiro

from data import utils


def create_boxplot(
    df: pd.DataFrame, x: str, y: str, ax: plt.Axes, order=None, *args, **kwargs
) -> sns.boxplot:
    """
    This function creates and returns a seaborn boxplot on a given DataFrame and axes.

    Inputs:
    :param df: pd.DataFrame - DataFrame containing the data to be plotted
    :param x: str - The column name in the DataFrame to be used as the x-axis variable
    :param y: str - The column name in the DataFrame to be used as the y-axis variable
    :param ax: plt.Axes - The axes on which the plot will be drawn
    :param order: list, optional - The order in which the boxes should appear (default: None)

    Outputs:
    :return: sns.boxplot - The boxplot object that is returned can be further customized using seaborn or matplotlib functions
    """
    return sns.boxplot(
        data=df,
        x=x,
        y=y,
        palette=sns.color_palette("colorblind"),
        flierprops=dict(
            marker="o", markersize=2, markerfacecolor="white", linestyle="none"
        ),
        linewidth=1,
        width=0.5,
        order=order,
        ax=ax,
        **kwargs  # Pass any additional keyword arguments to the seaborn boxplot
    )


# code is based on https://github.com/4dcu-be/CodeNuggets/blob/main/Post%20hoc%20tests%20with%20statannotations.ipynb
def get_pairs_values_for_posthoc_dunn(
    data: pd.DataFrame,
    value_vars: str,
    id_vars: str = "sub_category",
    p_adjust: str = "fdr_bh",
    segment_by: Optional[List] = None,
) -> Tuple[List[Tuple[str, str]], List[float]]:
    """
    This function performs a post-hoc Dunn test for multiple comparisons on a given DataFrame and returns the pairs and p-values of the significant comparisons.

    Inputs:
    data (pd.DataFrame): DataFrame containing the data to be analyzed
    value_vars (str): The variables of interest to be compared between the groups
    id_vars (str): The column name in the DataFrame used to define the groups. Default is 'sub_category'
    p_adjust (str): The method used to adjust p-values. Default is 'fdr_bh'
    segment_by (list): A list of values to filter the data by before the analysis is performed. Default is None

    Outputs:
    tuple: A tuple containing the list of the pairs of significant comparisons and a list of their corresponding p-values

    """
    if segment_by is not None:
        data = data.query("sub_category.isin(@segment_by)")

    data_melted = data.melt(id_vars=id_vars, value_vars=value_vars)
    test_result = sp.posthoc_dunn(
        a=data_melted, val_col="value", group_col=id_vars, p_adjust=p_adjust
    )

    remove = np.tril(np.ones(test_result.shape), k=0).astype("bool")
    test_result[remove] = np.nan
    molten_df = test_result.melt(ignore_index=False).reset_index().dropna()
    filtered_molten_df = molten_df[molten_df.value < 0.05]

    pairs = [(i[1]["index"], i[1]["variable"]) for i in filtered_molten_df.iterrows()]
    p_values = [i[1]["value"] for i in filtered_molten_df.iterrows()]

    return pairs, p_values


def get_pairs_values_for_mannwhitneyu(
    data: pd.DataFrame, value_var: str, id_vars: list, col: str
):
    """
    This function performs a Mann-Whitney U test on two groups defined by the first two elements of a given list of identifiers and a specified column from a given DataFrame.

    Inputs:
    data (pd.DataFrame): DataFrame containing the data to be analyzed
    value_var (str): The variable of interest to be compared between the two groups
    id_vars (list): List of identifiers to define the two groups to be compared. The subgroups within the group.
    col (str): The column name in the DataFrame used to define the two groups

    Outputs:
    tuple: A tuple containing the list of the two groups' identifiers, and a list containing the p-value of the Mann-Whitney U test

    """

    pairs = [(id_vars[0], id_vars[1])]

    array1 = (
        data.loc[data[col] == pairs[0][0], lambda x: x.columns.isin([value_var])]
    ).values.flatten()
    array2 = (
        data.loc[data[col] == pairs[0][1], lambda x: x.columns.isin([value_var])]
    ).values.flatten()

    p_value = mannwhitneyu(array1, array2, nan_policy="omit").pvalue

    return pairs, [p_value]


def get_pairs_values_for_mannwhitneyu_multiple(
    data: pd.DataFrame, value_var: str, id_vars: List[str], col: str
) -> Tuple[List[Tuple[str, str]], List[float]]:
    """
    This function performs the Mann-Whitney U test for multiple pairwise comparisons between groups defined by a specified column of a given DataFrame.
    It returns the pairs of groups and their corresponding p-values for which the test resulted in a significant difference.

    Inputs:
    data (pd.DataFrame): DataFrame containing the data to be analyzed
    value_var (str): The variable of interest to be compared between the groups
    id_vars (list): The list of identifiers to define the groups to be compared
    col (str): The column name in the DataFrame used to define the groups

    Outputs:
    tuple: A tuple containing the list of the pairs of significant comparisons and a list of their corresponding p-values

    """

    mask = np.isin(data.loc[:, col].unique(), id_vars)
    groups = data.loc[:, col].unique()[mask]
    combination = [x for x in combinations(groups, 2)]

    p_values = []
    IDs = []

    for idx, i in enumerate(combination):
        array1 = (
            data.loc[
                data[col] == combination[idx][0], lambda x: x.columns.isin([value_var])
            ]
        ).values.flatten()
        array2 = (
            data.loc[
                data[col] == combination[idx][1], lambda x: x.columns.isin([value_var])
            ]
        ).values.flatten()

        p_value = mannwhitneyu(array1, array2, nan_policy="omit").pvalue

        if p_value < 0.05:
            p_values.append(p_value)
            IDs.append(i)
    return IDs, p_values


def col_name_changer(df, what: str, how: str):
    """
    Rename columns of a Pandas DataFrame by replacing substrings.

    This function replaces specific substrings in the column names of a Pandas DataFrame
    with another substring using regular expressions. It operates in place and modifies
    the column names of the input DataFrame.

    Parameters:
    df (pd.DataFrame): The Pandas DataFrame whose column names will be modified.
    what (str): The substring to search for in column names.
    how (str): The replacement substring to replace 'what' with in column names.

    Returns:
    pd.DataFrame: The DataFrame with modified column names.

    Example:
    >>> df = pd.DataFrame({'A_Name': [1, 2], 'B_Name': [3, 4]})
    >>> col_name_changer(df, '_Name', '_Value')
    >>> df.columns
    Index(['A_Value', 'B_Value'], dtype='object')
    """
    df.columns = df.columns.str.replace(what, how, regex=True)
    return df


def count_years_worked(df):
    """
    Calculate the number of years worked based on a DataFrame of start and end years.

    This function calculates the number of years worked for each row in a DataFrame.
    The DataFrame is assumed to have columns representing start and end years, and
    each row corresponds to a period of employment. The result is a Pandas Series
    containing the calculated years worked for each row.

    Parameters:
    df (pd.DataFrame): The DataFrame containing start and end years.

    Returns:
    pd.Series: A Pandas Series containing the calculated years worked for each row.

    Example:
    >>> df = pd.DataFrame({'Start_Year': [2010, 2005, 2015], 'End_Year': [2015, 2012, 2020]})
    >>> count_years_worked(df)
    0    5
    1    7
    2    5
    dtype: int64
    """
    years_worked = []
    for idx, row in df.iterrows():
        if row.isna().all():
            years_worked.append(1)
        elif row.notna().sum() % 2 == 0:
            years_worked.append(row.nlargest(2).iloc[0] - row.nlargest(2).iloc[1])
        elif row.notna().sum() % 2 == 1:
            years_worked.append(2021 - row.nlargest(2).iloc[0])
    return pd.Series(years_worked)


def transform_data(df):
    """
    Apply common data transformations to a DataFrame.

    This function takes a DataFrame and performs a series of common data transformations:
    1. Removes the first row and resets the index.
    2. Replaces specific values ('No', 'no', 'Yes', 'Sorting e-waste') with numeric values.
    3. Converts columns to numeric, handling non-numeric values by replacing them with NaN.
    4. Fills NaN values with 0.

    Parameters:
    df (pd.DataFrame): The DataFrame to be transformed.

    Returns:
    pd.DataFrame: The transformed DataFrame.

    Example:
    >>> df = pd.DataFrame({'A': ['Yes', 'No'], 'B': ['1', '2']})
    >>> transform_data(df)
       A  B
    0  1  1
    1  0  2
    """
    return (
        df.iloc[1:, :]
        .reset_index(drop=True)
        .replace(
            {"No": 0, "no": 0, "Yes": 1, "Sorting e-waste": 0}
        )  # Handle non-numeric value
        .apply(
            pd.to_numeric, errors="coerce"
        )  # Use errors="coerce" to convert non-numeric to NaN
        .fillna(0)
    )
