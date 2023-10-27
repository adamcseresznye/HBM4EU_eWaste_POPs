from itertools import combinations
from pathlib import Path
from typing import List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import kstest, mannwhitneyu, shapiro

import utils


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
        **kwargs,  # Pass any additional keyword arguments to the seaborn boxplot
    )


# code is based on https://github.com/4dcu-be/CodeNuggets/blob/main/Post%20hoc%20tests%20with%20statannotations.ipynb
def get_pairs_values_for_posthoc_dunn(
    data: pd.DataFrame,
    value_vars: str,
    id_vars: str = "sub_category",
    p_adjust: str = "fdr_bh",
    segment_by=None,
):
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


def return_sns_box(df, x, y, ax, order=None):
    """
    This function creates and returns a seaborn boxplot on a given DataFrame and axes.

    Inputs:
    df (pd.DataFrame): DataFrame containing the data to be plotted
    x (str): The column name in the DataFrame to be used as the x-axis variable
    y (str): The column name in the DataFrame to be used as the y-axis variable
    ax (matplotlib axes object): The axes on which the plot will be drawn

    Outputs:
    seaborn boxplot object: The boxplot object that is returned can be further customized using seaborn or matplotlib functions
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
    )


def plot_diagnostics(
    df: pd.DataFrame,
    model,
    dependent_variable: str,
    figsize: tuple = (5, 5),
    save_to_disk: bool = False,
    folder_path: Path = utils.Configuration.PLOTS,
) -> None:
    """
    Plot diagnostics for a Generalized Linear Model (GLM).

    This function generates a diagnostic plot for a GLM, including observed vs. fitted values,
    a scatterplot of residuals vs. fitted values, a histogram of residuals, and a Q-Q plot of residuals.

    Args:
        df (pd.DataFrame): The DataFrame containing the data used in the model.
        model (sm.genmod.generalized_linear_model.GLM): The fitted GLM model.
        dependent_variable (str): The name of the dependent variable in the DataFrame.
        figsize (tuple, optional): The size of the figure (width, height). Defaults to (5, 5).
        save_to_disk (bool, optional): Whether to save the diagnostic plot to disk. Defaults to False.
        folder_path (Path, optional): The folder path where the diagnostic plot will be saved.
            Defaults to the 'PLOTS' directory specified in the 'utils.Configuration'.

    Returns:
        None

    Note:
        - The diagnostic plot is displayed but not saved by default. Set `save_to_disk` to True to save it.
        - The Shapiro-Wilk normality test is performed on the residuals, and the results are printed.

    Example:
        To plot diagnostics for a GLM model and save the plot to disk:
        >>> plot_diagnostics(df, model, "target_variable", figsize=(8, 6), save_to_disk=True,
                            folder_path=Path("/path/to/your/directory"))
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    fitted_values = model.fittedvalues

    axs[0, 0].scatter(np.sqrt(df[dependent_variable]), fitted_values)
    axs[0, 0].set_xlabel("Observed values")
    axs[0, 0].set_ylabel("Fitted values")

    residuals = model.resid_response

    axs[0, 1].scatter(fitted_values, residuals)
    axs[0, 1].set_xlabel("Fitted values")
    axs[0, 1].set_ylabel("Residuals")

    sns.histplot(residuals, kde=True, ax=axs[1, 0])
    axs[1, 0].set_xlabel("Residuals")
    axs[1, 0].set_ylabel("Frequency")

    sm.qqplot(residuals, line="s", ax=axs[1, 1])

    plt.suptitle(f"Dependent variable: {dependent_variable}")
    plt.tight_layout()

    if save_to_disk:
        plt.savefig(
            folder_path.joinpath(str(f"{dependent_variable}_GLM_diagnostics.png")),
            dpi=600,
        )

    shapiro_stat, shapiro_pvalue = stats.shapiro(residuals)
    print("Shapiro-Wilk test statistic:", shapiro_stat)
    print("Shapiro-Wilk test p-value:", shapiro_pvalue)


def is_detected(x, matrix):
    """
    Checks if the provided object, typically used with apply, goes through the columns.
    If the column is named 'BDE 209' and the column's value is 5, it returns True.
    In all other columns, it returns True if the instance is 0.5, otherwise False.

    Args:
        x: The object or value to be checked.

    Returns:
        True if x represents a 'BDE 209' column with a value of 5,
        or if x is 0.5 in any other column. Otherwise, returns False.

    Example Usage:
    --------------
    import pandas as pd

    # Sample DataFrame
    data = {
        'Column1': [5, 0.5, 5, 1],
        'Column2': [0.5, 0.5, 0.5, 5],
        'BDE 209': [5, 0.5, 1, 5],
    }

    df = pd.DataFrame(data)

    # Apply the function to each column
    results = df.apply(is_detected_wristband)

    print(results)
    """
    if matrix == "wristband":
        if x.name == "BDE 209" or x.name == "BDE-209":
            return x == 5
        else:
            return x == 0.5
    if matrix == "dust":
        if x.name == "BDE 209" or x.name == "BDE-209":
            return x == 0.5
        else:
            return x == 0.05
