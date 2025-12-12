import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_incidence_over_time(df, stages=None, x_col="Year", y_col="Observed Count", stage_col="Stage"):
    """
    Plot cancer incidence counts over time for selected stages.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing year, observed count, and stage columns.
    stages : list of str, optional
        List of stage names to include in the plot. Defaults to the 4 classic stages.
    x_col : str
        Column name for the x-axis (default = "Year").
    y_col : str
        Column name for the y-axis (default = "Observed Count").
    stage_col : str
        Column name representing cancer stage (default = "Stage").

    Returns
    -------
    matplotlib Axes object
    """

    # Default list of stages
    if stages is None:
        stages = ["Stage I", "Stage II", "Stage III", "Stage IV"]

    # Filter the dataframe
    df_filtered = df[df[stage_col].isin(stages)]

    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=df_filtered,
        x=x_col,
        y=y_col,
        hue=stage_col
    )

    ax.set_title("Cancer Incidence Over Years by Stage")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(title=stage_col)

    plt.tight_layout()
    plt.show()

    return ax



def plot_stacked_bar_by_stage(
    df,
    exclude_stages=None,
    x_col="Year",
    stage_col="Stage",
    value_col="Observed Count",
    figsize=(12, 7),
    title="Stacked Bar Chart of Cancer Incidence by Stage Over Years"
):
    """
    Plot a stacked bar chart of cancer incidence by stage over years.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing stage, year, and observed count.
    exclude_stages : list, optional
        List of stages to exclude from the plot (default: ['TotalKnown', 'Total', 'Unknown stage']).
    x_col : str
        Column name for x-axis (default: 'Year').
    stage_col : str
        Column name representing stages (default: 'Stage').
    value_col : str
        Column name representing observed counts (default: 'Observed Count').
    figsize : tuple
        Figure size (default: (12, 7)).
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object of the stacked bar chart.
    """

    # Default stages to exclude
    if exclude_stages is None:
        exclude_stages = ['TotalKnown', 'Total', 'Unknown stage']

    # Filter dataframe
    df_filtered = df[~df[stage_col].isin(exclude_stages)]

    # Pivot the data
    pivot_df = df_filtered.pivot_table(
        index=x_col,
        columns=stage_col,
        values=value_col,
        aggfunc='sum'
    )

    # Plot
    ax = pivot_df.plot(kind='bar', stacked=True, figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(value_col)
    ax.legend(title=stage_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return ax
