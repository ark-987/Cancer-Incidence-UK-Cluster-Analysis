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
