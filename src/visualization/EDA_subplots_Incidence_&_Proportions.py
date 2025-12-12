import matplotlib.pyplot as plt

def plot_cancer_sites_subplots(df_individual, 
                               exclude_stages=None,
                               ncols=4,
                               figsize_per_plot=(6, 6)):
    """
    Create subplots showing stage-level cancer incidence over time 
    for each individual cancer site in the dataframe.

    Parameters
    ----------
    df_individual : pandas.DataFrame
        MultiIndex DataFrame indexed by ['Cancer Site', 'Year'] 
        containing stage columns.

    exclude_stages : list, optional
        List of columns (stages) to exclude from plotting.
        Default excludes totals, unknowns, and combined stages.

    ncols : int
        Number of subplot columns. Default = 4.

    figsize_per_plot : tuple
        Size (width, height) of each subplot cell.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes
    """

    # Default excluded stages
    if exclude_stages is None:
        exclude_stages = [
            'Total', 'TotalKnown', 'Unknown stage',
            'Early stages (I+II)', 'Late stages (III+IV)'
        ]

    # Get unique cancer sites
    individual_sites = df_individual.index.get_level_values('Cancer Site').unique()
    num_sites = len(individual_sites)

    # Determine rows
    nrows = -(-num_sites // ncols)  # Same as math.ceil(num_sites / ncols)

    # Create subplots grid
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        sharey=True
    )
    axes = axes.flatten()

    # Plot each site
    for i, site in enumerate(individual_sites):
        site_data = df_individual.xs(site, level='Cancer Site')

        # Drop excluded stages
        site_data = site_data.drop(columns=exclude_stages, errors='ignore')

        # Plot
        site_data.plot(kind='bar', ax=axes[i], legend=False)
        axes[i].set_title(site)
        axes[i].set_xlabel('Year')
        axes[i].set_ylabel('Observed Count')

    # Build shared legend using first valid subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Stage', loc='upper center',
               ncol=len(labels), frameon=False)

    # Layout with space for legend
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()

    return fig, axes




def plot_stage_proportions_facetgrid(df_stage_props,
                                     stage_columns=None,
                                     id_vars=None,
                                     col_wrap=4,
                                     height=4,
                                     palette="Set2"):
    """
    Create a FacetGrid of bar plots showing stage proportions over time
    for each cancer site.

    Parameters
    ----------
    df_stage_props : pandas.DataFrame
        A dataframe containing columns for cancer sites, years, and
        stage proportion columns.

    stage_columns : list, optional
        Columns containing stage proportion values.
        Default = ['Stage I', 'Stage II', 'Stage III', 'Stage IV'].

    id_vars : list, optional
        Identifier variables to keep during melting.
        Default = ['Cancer Site', 'Year'].

    col_wrap : int
        Number of subplot columns. Default = 4.

    height : float
        Height of each subplot.

    palette : str
        Seaborn color palette for bars.

    Returns
    -------
    g : seaborn.axisgrid.FacetGrid
        The FacetGrid object for further customization.
    """

    # Default stage columns
    if stage_columns is None:
        stage_columns = ["Stage I", "Stage II", "Stage III", "Stage IV"]

    # Default id variables
    if id_vars is None:
        id_vars = ["Cancer Site", "Year"]

    # Ensure MultiIndex is flattened
    df_stage_proportions = df_stage_props.reset_index(drop=False)

    # Melt into long format
    df_melted = df_stage_proportions.melt(
        id_vars=id_vars,
        value_vars=stage_columns,
        var_name="Stage",
        value_name="Proportion"
    )

    # FacetGrid plot
    g = sns.FacetGrid(
        df_melted,
        col="Cancer Site",
        col_wrap=col_wrap,
        height=height,
        sharey=True
    )

    g.map_dataframe(
        sns.barplot,
        x="Year",
        y="Proportion",
        hue="Stage",
        palette=palette
    )

    # Add legend & labels
    g.add_legend(title="Stage")
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Year", "Proportion of Total Known Cases")

    plt.tight_layout()
    plt.show()

    return g
