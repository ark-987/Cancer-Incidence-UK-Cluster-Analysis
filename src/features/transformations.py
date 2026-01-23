import pandas as pd

stages_to_include = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
sites_to_exclude = ['All Cancers Combined', 'All Cancers Combined (excluding Lung and Prostate)', 'All Cancers Combined (excluding Lung)', 'All Cancers Combined (excluding Prostate)']


def convert_to_datetime(df):
        """
        Parse the 'Date' column to datetime, drop invalid dates,
        and add 'Year' and 'Month' columns.
        """
    
        df=df.copy() #to avoid modifying original DataFrame
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce') #coerce errors to NaT to avoid crashing if date format cannot be parsed
        df = df.dropna(subset=['Date']) #removes rows where 'Date' could not be parsed
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month  
        return df

def aggregate_stages(df:pd.DataFrame):
        """
        Aggregate observed counts by year, cancer site, and stage.
        """
        return df.groupby(['Year', 'Cancer Site','Stage'])['Observed Count'].sum().unstack()


def create_proportions(df_grouped):
        """
        Calculate stage proportions relative to total known cases
        and exclude specified cancer sites.
        """
        df_stage_proportions=df_grouped[stages_to_include].div(df_grouped['TotalKnown'], axis=0)
        return df_stage_proportions.drop(sites_to_exclude, level='Cancer Site', errors='ignore')
    
def transform_stage_data(df):
        """
        Run the full pipeline to clean dates, aggregate stages,
        and compute stage proportions.
        """
        return (df
        .pipe(convert_to_datetime)
        .pipe(aggregate_stages)
        .pipe(create_proportions))
      
