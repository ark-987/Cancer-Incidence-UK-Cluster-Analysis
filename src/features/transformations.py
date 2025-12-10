import pandas as pd
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month  

df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year

df_stages_for_site=df.groupby(['Year', 'Cancer Site','Stage'])['Observed Count'].sum().unstack()

stages_to_include = ['Stage I', 'Stage II', 'Stage III', 'Stage IV'] 

#Create the proportions DataFrame
df_stage_props = df_stages_for_site[stages_to_include].div(df_stages_for_site['TotalKnown'], axis=0)

Sites_to_exclude = ['All Cancers Combined', 'All Cancers Combined (excluding Lung and Prostate)', 'All Cancers Combined (excluding Lung)', 'All Cancers Combined (excluding Prostate)']
df_stage_props= df_stage_props.drop(Sites_to_exclude, level='Cancer Site', errors='ignore')