"""
Historical Geoid stats
"""
import numpy as np
import pandas as pd

pd.set_option('display.expand_frame_repr', False)

###############################################################################
# set up parameter

# Case 2: train based on 2
trainset1 = '../filtered_data/filter_train2_geoid.csv'

df_filename_2 = '../stat_feature/y_h_geo_first2_2.csv'
df_filename_5 = '../stat_feature/y_h_geo_first5_2.csv'
df_filename_11 = '../stat_feature/y_h_geo_first11_2.csv'
df_filename_all = '../stat_feature/y_h_geo_all_2_basedon_1.csv'

###############################################################################
# load dataset
###############################################################################
# load location and time only
fields = ['pickup_datetime', 'fare_amount', 'geoid_p', 'geoid_d']

# combine data
data = pd.read_csv(trainset1, usecols=fields)

# convert pickup datetime to informative features
data['pickup_datetime'] = data['pickup_datetime'].str.replace(" UTC", "")
data['pickup_datetime'] = pd.to_datetime(
    data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S'
)
data['hour_of_day'] = data.pickup_datetime.dt.hour
data["year"] = data.pickup_datetime.dt.year

# remove unused features
data = data.drop(['pickup_datetime'], axis=1)

###############################################################################
# data preprocessing
###############################################################################
# convert these float to string for matching
data['geoid_p'] = data['geoid_p'].astype(str)
data['geoid_d'] = data['geoid_d'].astype(str)

data['geoid_p_2'] = data['geoid_p'].str[:2]
data['geoid_d_2'] = data['geoid_d'].str[:2]

data['geoid_p_5'] = data['geoid_p'].str[:5]
data['geoid_d_5'] = data['geoid_d'].str[:5]

data['geoid_p_11'] = data['geoid_p'].str[:11]
data['geoid_d_11'] = data['geoid_d'].str[:11]

###############################################################################
# calculate the histrical stat based on these locations
###############################################################################
# get mean fare
df_stat_all1 = data.groupby(['geoid_p', 'geoid_d'],
                            as_index=False).agg(
    {'fare_amount': np.mean}
).reset_index()

df_stat_all1 = df_stat_all1.rename(columns={'fare_amount': 'mean_fare'})
df_stat_all1 = df_stat_all1[['geoid_p', 'geoid_d', 'mean_fare']]

# get median fare
df_stat_all2 = data.groupby(['geoid_p', 'geoid_d'],
                            as_index=False).agg(
    {'fare_amount': np.median}
).reset_index()

df_stat_all2 = df_stat_all2.rename(columns={'fare_amount': 'median_fare'})
df_stat_all2 = df_stat_all2[['geoid_p', 'geoid_d', 'median_fare']]

# get maximum fare
df_stat_all3 = data.groupby(['geoid_p', 'geoid_d'],
                            as_index=False).agg(
    {'fare_amount': max}
).reset_index()

df_stat_all3 = df_stat_all3.rename(columns={'fare_amount': 'max_fare'})
df_stat_all3 = df_stat_all3[['geoid_p', 'geoid_d', 'max_fare']]

# inner join to get the result
df_stat_all = pd.merge(df_stat_all1, df_stat_all2, how='inner',
                       on=['geoid_p', 'geoid_d']
                       ).drop_duplicates()

df_stat_all = pd.merge(df_stat_all, df_stat_all3, how='inner',
                       on=['geoid_p', 'geoid_d']
                       ).drop_duplicates()

###############################################################################
# calculate the histrical stat based on these locations (first 11)
###############################################################################
# get mean fare
df_stat_11_1 = data.groupby(['geoid_p_11', 'geoid_d_11'],
                            as_index=False).agg(
    {'fare_amount': np.mean}
).reset_index()

df_stat_11_1 = df_stat_11_1.rename(columns={'fare_amount': 'mean_fare'})
df_stat_11_1 = df_stat_11_1[['geoid_p_11', 'geoid_d_11', 'mean_fare']]

# get median fare
df_stat_11_2 = data.groupby(['geoid_p_11', 'geoid_d_11'],
                            as_index=False).agg(
    {'fare_amount': np.median}
).reset_index()

df_stat_11_2 = df_stat_11_2.rename(columns={'fare_amount': 'median_fare'})
df_stat_11_2 = df_stat_11_2[['geoid_p_11', 'geoid_d_11', 'median_fare']]

# get maximum fare
df_stat_11_3 = data.groupby(['geoid_p_11', 'geoid_d_11'],
                            as_index=False).agg(
    {'fare_amount': max}
).reset_index()

df_stat_11_3 = df_stat_11_3.rename(columns={'fare_amount': 'max_fare'})
df_stat_11_3 = df_stat_11_3[['geoid_p_11', 'geoid_d_11', 'max_fare']]

# inner join to get the result
df_stat_11 = pd.merge(df_stat_11_1, df_stat_11_2, how='inner',
                      on=['geoid_p_11', 'geoid_d_11']
                      ).drop_duplicates()

df_stat_11 = pd.merge(df_stat_11, df_stat_11_3, how='inner',
                      on=['geoid_p_11', 'geoid_d_11']
                      ).drop_duplicates()

###############################################################################
# calculate the histrical stat based on these locations (first 5)
###############################################################################
# get mean fare
df_stat_5_1 = data.groupby(['year', 'hour_of_day', 'geoid_p_5', 'geoid_d_5'],
                           as_index=False).agg(
    {'fare_amount': np.mean}
).reset_index()

df_stat_5_1 = df_stat_5_1.rename(columns={'fare_amount': 'mean_fare'})
df_stat_5_1 = df_stat_5_1[['year', 'hour_of_day', 'geoid_p_5', 'geoid_d_5', 'mean_fare']]

# get median fare
df_stat_5_2 = data.groupby(['year', 'hour_of_day', 'geoid_p_5', 'geoid_d_5'],
                           as_index=False).agg(
    {'fare_amount': np.median}
).reset_index()

df_stat_5_2 = df_stat_5_2.rename(columns={'fare_amount': 'median_fare'})
df_stat_5_2 = df_stat_5_2[['year', 'hour_of_day', 'geoid_p_5', 'geoid_d_5', 'median_fare']]

# get maximum fare
df_stat_5_3 = data.groupby(['year', 'hour_of_day', 'geoid_p_5', 'geoid_d_5'],
                           as_index=False).agg(
    {'fare_amount': max}
).reset_index()

df_stat_5_3 = df_stat_5_3.rename(columns={'fare_amount': 'max_fare'})
df_stat_5_3 = df_stat_5_3[['year', 'hour_of_day', 'geoid_p_5', 'geoid_d_5', 'max_fare']]

# inner join to get the result
df_stat_5 = pd.merge(df_stat_5_1, df_stat_5_2, how='inner',
                     on=['year', 'hour_of_day', 'geoid_p_5', 'geoid_d_5']
                     ).drop_duplicates()

df_stat_5 = pd.merge(df_stat_5, df_stat_5_3, how='inner',
                     on=['year', 'hour_of_day', 'geoid_p_5', 'geoid_d_5']
                     ).drop_duplicates()

###############################################################################
# calculate the histrical stat based on these locations (first 2)
###############################################################################
# get mean fare
df_stat_2_1 = data.groupby(['year', 'hour_of_day', 'geoid_p_2', 'geoid_d_2'],
                           as_index=False).agg(
    {'fare_amount': np.mean}
).reset_index()

df_stat_2_1 = df_stat_2_1.rename(columns={'fare_amount': 'mean_fare'})
df_stat_2_1 = df_stat_2_1[['year', 'hour_of_day', 'geoid_p_2', 'geoid_d_2', 'mean_fare']]

# get median fare
df_stat_2_2 = data.groupby(['year', 'hour_of_day', 'geoid_p_2', 'geoid_d_2'],
                           as_index=False).agg(
    {'fare_amount': np.median}
).reset_index()

df_stat_2_2 = df_stat_2_2.rename(columns={'fare_amount': 'median_fare'})
df_stat_2_2 = df_stat_2_2[['year', 'hour_of_day', 'geoid_p_2', 'geoid_d_2', 'median_fare']]

# get maximum fare
df_stat_2_3 = data.groupby(['year', 'hour_of_day', 'geoid_p_2', 'geoid_d_2'],
                           as_index=False).agg(
    {'fare_amount': max}
).reset_index()

df_stat_2_3 = df_stat_2_3.rename(columns={'fare_amount': 'max_fare'})
df_stat_2_3 = df_stat_2_3[['year', 'hour_of_day', 'geoid_p_2', 'geoid_d_2', 'max_fare']]

# inner join to get the result
df_stat_2 = pd.merge(df_stat_2_1, df_stat_2_2, how='inner',
                     on=['year', 'hour_of_day', 'geoid_p_2', 'geoid_d_2']
                     ).drop_duplicates()

df_stat_2 = pd.merge(df_stat_2, df_stat_2_3, how='inner',
                     on=['year', 'hour_of_day', 'geoid_p_2', 'geoid_d_2']
                     ).drop_duplicates()

###############################################################################
# save result
###############################################################################
df_stat_all.to_csv(df_filename_all, index=False)
df_stat_11.to_csv(df_filename_11, index=False)
df_stat_5.to_csv(df_filename_5, index=False)
df_stat_2.to_csv(df_filename_2, index=False)
