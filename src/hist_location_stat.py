"""
Historic Location stats
"""

import pandas as pd
import numpy as np

###############################################################################
# set up parameter
###############################################################################
# Case 1: only train based on 2,3,4,5 subset
trainset1 = '../filtered_data/filter_train2.csv'
trainset2 = '../filtered_data/filter_train3.csv'
trainset3 = '../filtered_data/filter_train4.csv'
trainset4 = '../filtered_data/filter_train5.csv'

df_filename = '../stat_feature/y_wd_loc_1.csv'

###############################################################################
# load dataset
###############################################################################
# load location and time only
fields = ['pickup_datetime', 'fare_amount',
          'pickup_latitude', 'pickup_longitude',
          'dropoff_latitude', 'dropoff_longitude'
          ]

# read the data
data1 = pd.read_csv(trainset1, usecols=fields)
data2 = pd.read_csv(trainset2, usecols=fields)
data3 = pd.read_csv(trainset3, usecols=fields)
data4 = pd.read_csv(trainset4, usecols=fields)

# combine data
data = pd.concat([data1, data2, data3, data4], axis=0)

# convert pickup datetime to informative features
data['pickup_datetime'] = data['pickup_datetime'].str.replace(" UTC", "")
data['pickup_datetime'] = pd.to_datetime(
    data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S'
)
data["weekday"] = data.pickup_datetime.dt.weekday
data["year"] = data.pickup_datetime.dt.year

# remove unused features
data = data.drop(['pickup_datetime'], axis=1)

###############################################################################
# data preprocessing
###############################################################################
# get the first 4 decimal of longtitude and latitude
data['plong'] = data['pickup_longitude'].round(2)
data['plat'] = data['pickup_latitude'].round(2)
data['dlong'] = data['dropoff_longitude'].round(2)
data['dlat'] = data['dropoff_latitude'].round(2)

# convert these float to string for matching
data['plong'] = data['plong'].astype(str)
data['plat'] = data['plat'].astype(str)
data['dlong'] = data['dlong'].astype(str)
data['dlat'] = data['dlat'].astype(str)

# removed unused features
data = data.drop([
    'pickup_latitude', 'pickup_longitude',
    'dropoff_latitude', 'dropoff_longitude'
], axis=1)

###############################################################################
# calculate the histrical stat based on these locations
###############################################################################
# get mean fare
df_stat1 = data.groupby(
    ['year', 'weekday', 'plong', 'plat', 'dlong', 'dlat'],
    as_index=False).agg(
    {'fare_amount': np.mean}
).reset_index()

df_stat1 = df_stat1.rename(columns={'fare_amount': 'mean_fare'})
df_stat1 = df_stat1[[
    'year', 'weekday', 'plong', 'plat', 'dlong', 'dlat', 'mean_fare'
]]

# get median fare
df_stat2 = data.groupby(
    ['year', 'weekday', 'plong', 'plat', 'dlong', 'dlat'],
    as_index=False).agg(
    {'fare_amount': np.median}
).reset_index()

df_stat2 = df_stat2.rename(columns={'fare_amount': 'median_fare'})
df_stat2 = df_stat2[[
    'year', 'weekday', 'plong', 'plat', 'dlong', 'dlat', 'median_fare'
]]

# get maximum fare
df_stat3 = data.groupby(
    ['year', 'weekday', 'plong', 'plat', 'dlong', 'dlat'],
    as_index=False).agg(
    {'fare_amount': max}
).reset_index()

df_stat3 = df_stat3.rename(columns={'fare_amount': 'max_fare'})
df_stat3 = df_stat3[[
    'year', 'weekday', 'plong', 'plat', 'dlong', 'dlat', 'max_fare'
]]

# inner join to get the result
df_stat = pd.merge(df_stat1, df_stat2, how='inner',
                   left_on=['year', 'weekday', 'plong', 'plat', 'dlong', 'dlat'],
                   right_on=['year', 'weekday', 'plong', 'plat', 'dlong', 'dlat']
                   ).drop_duplicates()

df_stat = pd.merge(df_stat, df_stat3, how='inner',
                   left_on=['year', 'weekday', 'plong', 'plat', 'dlong', 'dlat'],
                   right_on=['year', 'weekday', 'plong', 'plat', 'dlong', 'dlat']
                   ).drop_duplicates()

###############################################################################
# save result
###############################################################################
df_stat.to_csv(df_filename, index=False)
