"""
Map Location to geoid data
"""

import pandas as pd

###############################################################################
# load geoIDs
###############################################################################
df_train1_geoid = pd.read_csv('../filtered_data/filter_train1_geoid.csv')

df_train2_geoid = pd.read_csv('../filtered_data/filter_train2_geoid.csv')

###############################################################################
# get location and geoid only
###############################################################################
df_loc_geoid1 = df_train1_geoid[[
    'pickup_longitude', 'pickup_latitude', 'GEOID_p'
]]

df_loc_geoid2 = df_train1_geoid[[
    'dropoff_longitude', 'dropoff_latitude', 'GEOID_d'
]]

df_loc_geoid3 = df_train2_geoid[[
    'pickup_longitude', 'pickup_latitude', 'GEOID_p'
]]

df_loc_geoid4 = df_train2_geoid[[
    'dropoff_longitude', 'dropoff_latitude', 'GEOID_d'
]]

###############################################################################
# rename the column, combine the dataframe, and drop duplicates
###############################################################################
# rename
df_loc_geoid1 = df_loc_geoid1.rename(
    columns={
        'pickup_longitude': 'longitude',
        'pickup_latitude': 'latitude',
        'GEOID_p': 'geoid'
    }
)

df_loc_geoid2 = df_loc_geoid2.rename(
    columns={
        'dropoff_longitude': 'longitude',
        'dropoff_latitude': 'latitude',
        'GEOID_d': 'geoid'
    }
)

df_loc_geoid3 = df_loc_geoid3.rename(
    columns={
        'pickup_longitude': 'longitude',
        'pickup_latitude': 'latitude',
        'GEOID_p': 'geoid'
    }
)

df_loc_geoid4 = df_loc_geoid4.rename(
    columns={
        'dropoff_longitude': 'longitude',
        'dropoff_latitude': 'latitude',
        'GEOID_d': 'geoid'
    }
)

# change type to strings
# convert location to string
df_loc_geoid1[['longitude', 'latitude']] = \
    df_loc_geoid1[['longitude', 'latitude']].astype(str)

df_loc_geoid2[['longitude', 'latitude']] = \
    df_loc_geoid2[['longitude', 'latitude']].astype(str)

df_loc_geoid3[['longitude', 'latitude']] = \
    df_loc_geoid3[['longitude', 'latitude']].astype(str)

df_loc_geoid4[['longitude', 'latitude']] = \
    df_loc_geoid4[['longitude', 'latitude']].astype(str)

###############################################################################
# combine the dataframe
###############################################################################
df_loc_geoid1 = df_loc_geoid1.drop_duplicates()
df_loc_geoid2 = df_loc_geoid2.drop_duplicates()
df_loc_geoid3 = df_loc_geoid3.drop_duplicates()
df_loc_geoid4 = df_loc_geoid4.drop_duplicates()

df_loc_geoid = pd.concat(
    [df_loc_geoid1, df_loc_geoid2, df_loc_geoid3, df_loc_geoid4],
    axis=0
).drop_duplicates()

###############################################################################
# save the result
###############################################################################
df_loc_geoid.to_csv('../all/loc_geoid_train12.csv', index=False)
