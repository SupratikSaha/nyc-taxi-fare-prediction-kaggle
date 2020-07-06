"""
Basic functions for feature engineering
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib

pd.set_option('display.expand_frame_repr', False)


###############################################################################
# feature engineering: datetime
###############################################################################

# convert pickup datetime to informative features
def prepare_time_features(df):
    df['pickup_datetime'] = df['pickup_datetime'].str.replace(" UTC", "")
    df['pickup_datetime'] = pd.to_datetime(
        df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S'
    )
    df['hour_of_day'] = df.pickup_datetime.dt.hour
    df['week'] = df.pickup_datetime.dt.week
    df['month'] = df.pickup_datetime.dt.month
    df['year'] = df.pickup_datetime.dt.year
    df['day_of_year'] = df.pickup_datetime.dt.dayofyear
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['quarter'] = df.pickup_datetime.dt.quarter
    df['day_of_month'] = df.pickup_datetime.dt.day

    # remove unused features
    df = df.drop(['pickup_datetime'], axis=1)

    return df[['hour_of_day', 'week', 'month', 'year',
               'day_of_year', 'weekday', 'quarter',
               'day_of_month']]


###############################################################################
# feature engineering: distance
###############################################################################
# calculate the sphere distance
def sphere_distance(longitude_1, latitude_1, longitude_2, latitude_2):
    longitude_1, latitude_1, longitude_2, latitude_2 = map(
        np.radians, [longitude_1, latitude_1, longitude_2, latitude_2]
    )

    dlongitude = longitude_2 - longitude_1
    dlatitude = latitude_2 - latitude_1

    value = (np.sin(dlatitude / 2.0) ** 2 +
             np.cos(latitude_1) * np.cos(latitude_2) *
             np.sin(dlongitude / 2.0) ** 2
             )

    sphere_distance = 2 * np.arcsin(np.sqrt(value)) * 6367

    return sphere_distance


# calculate the distance based on data frame containing latitude and longitude
def distance(df):
    meas_ang = 0.506  # 29 degrees = 0.506 radians

    # difference
    df['dlongitude'] = abs(
        df['dropoff_longitude'] - df['pickup_longitude']
    ) * 50

    df['dlatitude'] = abs(df['dropoff_latitude'] - df['pickup_latitude']) * 69

    df['Euclidean'] = (df['dlongitude'] ** 2 + df['dlatitude'] ** 2) ** 0.5

    df['delta_manh_long'] = np.abs(
        df['Euclidean'] * np.sin(
            np.arctan(df['dlongitude'] / df['dlatitude']) - meas_ang
        )
    )

    df['delta_manh_lat'] = np.abs(
        df['Euclidean'] * np.cos(
            np.arctan(df['dlongitude'] / df['dlatitude']) - meas_ang
        )
    )

    df['manh_length'] = df['delta_manh_long'] + df['delta_manh_lat']

    df['Euc_error'] = (
            (df['manh_length'] - df['Euclidean']) * 100
            / df['Euclidean']
    )

    # replace nan with 0
    df[['dlongitude', 'dlatitude',
        'Euclidean', 'delta_manh_long',
        'delta_manh_lat', 'manh_length',
        'Euc_error']] = df[['dlongitude', 'dlatitude',
                            'Euclidean', 'delta_manh_long',
                            'delta_manh_lat', 'manh_length',
                            'Euc_error']].fillna(0)

    # get sphere distance
    df['sphere_dist'] = sphere_distance(
        df['pickup_longitude'],
        df['pickup_latitude'],
        df['dropoff_longitude'],
        df['dropoff_latitude']
    )

    return df[['sphere_dist', 'Euclidean', 'manh_length', 'Euc_error']]


###############################################################################
# feature engineering: distance to the airport
###############################################################################
def dist_to_airport(data, lat_airport, lon_airport, airport_name):
    df = data.copy()

    meas_ang = 0.506  # 29 degrees = 0.506 radians

    # drop off
    df['dropoff_dlon'] = np.abs(df['dropoff_longitude'] - lon_airport) * 50
    df['dropoff_dlat'] = np.abs(df['dropoff_latitude'] - lat_airport) * 69

    df['dropoff_Euclidean_' + airport_name] = \
        (df['dropoff_dlon'] ** 2 + df['dropoff_dlat'] ** 2) ** 0.5

    df['dropoff_delta_manh_long'] = np.abs(
        df['dropoff_Euclidean_' + airport_name] *
        np.sin(np.arctan(df['dropoff_dlon'] / df['dropoff_dlat']) -
               meas_ang)
    )

    df['dropoff_delta_manh_lat'] = np.abs(
        df['dropoff_Euclidean_' + airport_name] *
        np.cos(np.arctan(df['dropoff_dlon'] / df['dropoff_dlat']) -
               meas_ang)
    )

    df['dropoff_manh_length_' + airport_name] = \
        df['dropoff_delta_manh_long'] + df['dropoff_delta_manh_lat']

    # pickup
    df['pickup_dlon'] = np.abs(df['pickup_longitude'] - lon_airport) * 50
    df['pickup_dlat'] = np.abs(df['pickup_latitude'] - lat_airport) * 69

    df['pickup_Euclidean_' + airport_name] = (df['pickup_dlon'] ** 2 + df['pickup_dlat'] ** 2
                                              ) ** 0.5

    df['pickup_delta_manh_long'] = np.abs(
        df['pickup_Euclidean_' + airport_name] *
        np.sin(np.arctan(df['pickup_dlon'] / df['pickup_dlat']) - meas_ang)
    )
    df['pickup_delta_manh_lat'] = np.abs(
        df['pickup_Euclidean_' + airport_name] *
        np.cos(np.arctan(df['pickup_dlon'] / df['pickup_dlat']) - meas_ang)
    )
    df['pickup_manh_length_' + airport_name] = \
        df['pickup_delta_manh_long'] + df['pickup_delta_manh_lat']

    # remove unused features
    df = df.drop(['dropoff_dlon', 'dropoff_dlat', 'pickup_dlon', 'pickup_dlat',
                  'dropoff_delta_manh_long', 'dropoff_delta_manh_lat',
                  'pickup_delta_manh_long', 'pickup_delta_manh_lat',
                  'dropoff_Euclidean_' + airport_name,
                  'pickup_Euclidean_' + airport_name], axis=1)

    return df


# identify airport
def airport_feats(df):
    nyc = (-74.0060, 40.7128)
    jfk = (-73.7781, 40.6413)
    ewr = (-74.1745, 40.6895)
    lgr = (-73.8740, 40.7769)

    # calculate the distance to each airport
    df = dist_to_airport(df, nyc[1], nyc[0], 'nyc')
    df = dist_to_airport(df, jfk[1], jfk[0], 'jfk')
    df = dist_to_airport(df, ewr[1], ewr[0], 'ewr')
    df = dist_to_airport(df, lgr[1], lgr[0], 'lgr')

    return df[['pickup_manh_length_nyc', 'dropoff_manh_length_nyc',
               'pickup_manh_length_jfk', 'dropoff_manh_length_jfk',
               'pickup_manh_length_ewr', 'dropoff_manh_length_ewr',
               'pickup_manh_length_lgr', 'dropoff_manh_length_lgr']]


# identify county
def county_feats(df):
    Nassau = (-73.5594, 40.6546)
    Suffolk = (-72.6151, 40.9849)
    Westchester = (-73.7949, 41.1220)
    Rockland = (-73.9830, 41.1489)
    Dutchess = (-73.7478, 41.7784)
    Orange = (-74.3118, 41.3912)
    Putnam = (-73.7949, 41.4351)

    # calculate the distance to each airport
    df = dist_to_airport(df, Nassau[1], Nassau[0], 'nas')
    df = dist_to_airport(df, Suffolk[1], Suffolk[0], 'suf')
    df = dist_to_airport(df, Westchester[1], Westchester[0], 'wes')
    df = dist_to_airport(df, Rockland[1], Rockland[0], 'roc')
    df = dist_to_airport(df, Dutchess[1], Dutchess[0], 'dut')
    df = dist_to_airport(df, Orange[1], Orange[0], 'ora')
    df = dist_to_airport(df, Putnam[1], Putnam[0], 'put')

    return df[['pickup_manh_length_nas', 'dropoff_manh_length_nas',
               'pickup_manh_length_suf', 'dropoff_manh_length_suf',
               'pickup_manh_length_wes', 'dropoff_manh_length_wes',
               'pickup_manh_length_roc', 'dropoff_manh_length_roc',
               'pickup_manh_length_dut', 'dropoff_manh_length_dut',
               'pickup_manh_length_ora', 'dropoff_manh_length_ora',
               'pickup_manh_length_put', 'dropoff_manh_length_put']]


###############################################################################
# feature engineering: OSRM distance
###############################################################################
def predict_osrm_feature(df):
    # get nn model
    # dist
    pkl_dist_filename = '../simple_model/nn_osrm_dist.pkl'
    neigh_dist = joblib.load(pkl_dist_filename)

    # time
    pkl_time_filename = '../simple_model/nn_osrm_time.pkl'
    neigh_time = joblib.load(pkl_time_filename)

    # step
    pkl_step_filename = '../simple_model/nn_osrm_step.pkl'
    neigh_step = joblib.load(pkl_step_filename)

    # predict OSRM distance  
    df['osrm_distance'] = neigh_dist.predict(
        df[[
            'pickup_latitude', 'pickup_longitude',
            'dropoff_latitude', 'dropoff_longitude'
        ]]
    )

    df['osrm_time'] = neigh_time.predict(
        df[[
            'pickup_latitude', 'pickup_longitude',
            'dropoff_latitude', 'dropoff_longitude'
        ]]
    )

    df['osrm_number_of_steps'] = neigh_step.predict(
        df[[
            'pickup_latitude', 'pickup_longitude',
            'dropoff_latitude', 'dropoff_longitude'
        ]]
    )

    return df[['osrm_distance', 'osrm_time', 'osrm_number_of_steps']]


###############################################################################
# feature engineering: fare based on distance and rule
###############################################################################
# calculate the fare based on the following rules
# Initial charge of 2.50
# 0.50 per 1/5 mile as driven or 0.50 per 60 seconds in slow traffic
# 0.50 MTA state surcharge for trips ending in NY
# 0.30 "improvement surcharge"
# 0.50 nighttime surcharge 8pm to 6am
# 1.00 rush hour surcharge 4pm-8pm weekdays
# Tolls for bridges and tunnels
# Airport special rates for JFK and Newark
# Out-of-city rates
# Negotiated flat rates for some trips outside NYC   
def calculate_fare(df):
    # based on manh langth
    # Initial charge of 2.50
    # 0.50 per 1/5 mile as driven
    df['manh_fare'] = 2.5 + np.floor(df['manh_length'] / 0.2) * 0.5

    # 0.50 nighttime surcharge EST 8pm to 6am
    df.loc[((df['hour_of_day'] >= 20) | (df['hour_of_day'] < 6)),
           'manh_fare'] = \
        df.loc[((df['hour_of_day'] >= 20) | (df['hour_of_day'] < 6)),
               'manh_fare'] + 0.5

    # 1.00 rush hour surcharge EST 4pm-8pm weekdays
    df.loc[
        ((df['hour_of_day'] < 20) & (df['hour_of_day'] >= 16) &
         (df['weekday'] <= 4)), 'manh_fare'
    ] = \
        df.loc[
            ((df['hour_of_day'] < 20) & (df['hour_of_day'] >= 16) &
             (df['weekday'] <= 4)), 'manh_fare'
        ] + 1.0

    # based on Euclidean langth
    # Initial charge of 2.50
    # 0.50 per 1/5 mile as driven
    df['euc_fare'] = 2.5 + np.floor(df['Euclidean'] / 0.2) * 0.5

    # 0.50 nighttime surcharge EST 8pm to 6am
    df.loc[((df['hour_of_day'] >= 20) | (df['hour_of_day'] < 6)),
           'euc_fare'] = \
        df.loc[((df['hour_of_day'] >= 20) | (df['hour_of_day'] < 6)),
               'euc_fare'] + 0.5

    # 1.00 rush hour surcharge EST 4pm-8pm weekdays
    df.loc[
        ((df['hour_of_day'] < 20) & (df['hour_of_day'] >= 16) &
         (df['weekday'] <= 4)), 'euc_fare'
    ] = \
        df.loc[
            ((df['hour_of_day'] < 20) & (df['hour_of_day'] >= 16) &
             (df['weekday'] <= 4)), 'euc_fare'
        ] + 1.0

    df['peak_hour'] = np.where((df['hour_of_day'] >= 16) &
                               (df['hour_of_day'] <= 20) &
                               (df['weekday'] >= 0) &
                               (df['weekday'] <= 4), 1, 0)

    # There is a daily 50-cent surcharge from 8pm to 6am.
    df['night_hour'] = np.where(
        (df['hour_of_day'] >= 20) | (df['hour_of_day'] <= 6), 1, 0
    )

    # There is a 50-cent MTA State Surcharge for all trips that end in New York 
    # City or Nassau, Suffolk, Westchester, Rockland, Dutchess, Orange or 
    # Putnam Counties.
    # The following two variables can be merged into one.
    # The following only considers trips that starts in city center and ends 
    # in nearby counties, while the opposite direction could also be considered
    # counties
    df['county_dropoff_1'] = np.where(
        (
                df['pickup_manh_length_nyc'] <= 5 * 0.621371) &
        (
                (df['dropoff_manh_length_nas'] <= 21.3 * 0.621371) |
                (df['dropoff_manh_length_wes'] <= 22.4 * 0.621371)
        ), 1, 0
    )

    df['county_dropoff_2'] = np.where(
        (
                df['pickup_manh_length_nyc'] <= 5 * 0.621371) &
        (
                (df['dropoff_manh_length_suf'] <= 48.7 * 0.621371) |
                (df['dropoff_manh_length_roc'] <= 14.1 * 0.621371) |
                (df['dropoff_manh_length_dut'] <= 28.7 * 0.621371) |
                (df['dropoff_manh_length_ora'] <= 29 * 0.621371) |
                (df['dropoff_manh_length_put'] <= 15.7 * 0.621371)
        ), 1, 0
    )

    # This is a flat fare of $52 plus tolls, the 50-cent MTA State Surcharge, 
    # the 30-cent Improvement Surcharge, 
    # to/from JFK and any location in Manhattan:
    df['to_from_jfk'] = np.where(
        (
                (df['pickup_manh_length_jfk'] <= 2 * 0.621371) &
                (df['dropoff_manh_length_nyc'] <= 5 * 0.621371)
        ) | (
                (df['pickup_manh_length_nyc'] <= 5 * 0.621371) &
                (df['dropoff_manh_length_jfk'] <= 2 * 0.621371)
        ), 1, 0
    )

    # There is a $4.50 rush hour surcharge (4 PM to 8 PM weekdays, excluding 
    # legal holidays). o/from JFK and any location in Manhattan:
    df['jfk_rush_hour'] = np.where((df['to_from_jfk'] == 1) &
                                   (df['hour_of_day'] >= 16) &
                                   (df['hour_of_day'] <= 20), 1, 0)

    # There is a $17.50 Newark Surcharge to Newark Airport:
    df['ewr'] = np.where((df['pickup_manh_length_nyc'] <= 5 * 0.621371) &
                         (df['dropoff_manh_length_ewr'] <= 1 * 0.621371), 1, 0)

    return df[['manh_fare', 'euc_fare', 'peak_hour', 'night_hour',
               'county_dropoff_1', 'county_dropoff_2', 'to_from_jfk',
               'jfk_rush_hour', 'ewr']]


###############################################################################
# feature engineering: knn prediction
############################################################################### 
# KNN regression based on pure location
def nn_reg_on_location(df, pkl_filename):
    # Load from file
    regr = joblib.load(pkl_filename)

    # get the prediction    
    df['nn_fare_pure_location'] = regr.predict(
        df[[
            'pickup_latitude', 'pickup_longitude',
            'dropoff_latitude', 'dropoff_longitude'
        ]]
    )

    return df[['nn_fare_pure_location']]


###############################################################################
# feature engineering: cat-boost prediction (based on pure time)
############################################################################### 
def catboost_on_time(df, pkl_filename):
    # Load from file
    catboost = joblib.load(pkl_filename)

    # get the prediction    
    df['catboost_fare_pure_time'] = catboost.predict(
        df[['hour_of_day', 'weekday', 'month', 'year']]
    )
    return df[['catboost_fare_pure_time']]


def catboost_on_time_loc(df, pkl_filename):
    # Load from file
    catboost = joblib.load(pkl_filename)

    # get the prediction    
    df['catboost_fare_time_loc'] = catboost.predict(
        df[['hour_of_day', 'weekday', 'month', 'year',
            'pickup_latitude', 'pickup_longitude',
            'dropoff_latitude', 'dropoff_longitude']]
    )
    return df[['catboost_fare_time_loc']]


###############################################################################
# feature engineering: geo stat features
############################################################################### 
def year_hour_geo_fare_stat(df, geo_stat_file_all, geo_stat_file_11,
                            geo_stat_file_5, geo_stat_file_2):
    # convert these float to string for matching
    df['geoid_p'] = df['geoid_p'].astype(str)
    df['geoid_d'] = df['geoid_d'].astype(str)

    df['geoid_p_2'] = df['geoid_p'].str[:2]
    df['geoid_d_2'] = df['geoid_d'].str[:2]

    df['geoid_p_5'] = df['geoid_p'].str[:5]
    df['geoid_d_5'] = df['geoid_d'].str[:5]

    df['geoid_p_11'] = df['geoid_p'].str[:11]
    df['geoid_d_11'] = df['geoid_d'].str[:11]

    # load stat feature
    df_stat_all = pd.read_csv(geo_stat_file_all)
    df_stat_11 = pd.read_csv(geo_stat_file_11)
    df_stat_5 = pd.read_csv(geo_stat_file_5)
    df_stat_2 = pd.read_csv(geo_stat_file_2)

    # convert these float to string for matching
    df_stat_all['geoid_p'] = df_stat_all['geoid_p'].astype(str)
    df_stat_all['geoid_d'] = df_stat_all['geoid_d'].astype(str)

    df_stat_11['geoid_p_11'] = df_stat_11['geoid_p_11'].astype(str)
    df_stat_11['geoid_d_11'] = df_stat_11['geoid_d_11'].astype(str)

    df_stat_5['geoid_p_5'] = df_stat_5['geoid_p_5'].astype(str)
    df_stat_5['geoid_d_5'] = df_stat_5['geoid_d_5'].astype(str)

    df_stat_2['geoid_p_2'] = df_stat_2['geoid_p_2'].astype(str)
    df_stat_2['geoid_d_2'] = df_stat_2['geoid_d_2'].astype(str)

    # rename
    df_stat_all = df_stat_all.rename(columns={
        'mean_fare': 'mean_fare_all',
        'median_fare': 'median_fare_all',
        'max_fare': 'max_fare_all'
    })

    df_stat_11 = df_stat_11.rename(columns={
        'mean_fare': 'mean_fare_11',
        'median_fare': 'median_fare_11',
        'max_fare': 'max_fare_11'
    })

    df_stat_5 = df_stat_5.rename(columns={
        'mean_fare': 'mean_fare_5',
        'median_fare': 'median_fare_5',
        'max_fare': 'max_fare_5'
    })

    df_stat_2 = df_stat_2.rename(columns={
        'mean_fare': 'mean_fare_2',
        'median_fare': 'median_fare_2',
        'max_fare': 'max_fare_2'
    })

    # merge to get the stat feature
    df = pd.merge(df, df_stat_all, how='left',
                  left_on=['geoid_p', 'geoid_d'],
                  right_on=['geoid_p', 'geoid_d']
                  ).drop_duplicates()

    df = pd.merge(df, df_stat_11, how='left',
                  left_on=['geoid_p_11', 'geoid_d_11'],
                  right_on=['geoid_p_11', 'geoid_d_11']
                  ).drop_duplicates()

    df = pd.merge(df, df_stat_5, how='left',
                  left_on=['year', 'hour_of_day', 'geoid_p_5', 'geoid_d_5'],
                  right_on=['year', 'hour_of_day', 'geoid_p_5', 'geoid_d_5']
                  ).drop_duplicates()

    df = pd.merge(df, df_stat_2, how='left',
                  left_on=['year', 'hour_of_day', 'geoid_p_2', 'geoid_d_2'],
                  right_on=['year', 'hour_of_day', 'geoid_p_2', 'geoid_d_2']
                  ).drop_duplicates()

    # replace null by 0s
    df = df.fillna(0)

    # convert string back to float
    df[['geoid_p', 'geoid_d', 'geoid_p_2', 'geoid_d_2',
        'geoid_p_5', 'geoid_d_5', 'geoid_p_11', 'geoid_d_11']] = df[[
        'geoid_p', 'geoid_d', 'geoid_p_2', 'geoid_d_2',
        'geoid_p_5', 'geoid_d_5', 'geoid_p_11', 'geoid_d_11'
    ]].astype(int)

    return df[['geoid_p', 'geoid_d', 'geoid_p_2', 'geoid_d_2',
               'geoid_p_5', 'geoid_d_5', 'geoid_p_11', 'geoid_d_11',
               'mean_fare_all', 'median_fare_all', 'max_fare_all',
               'mean_fare_11', 'median_fare_11', 'max_fare_11',
               'mean_fare_5', 'median_fare_5', 'max_fare_5',
               'mean_fare_2', 'median_fare_2', 'max_fare_2']]


###############################################################################
# feature engineering: stat features
###############################################################################
# historical stat based on year/weekday/location
def year_weekday_location_fare_stat(df, stat_file):
    # load stat feature
    df_stat = pd.read_csv(stat_file)

    # convert these float to string for matching
    df_stat['plong'] = df_stat['plong'].astype(str)
    df_stat['plat'] = df_stat['plat'].astype(str)
    df_stat['dlong'] = df_stat['dlong'].astype(str)
    df_stat['dlat'] = df_stat['dlat'].astype(str)

    # convert the location features
    # get the first 4 decimal of longtitude and latitude
    df['plong'] = df['pickup_longitude'].round(2)
    df['plat'] = df['pickup_latitude'].round(2)
    df['dlong'] = df['dropoff_longitude'].round(2)
    df['dlat'] = df['dropoff_latitude'].round(2)

    # convert these float to string for matching
    df['plong'] = df['plong'].astype(str)
    df['plat'] = df['plat'].astype(str)
    df['dlong'] = df['dlong'].astype(str)
    df['dlat'] = df['dlat'].astype(str)

    # merge to get the stat feature
    df = pd.merge(df, df_stat, how='left',
                  left_on=['year', 'weekday', 'plong', 'plat', 'dlong', 'dlat'],
                  right_on=['year', 'weekday', 'plong', 'plat', 'dlong', 'dlat']
                  ).drop_duplicates()

    # replace null by 0s
    df = df.fillna(0)

    # remove unused features
    df = df.drop(['plong', 'plat', 'dlong', 'dlat'], axis=1)

    return df[['mean_fare', 'median_fare', 'max_fare']]


###############################################################################
# feature engineering: historical features
###############################################################################
# add fare distribution based on longitude and latitude
def long_lat_stat(df):
    # load prepared features
    long_lat_stat = pd.read_csv('../new_feature/long_lat_stat.csv')

    # get the first 1 decimal of longtitude and latitude
    df['plong'] = df['pickup_longitude'].round(1)
    df['plat'] = df['pickup_latitude'].round(1)
    df['dlong'] = df['dropoff_longitude'].round(1)
    df['dlat'] = df['dropoff_latitude'].round(1)

    # convert these float to string for matching
    df['plong'] = df['plong'].astype(str)
    df['plat'] = df['plat'].astype(str)
    df['dlong'] = df['dlong'].astype(str)
    df['dlat'] = df['dlat'].astype(str)
    long_lat_stat['plong'] = long_lat_stat['plong'].astype(str)
    long_lat_stat['plat'] = long_lat_stat['plat'].astype(str)
    long_lat_stat['dlong'] = long_lat_stat['dlong'].astype(str)
    long_lat_stat['dlat'] = long_lat_stat['dlat'].astype(str)

    # left join two table to get stats
    df = pd.merge(
        df, long_lat_stat, how='left',
        left_on=['plong', 'plat', 'dlong', 'dlat'],
        right_on=['plong', 'plat', 'dlong', 'dlat']
    ).drop_duplicates()

    # rename column
    df = df.rename(
        columns={
            'counts': 'counts_ll',
            'max_fare': 'max_fare_ll',
            'min_fare': 'min_fare_ll',
            'avg_fare': 'avg_fare_ll',
            'med_fare': 'med_fare_ll'
        }
    )

    # remove unused columns
    df = df.drop(['plong', 'plat', 'dlong', 'dlat'], axis=1)
    df = df.drop(['counts_ll', 'min_fare_ll'], axis=1)
    return df


# add fare distribution based on month, longitude and latitude
def month_long_lat_stat(df):
    # load prepared features
    month_long_lat_stat = pd.read_csv('../new_feature/month_long_lat_stat.csv')

    # get the first 1 decimal of longtitude and latitude
    df['plong'] = df['pickup_longitude'].round(1)
    df['plat'] = df['pickup_latitude'].round(1)
    df['dlong'] = df['dropoff_longitude'].round(1)
    df['dlat'] = df['dropoff_latitude'].round(1)

    # convert these float to string for matching
    df['plong'] = df['plong'].astype(str)
    df['plat'] = df['plat'].astype(str)
    df['dlong'] = df['dlong'].astype(str)
    df['dlat'] = df['dlat'].astype(str)
    month_long_lat_stat['plong'] = month_long_lat_stat['plong'].astype(str)
    month_long_lat_stat['plat'] = month_long_lat_stat['plat'].astype(str)
    month_long_lat_stat['dlong'] = month_long_lat_stat['dlong'].astype(str)
    month_long_lat_stat['dlat'] = month_long_lat_stat['dlat'].astype(str)

    # left join two table to get stats
    df = pd.merge(
        df, month_long_lat_stat, how='left',
        left_on=['month', 'plong', 'plat', 'dlong', 'dlat'],
        right_on=['month', 'plong', 'plat', 'dlong', 'dlat']
    ).drop_duplicates()

    # rename column
    df = df.rename(
        columns={
            'counts': 'counts_mll',
            'max_fare': 'max_fare_mll',
            'min_fare': 'min_fare_mll',
            'avg_fare': 'avg_fare_mll',
            'med_fare': 'med_fare_mll'
        }
    )

    # remove unused columns
    df = df.drop(['plong', 'plat', 'dlong', 'dlat'], axis=1)
    df = df.drop(['counts_mll', 'min_fare_mll'], axis=1)
    return df


# add fare distribution based on weekday, longitude and latitude
def weekday_long_lat_stat(df):
    # load prepared features
    weekday_long_lat_stat = pd.read_csv('../new_feature/weekday_long_lat_stat.csv')

    # to match the weekday between big-query and python
    # for big-query: Sunday -> Saturday: 1 -> 7
    # for python: Sunday -> Saturday: 6, 0 -> 5
    weekday_long_lat_stat['weekday'] = (weekday_long_lat_stat['weekday'] - 2)
    weekday_long_lat_stat.loc[
        weekday_long_lat_stat['weekday'] == -1, 'weekday'
    ] = 6

    # get the first 1 decimal of longitude and latitude
    df['plong'] = df['pickup_longitude'].round(1)
    df['plat'] = df['pickup_latitude'].round(1)
    df['dlong'] = df['dropoff_longitude'].round(1)
    df['dlat'] = df['dropoff_latitude'].round(1)

    # convert these float to string for matching
    df['plong'] = df['plong'].astype(str)
    df['plat'] = df['plat'].astype(str)
    df['dlong'] = df['dlong'].astype(str)
    df['dlat'] = df['dlat'].astype(str)
    weekday_long_lat_stat['plong'] = weekday_long_lat_stat['plong'].astype(str)
    weekday_long_lat_stat['plat'] = weekday_long_lat_stat['plat'].astype(str)
    weekday_long_lat_stat['dlong'] = weekday_long_lat_stat['dlong'].astype(str)
    weekday_long_lat_stat['dlat'] = weekday_long_lat_stat['dlat'].astype(str)

    # left join two table to get stats
    df = pd.merge(
        df, weekday_long_lat_stat, how='left',
        left_on=['weekday', 'plong', 'plat', 'dlong', 'dlat'],
        right_on=['weekday', 'plong', 'plat', 'dlong', 'dlat']
    ).drop_duplicates()

    # rename column
    df = df.rename(
        columns={
            'counts': 'counts_Wdll',
            'max_fare': 'max_fare_Wdll',
            'min_fare': 'min_fare_Wdll',
            'avg_fare': 'avg_fare_Wdll',
            'med_fare': 'med_fare_Wdll'
        }
    )

    # remove unused columns
    df = df.drop(['plong', 'plat', 'dlong', 'dlat'], axis=1)
    df = df.drop(['counts_Wdll', 'min_fare_Wdll'], axis=1)
    return df


# add fare distribution based on year, longitude and latitude
def year_long_lat_stat(df):
    # load prepared features
    year_long_lat_stat = pd.read_csv('../new_feature/year_long_lat_stat.csv')

    # get the first 1 decimal of longtitude and latitude
    df['plong'] = df['pickup_longitude'].round(1)
    df['plat'] = df['pickup_latitude'].round(1)
    df['dlong'] = df['dropoff_longitude'].round(1)
    df['dlat'] = df['dropoff_latitude'].round(1)

    # convert these float to string for matching
    df['plong'] = df['plong'].astype(str)
    df['plat'] = df['plat'].astype(str)
    df['dlong'] = df['dlong'].astype(str)
    df['dlat'] = df['dlat'].astype(str)
    year_long_lat_stat['plong'] = year_long_lat_stat['plong'].astype(str)
    year_long_lat_stat['plat'] = year_long_lat_stat['plat'].astype(str)
    year_long_lat_stat['dlong'] = year_long_lat_stat['dlong'].astype(str)
    year_long_lat_stat['dlat'] = year_long_lat_stat['dlat'].astype(str)

    # left join two table to get stats
    df = pd.merge(
        df, year_long_lat_stat, how='left',
        left_on=['year', 'plong', 'plat', 'dlong', 'dlat'],
        right_on=['year', 'plong', 'plat', 'dlong', 'dlat']
    ).drop_duplicates()

    # rename column
    df = df.rename(
        columns={
            'counts': 'counts_yll',
            'max_fare': 'max_fare_yll',
            'min_fare': 'min_fare_yll',
            'avg_fare': 'avg_fare_yll',
            'med_fare': 'med_fare_yll'
        }
    )

    # remove unused columns
    df = df.drop(['plong', 'plat', 'dlong', 'dlat'], axis=1)
    df = df.drop(['counts_yll', 'min_fare_yll'], axis=1)
    return df


# add fare distribution based on year, month, day, and hour
def year_month_day_hour_stat(df):
    # load prepared features
    year_month_day_hour_stat_2009 = pd.read_csv('../new_feature/year_month_day_hour_2009.csv')

    year_month_day_hour_stat_2010 = pd.read_csv('../new_feature/year_month_day_hour_2010.csv')

    year_month_day_hour_stat_2011 = pd.read_csv('../new_feature/year_month_day_hour_2011.csv')

    year_month_day_hour_stat_2012 = pd.read_csv('../new_feature/year_month_day_hour_2012.csv')

    year_month_day_hour_stat_2013 = pd.read_csv('../new_feature/year_month_day_hour_2013.csv')

    year_month_day_hour_stat_2014 = pd.read_csv('../new_feature/year_month_day_hour_2014.csv')

    year_month_day_hour_stat_2015 = pd.read_csv('../new_feature/year_month_day_hour_2015.csv')

    year_month_day_hour_stat = pd.concat(
        [
            year_month_day_hour_stat_2009,
            year_month_day_hour_stat_2010,
            year_month_day_hour_stat_2011,
            year_month_day_hour_stat_2012,
            year_month_day_hour_stat_2013,
            year_month_day_hour_stat_2014,
            year_month_day_hour_stat_2015
        ],
        axis=0
    )

    # left join two table to get stats
    df = pd.merge(
        df, year_month_day_hour_stat, how='left',
        left_on=['year', 'month', 'day_of_month', 'hour_of_day'],
        right_on=['year', 'month', 'day', 'hour']
    ).drop_duplicates()

    # rename column
    df = df.rename(
        columns={
            'counts': 'counts_ymdh',
            'max_fare': 'max_fare_ymdh',
            'min_fare': 'min_fare_ymdh',
            'avg_fare': 'avg_fare_ymdh',
            'med_fare': 'med_fare_ymdh'
        }
    )

    # remove unused columns
    df = df.drop(['day', 'hour'], axis=1)
    df = df.drop(['counts_ymdh', 'min_fare_ymdh'], axis=1)

    return df


# add fare distribution based on year, month, day
def year_month_day_stat(df):
    # load prepared features
    year_month_day_stat = pd.read_csv('../new_feature/year_month_day_stat.csv')

    # left join two table to get stats
    df = pd.merge(
        df, year_month_day_stat, how='left',
        left_on=['year', 'month', 'day_of_month'],
        right_on=['year', 'month', 'day']
    ).drop_duplicates()

    # rename column
    df = df.rename(
        columns={
            'counts': 'counts_ymd',
            'max_fare': 'max_fare_ymd',
            'min_fare': 'min_fare_ymd',
            'avg_fare': 'avg_fare_ymd',
            'med_fare': 'med_fare_ymd'
        }
    )

    # remove unused columns
    df = df.drop(['day'], axis=1)
    df = df.drop(['counts_ymd', 'min_fare_ymd'], axis=1)

    return df


# add fare distribution based on year, month, hour
def year_month_hour_stat(df):
    # load prepared features
    year_month_hour_stat = pd.read_csv('../new_feature/year_month_hour_stat.csv')

    # left join two table to get stats
    df = pd.merge(
        df, year_month_hour_stat, how='left',
        left_on=['year', 'month', 'hour_of_day'],
        right_on=['year', 'month', 'hour']
    ).drop_duplicates()

    # rename column
    df = df.rename(
        columns={
            'counts': 'counts_ymh',
            'max_fare': 'max_fare_ymh',
            'min_fare': 'min_fare_ymh',
            'avg_fare': 'avg_fare_ymh',
            'med_fare': 'med_fare_ymh'
        }
    )

    # remove unused columns
    df = df.drop(['hour'], axis=1)
    df = df.drop(['counts_ymh', 'min_fare_ymh'], axis=1)

    return df


# add fare distribution based on year, month, weekday
def year_month_weekday_stat(df):
    # load prepared features
    year_month_weekday_stat = pd.read_csv('../new_feature/year_month_weekday_stat.csv')

    # to match the weekday between bigquery and python
    # for bigquery: Sunday -> Saturday: 1 -> 7
    # for python: Sunday -> Saturday: 6, 0 -> 5
    year_month_weekday_stat['weekday'] = (
            year_month_weekday_stat['weekday'] - 2
    )

    year_month_weekday_stat.loc[
        year_month_weekday_stat['weekday'] == -1, 'weekday'
    ] = 6

    # left join two table to get stats
    df = pd.merge(
        df, year_month_weekday_stat, how='left',
        left_on=['year', 'month', 'weekday'],
        right_on=['year', 'month', 'weekday']
    ).drop_duplicates()

    # rename column
    df = df.rename(
        columns={
            'counts': 'counts_ymWd',
            'max_fare': 'max_fare_ymWd',
            'min_fare': 'min_fare_ymWd',
            'avg_fare': 'avg_fare_ymWd',
            'med_fare': 'med_fare_ymWd'
        }
    )

    # remove unused columns
    df = df.drop(['counts_ymWd', 'min_fare_ymWd'], axis=1)

    return df


# add fare distribution based on year, week
def year_week_stat(df):
    # load prepared features
    year_week_stat = pd.read_csv('../new_feature/year_week_stat.csv')

    # left join two table to get stats
    df = pd.merge(
        df, year_week_stat, how='left',
        left_on=['year', 'week'],
        right_on=['year', 'week']
    ).drop_duplicates()

    # rename column
    df = df.rename(
        columns={
            'counts': 'counts_yw',
            'max_fare': 'max_fare_yw',
            'min_fare': 'min_fare_yw',
            'avg_fare': 'avg_fare_yw',
            'med_fare': 'med_fare_yw'
        }
    )

    # remove unused columns
    df = df.drop(['counts_yw', 'min_fare_yw'], axis=1)

    return df


###############################################################################
# feature engineering: distance to the airport
###############################################################################
# simple regression based on distance
def reg_on_distance(df):
    # Load from file
    pkl_filename = "'../simple_regression.pkl"

    with open(pkl_filename, 'rb') as file:
        regr = pickle.load(file)

    # get the prediction    
    df['lr_fit_fare'] = regr.predict(df[['manh_length']])

    return df
