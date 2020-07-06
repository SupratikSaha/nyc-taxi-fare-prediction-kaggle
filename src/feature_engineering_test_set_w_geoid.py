"""
Adding geoid data to tests set
"""

from basic_function import prepare_time_features, distance, airport_feats
from basic_function import calculate_fare, nn_reg_on_location
from basic_function import predict_osrm_feature, catboost_on_time
from basic_function import catboost_on_time_loc, county_feats
from basic_function import year_hour_geo_fare_stat
import pandas as pd
import numpy as np
import sys

sys.path.append('../taxi_fare_prediction/src')

###############################################################################
# Case 2: from the way used in training set 2
df_test = pd.read_csv('../all/test_w_geoid.csv')

# file for saving statistic features
stat_file = '../stat_feature/y_wd_loc_2.csv'

# file for saving processed dataset
df_test_file = '../all/processed_filtered_data/processed_test2_geoid.csv'

df_test_without_key_file = '../all/processed_filtered_data/processed_test2_without_key_geoid.csv'

# simple models
pkl_nn_pure_location = '../simple_model/train2/nn_pure_loc_2.pkl'

pkl_catboost_pure_location = '../simple_model/train2/catboost_pure_time_2.pkl'

pkl_catboost_time_loc = '../simple_model/train2/catboost_loc_time_2.pkl'

geo_stat_file_2 = '../stat_feature/y_h_geo_first2_1.csv'
geo_stat_file_5 = '../stat_feature/y_h_geo_first5_1.csv'
geo_stat_file_11 = '../stat_feature/y_h_geo_first11_1.csv'
geo_stat_file_all = '../stat_feature/y_h_geo_all_1_basedon_2.csv'

###############################################################################
# feature engineering
###############################################################################
# convert the pickup datetime
df_test[['hour_of_day', 'week', 'month', 'year',
         'day_of_year', 'weekday', 'quarter',
         'day_of_month']] = prepare_time_features(
    df_test[['pickup_datetime']].copy()
)

# get geo stat features
df_test[['geoid_p', 'geoid_d', 'geoid_p_2', 'geoid_d_2',
         'geoid_p_5', 'geoid_d_5', 'geoid_p_11', 'geoid_d_11',
         'mean_fare_all', 'median_fare_all', 'max_fare_all',
         'mean_fare_11', 'median_fare_11', 'max_fare_11',
         'mean_fare_5', 'median_fare_5', 'max_fare_5',
         'mean_fare_2', 'median_fare_2', 'max_fare_2']] = \
    year_hour_geo_fare_stat(
        df_test[['geoid_p', 'geoid_d', 'year', 'hour_of_day', 'key']].copy(),
        geo_stat_file_all, geo_stat_file_11,
        geo_stat_file_5, geo_stat_file_2
    )

# drop unused feature
df_test = df_test.drop(['pickup_datetime'], axis=1)

# calculate the fare by pure location-based KNN prediction 
df_test[['nn_fare_pure_location']] = nn_reg_on_location(
    df_test[[
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude'
    ]].copy(),
    pkl_nn_pure_location
)

# calculate the fare by OSRM data-based KNN prediction 
df_test[['osrm_distance', 'osrm_time', 'osrm_number_of_steps']] = \
    predict_osrm_feature(df_test[[
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude'
    ]].copy())

# calculate the fare by pure time-based Catboost prediction 
df_test[['catboost_fare_pure_time']] = \
    catboost_on_time(df_test[['hour_of_day', 'weekday', 'month', 'year']].copy(),
                     pkl_catboost_pure_location)

# calculate the fare by time & location-based Catboost prediction 
df_test[['catboost_fare_time_loc']] = \
    catboost_on_time_loc(df_test[[
        'hour_of_day', 'weekday', 'month', 'year',
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude']].copy(),
                         pkl_catboost_time_loc)

# calculate the distance based on dropoff and pickup locations
df_test[['sphere_dist', 'Euclidean', 'manh_length', 'Euc_error']] = distance(
    df_test[[
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude'
    ]].copy()
)

# check if the distance is nearly 0
df_test['no_loc_change'] = np.where((df_test['manh_length'] < 0.01), 1, 0)

# calculate the distance to each airport and downtown
df_test[['pickup_manh_length_nyc', 'dropoff_manh_length_nyc',
         'pickup_manh_length_jfk', 'dropoff_manh_length_jfk',
         'pickup_manh_length_ewr', 'dropoff_manh_length_ewr',
         'pickup_manh_length_lgr', 'dropoff_manh_length_lgr']] = \
    airport_feats(df_test[[
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude'
    ]].copy())

# calculate the distance to seven counties
df_test[['pickup_manh_length_nas', 'dropoff_manh_length_nas',
         'pickup_manh_length_suf', 'dropoff_manh_length_suf',
         'pickup_manh_length_wes', 'dropoff_manh_length_wes',
         'pickup_manh_length_roc', 'dropoff_manh_length_roc',
         'pickup_manh_length_dut', 'dropoff_manh_length_dut',
         'pickup_manh_length_ora', 'dropoff_manh_length_ora',
         'pickup_manh_length_put', 'dropoff_manh_length_put']] = \
    county_feats(df_test[[
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude'
    ]].copy())

# calculate the fare
df_test[['manh_fare', 'euc_fare', 'peak_hour', 'night_hour',
         'county_dropoff_1', 'county_dropoff_2', 'to_from_jfk',
         'jfk_rush_hour', 'ewr']] = \
    calculate_fare(
        df_test[['manh_length', 'hour_of_day', 'weekday', 'Euclidean',
                 'pickup_manh_length_nyc', 'dropoff_manh_length_nas',
                 'dropoff_manh_length_wes', 'dropoff_manh_length_suf',
                 'dropoff_manh_length_roc', 'dropoff_manh_length_dut',
                 'dropoff_manh_length_ora', 'dropoff_manh_length_put',
                 'pickup_manh_length_jfk', 'dropoff_manh_length_nyc',
                 'dropoff_manh_length_jfk', 'dropoff_manh_length_ewr']].copy()
    )

###############################################################################
# save result
###############################################################################
df_test.to_csv(df_test_file, index=False)

# drop key
df_test = df_test.drop(['key'], axis=1)
df_test.to_csv(df_test_without_key_file, index=False)
