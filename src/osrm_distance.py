"""
Calculate osrm distance
"""

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib

###############################################################################
# load osrm and training dataset
###############################################################################
# Adapted from https://www.kaggle.com/maheshdadhich/
# strength-of-visualization-python-visuals-tutorial

train_fr_1 = pd.read_csv('../new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv')

train_fr_2 = pd.read_csv('../new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv')

train_fr = pd.concat([train_fr_1, train_fr_2])
train_fr_new = train_fr[
    ['id', 'total_distance', 'total_travel_time', 'number_of_steps']
]

# a different training set from OSRM (no fare amount)
data = pd.read_csv('../new-york-city-taxi-with-osrm/train.csv')

###############################################################################
# data pre-processing
###############################################################################
# clean data
# remove longitude outliers
data = data[data['pickup_longitude'] >= -80]
data = data[data['pickup_longitude'] <= -70]
data = data[data['dropoff_longitude'] >= -80]
data = data[data['dropoff_longitude'] <= -70]

# remove latitude outliers
data = data[data['pickup_latitude'] >= 35]
data = data[data['pickup_latitude'] <= 45]
data = data[data['dropoff_latitude'] >= 35]
data = data[data['dropoff_latitude'] <= 45]

# remove missing samples
data = data.dropna()

# join two data frames
data = pd.merge(data, train_fr_new, on='id', how='left')

osrm_data = data[
    ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
     'dropoff_latitude', 'total_distance', 'total_travel_time',
     'number_of_steps']
]

# remove missing samples
osrm_data = osrm_data.dropna()

# save result
osrm_data.to_csv('../new-york-city-taxi-with-osrm/osrm_data.csv', index=False)

###############################################################################
# fit KNN for OSRM train data
###############################################################################
# get feature and label
X = osrm_data[[
    'pickup_latitude', 'pickup_longitude',
    'dropoff_latitude', 'dropoff_longitude'
]]
Y_dist = osrm_data['total_distance']
Y_time = osrm_data['total_travel_time']
Y_step = osrm_data['number_of_steps']

# fit the model
# distance
neigh_dist = KNeighborsRegressor(n_neighbors=6)
neigh_dist.fit(X, Y_dist)

# time
neigh_time = KNeighborsRegressor(n_neighbors=7)
neigh_time.fit(X, Y_time)

# step
neigh_step = KNeighborsRegressor(n_neighbors=9)
neigh_step.fit(X, Y_step)

###############################################################################
# save model
###############################################################################
# Save to file in the current working directory    
pkl_dist_filename = '../simple_model/nn_osrm_dist.pkl'

pkl_time_filename = './simple_model/nn_osrm_time.pkl'

pkl_step_filename = '../simple_model/nn_osrm_step.pkl'

joblib.dump(neigh_dist, pkl_dist_filename)
joblib.dump(neigh_time, pkl_time_filename)
joblib.dump(neigh_step, pkl_step_filename)
