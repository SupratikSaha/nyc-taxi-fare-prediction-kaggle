"""
KNN on locations
"""

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib

###############################################################################
# set up parameter
###############################################################################
# Case 1: only train based on 2,3,4,5 subset
trainset1 = '../all/filtered_data/filter_train2.csv'
trainset2 = '../all/filtered_data/filter_train3.csv'
trainset3 = '../all/filtered_data/filter_train4.csv'
trainset4 = '../all/filtered_data/filter_train5.csv'

pkl_filename = '../simple_model/train1/nn_pure_loc_1.pkl'

###############################################################################
# load dataset
###############################################################################
# load location and time only
fields = ['pickup_latitude', 'pickup_longitude',
          'dropoff_latitude', 'dropoff_longitude',
          'fare_amount']

# read the data
data1 = pd.read_csv(trainset1, usecols=fields)
data2 = pd.read_csv(trainset2, usecols=fields)
data3 = pd.read_csv(trainset3, usecols=fields)
data4 = pd.read_csv(trainset4, usecols=fields)

# combine data
data = pd.concat([data1, data2, data3, data4], axis=0)

###############################################################################
# KNN regression
###############################################################################
# get feature and label
X = data[[
    'pickup_latitude', 'pickup_longitude',
    'dropoff_latitude', 'dropoff_longitude'
]]
Y = data['fare_amount']

# fit the model
neigh = KNeighborsRegressor(n_neighbors=5)
neigh.fit(X, Y)

###############################################################################
# save result
###############################################################################
# Save to file in the current working directory    
joblib.dump(neigh, pkl_filename)
