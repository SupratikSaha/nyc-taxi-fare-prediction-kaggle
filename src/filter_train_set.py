"""
Filter Training data set
"""

import pandas as pd
import numpy as np

###############################################################################
## load training set
###############################################################################
data = pd.read_csv('../data/train.csv')

###############################################################################
## filter outlier
###############################################################################
# remove the fare smaller than $2.5
data = data[data['fare_amount'] >= 2.5] 

# remove the fare larger than $500
data = data[data['fare_amount'] <= 500] 

# remove longitudes outliers
data = data[data['pickup_longitude'] >= -80]
data = data[data['pickup_longitude'] <= -70]
data = data[data['dropoff_longitude'] >= -80]
data = data[data['dropoff_longitude'] <= -70]

# remove latittudes outliers
data = data[data['pickup_latitude'] >= 35]
data = data[data['pickup_latitude'] <= 45] 
data = data[data['dropoff_latitude'] >= 35]
data = data[data['dropoff_latitude'] <= 45]

# remove 0 or larger than 10 passenger
data = data[data['passenger_count'] > 0] 
data = data[data['passenger_count'] < 10]

# remove missing samples
data = data.dropna() # (54077798, 8)

# split key into datetime and ID
data['ID'] = data['key'].str.split('.').str[-1].astype(np.int64)

# remove unused feature
data = data.drop(['key'], axis = 1)

# save filtered training set
data.to_csv('../filtered_data/filter_train.csv',index = False)