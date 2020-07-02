"""
Filter Training Data further post processing
"""

import pandas as pd

###############################################################################
# load training set
###############################################################################

# Train holdout data
train_file = '../filtered_data/filter_train_holdout.csv'

# load data
train_df = pd.read_csv(train_file)

###############################################################################
# filter outlier
###############################################################################
# Remove observations with useless values base on the test data boundary
mask = train_df['pickup_longitude'].between(-74.3, -72.9)
mask &= train_df['dropoff_longitude'].between(-74.3, -72.9)
mask &= train_df['pickup_latitude'].between(40.5, 41.8)
mask &= train_df['dropoff_latitude'].between(40.5, 41.7)
mask &= train_df['passenger_count'].between(0, 6)  # 1 - 6
mask &= train_df['fare_amount'].between(2, 200)  # 2.5 - 200

# update training set
train_df = train_df[mask]

###############################################################################
# save results
###############################################################################
# save filtered training set
train_df.to_csv(train_file, index=False)
