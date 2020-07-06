"""
Add geoids
"""

import pandas as pd

filter1 = pd.read_csv('../filtered_data/filter_train_holdout.csv')

filter1 = filter1[['pickup_longitude', 'pickup_latitude',
                   'dropoff_longitude', 'dropoff_latitude']]

filter1.to_csv('../filtered_data/filter_train_holdout_for_block.csv')
