"""
Split Training Data
"""
from sklearn.utils import shuffle
import pandas as pd
pd.set_option('display.expand_frame_repr', False)


###############################################################################
# load training set
###############################################################################
data = pd.read_csv('../filtered_data/filter_train.csv')

###############################################################################
# random split data
###############################################################################
# shuffle the table
data = shuffle(data, random_state = 0)

# split the table into one holdout table and 5 equal size tables
df_train1 = data.iloc[0: 10000000]
df_train2 = data.iloc[10000000: 20000000]

###############################################################################
# save tables
###############################################################################
df_train1.to_csv('../filtered_data/filter_train1.csv', index=True)

df_train2.to_csv('../filtered_data/filter_train2.csv', index=True)