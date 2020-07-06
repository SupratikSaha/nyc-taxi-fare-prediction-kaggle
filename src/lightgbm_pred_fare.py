"""
LGBM model for fare prediction
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
from sklearn.externals import joblib

###############################################################################
# set up parameter
###############################################################################
# file for saving processed dataset
df_train_file = '../all/processed_filtered_data/processed_train1_geoid.csv'

###############################################################################
# load dataset
###############################################################################
df_train = pd.read_csv(df_train_file)
df_train = df_train.drop(columns=['pickup_datetime'])

###############################################################################
# fit model
###############################################################################
# get feature and label
Y = df_train['fare_amount']
X = df_train.drop(columns=['fare_amount'])

# split the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    random_state=10,
                                                    test_size=0.1)

# remove unused dataset
del df_train
del Y
del X
gc.collect()

# set up lgb parameter
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'nthread': 4,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'max_depth': -1,
    'subsample': 0.8,
    'bagging_fraction': 1,
    'max_bin': 5000,
    'bagging_freq': 20,
    'colsample_bytree': 0.6,
    'metric': 'rmse',
    'min_split_gain': 0.5,
    'min_child_weight': 1,
    'min_child_samples': 10,
    'scale_pos_weight': 1,
    'zero_as_missing': True,
    'seed': 0,
    'num_rounds': 50000
}


# fit the model
def LGBMmodel(X_train, X_test, y_train, y_test, params):
    matrix_train = lgb.Dataset(X_train, y_train)
    matrix_test = lgb.Dataset(X_test, y_test)
    model = lgb.train(params=params,
                      train_set=matrix_train,
                      num_boost_round=100000,
                      early_stopping_rounds=500,
                      verbose_eval=100,
                      valid_sets=matrix_test)
    return model


# Train the model
model = LGBMmodel(X_train, X_test, Y_train, Y_test, params)

# remove unused dataset
del X_train
del Y_train
del X_test
del Y_test
gc.collect()

###############################################################################
# save result
###############################################################################
# Save to file in the current working directory   
pkl_filename = '../simple_model/lightgbm_train1_geo.pkl'

joblib.dump(model, pkl_filename)

###############################################################################
# prediction
############################################################################### 
# file for saving processed dataset
df_test_file = '../all/processed_filtered_data/processed_test1_geoid.csv'

df_test = pd.read_csv(df_test_file)
X_quiz = df_test.drop(columns=['key'])

# Predict the 'fare_amount' and save file
prediction = model.predict(X_quiz, num_iteration=model.best_iteration)
submission = pd.DataFrame(
    {'key': df_test.key, 'fare_amount': prediction},
    columns=['key', 'fare_amount']
)

submission.to_csv('../all/submission/test_submission/submission_train1_geo.csv',
                  index=False
                  )
