
from __future__ import print_function

import argparse
import os
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
from pandas.io.json import json_normalize
from dateutil.relativedelta import SU
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from pandas.tseries.holiday import AbstractHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import Holiday
import datetime
from io import StringIO
from sklearn.externals import joblib
import numpy as np



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--n_estimators', type=int, default=1000)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    dataloc = os.path.join(args.train, 'training_data.json') 
    with open(dataloc) as data_file:
        data = json.load(data_file)
    df = pd.io.json.json_normalize(data['prediction_data'])

    #Determine the mean of each department by hour
    means = df[['fire_department.firecares_id','calls']]
    means = means.groupby('fire_department.firecares_id').aggregate(np.mean).reset_index()
    means = means.rename(columns={'calls':'mean_calls'})
    df = df.merge(means, on='fire_department.firecares_id')
    df['norm_calls'] = df['calls']/df['mean_calls']
    features = pd.get_dummies(df.drop(['calls', 'mean_calls', 'norm_calls'], axis=1))
    features = features.reindex(sorted(features.columns),axis=1)
    labels = df['norm_calls']
    
    # Here we support a single hyperparameter, 'max_leaf_nodes'. Note that you can add as many
    # as your training my require in the ArgumentParser above.
    n_estimators = args.n_estimators

    # Now use scikit-learn's random forestion regression model to predict daily ems call volumes
    rf = RandomForestRegressor(n_estimators = n_estimators)

    # Train the model on training data
    rf.fit(features, labels)

#     # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(rf, os.path.join(args.model_dir, "model.joblib"))

    
def input_fn(input_data, content_type):
    features = pd.read_csv(input_data)
    features = pd.get_dummies(features)
    features = features.reindex(sorted(features.columns),axis=1)
    return features

#     data = json.loads(input_data)
#     df = pd.io.json.json_normalize(data['prediction_data'])
#     features = pd.get_dummies(df)
#     return features

def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    rf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return rf
