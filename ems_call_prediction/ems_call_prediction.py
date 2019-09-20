
from __future__ import print_function

import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
    dataloc = os.path.join(args.train, 'query_results.json') 
    df = pd.read_json(dataloc).reset_index(drop=True)
    
    #Converting date
    df['date'] = df['description.event_opened'].apply(lambda x: x[:10])
    #Aggregation function
    def myagg(x):

        #First need to group
        d = {
            'ems_calls': np.sum(x['NFPA.type']=='EMS'),
            'snow': 'snow' in x['weather.daily.precipType'].values,
            'rain': 'rain' in x['weather.daily.precipType'].values,
            'high_temp': np.mean(x['weather.daily.temperatureHigh'])
        }

        return pd.Series(d,index=d.keys())

    #Day aggregation
    features = df.groupby(['fire_department.firecares_id','date']).apply(myagg).reset_index()

    #Adding day of week
    features = features.merge(df[['date','description.day_of_week']].drop_duplicates(), on='date')
    #Renaming the day of week column to make it shorter
    features = features.rename(columns={'description.day_of_week':'day'})
    features['month'] = features.apply(lambda x: x['date'][5:7], axis=1)
    #No longer need the date since we have all the information we need (day of week and month)
    features = features.drop('date',axis=1)
    #Using one hot encoding for categorical variables. Ask me if you want me to explain this further.
    features = pd.get_dummies(features)

    #Splitting the data into features (predictors) and labels (the quantity we want to predict)
    labels = features['ems_calls']
    features = features.drop('ems_calls',axis=1)

    # Here we support a single hyperparameter, 'max_leaf_nodes'. Note that you can add as many
    # as your training my require in the ArgumentParser above.
    n_estimators = args.n_estimators

    # Now use scikit-learn's random forestion regression model to predict daily ems call volumes
    rf = RandomForestRegressor(n_estimators = n_estimators)
    # Train the model on training data
    rf.fit(features, labels)

#     # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(rf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    rf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return rf
