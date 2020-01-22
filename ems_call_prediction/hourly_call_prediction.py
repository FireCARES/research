
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


#Feature list
feature_list = ['fire_department.firecares_id_79592',
       'fire_department.firecares_id_93345', 'description.day_of_week_Friday',
       'description.day_of_week_Monday', 'description.day_of_week_Saturday',
       'description.day_of_week_Sunday', 'description.day_of_week_Thursday',
       'description.day_of_week_Tuesday', 'description.day_of_week_Wednesday',
       'hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05',
       'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11',
       'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17',
       'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23']  


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
    features = pd.io.json.json_normalize(data['prediction_data'])
    
    #Converting df dates to datetime objects for holiday merging
    features['date'] = features.apply(lambda x: datetime.datetime.strptime(x['date'],'%Y-%m-%d'),axis=1)
    
    #We can create new rules with this class inheritance
    class custom_calendar(AbstractHolidayCalendar):
        new_rules = [
            Holiday('Halloween', month=10,day=31),
            Holiday('Christmas Eve', month=12,day=24),
            Holiday('New Years Eve', month=12,day=31),
            Holiday('DST time change', month=3, day=1, offset=pd.DateOffset(weekday=SU(2)))
        ]
        rules = calendar().rules + new_rules

    cal = custom_calendar()
    start = np.min(features['date'])
    end = np.max(features['date'])
    holidays = cal.holidays(start=start,end=end,return_name=True).reset_index()
    holidays = holidays.rename(columns={'index':'date', 0:'holiday'})

    #But really we want the Monday after the time change
    holidays['date'] = holidays.apply(lambda x: x['date'] + datetime.timedelta(days=1) 
                   if x['holiday'] == 'DST time change' else x['date'], axis=1)

    features = features.merge(holidays, on='date', how='left')
    features['holiday'] = features['holiday'].fillna('none')
    #Filling missing values with the average
    features['high_temp'] = features['high_temp'].fillna(np.mean(features['high_temp']))

    #No longer need the date since we have all the information we need (day of week and month)
    features = features.drop('date',axis=1)
    #Using one hot encoding for categorical variables. Ask me if you want me to explain this further.
    features = pd.get_dummies(features)

    #Splitting the data into features (predictors) and labels (the quantity we want to predict)
    labels = features['ems_calls']
    features = features.drop('ems_calls',axis=1)
    
    #Making sure the column order is correct
    features = features[feature_list]

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
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    
   
    data = json.loads(input_data)
    features = pd.io.json.json_normalize(data['prediction_data'])
    
    #Converting df dates to datetime objects for holiday merging
    features['date'] = features.apply(lambda x: datetime.datetime.strptime(x['date'],'%Y-%m-%d'),axis=1)
    
    #We can create new rules with this class inheritance
    class custom_calendar(AbstractHolidayCalendar):
        new_rules = [
            Holiday('Halloween', month=10,day=31),
            Holiday('Christmas Eve', month=12,day=24),
            Holiday('New Years Eve', month=12,day=31),
            Holiday('DST time change', month=3, day=1, offset=pd.DateOffset(weekday=SU(2)))
        ]
        rules = calendar().rules + new_rules

    cal = custom_calendar()
    start = np.min(features['date'])
    end = np.max(features['date'])
    holidays = cal.holidays(start=start,end=end,return_name=True).reset_index()
    holidays = holidays.rename(columns={'index':'date', 0:'holiday'})

    #But really we want the Monday after the time change
    holidays['date'] = holidays.apply(lambda x: x['date'] + datetime.timedelta(days=1) 
                   if x['holiday'] == 'DST time change' else x['date'], axis=1)

    features = features.merge(holidays, on='date', how='left')
    features['holiday'] = features['holiday'].fillna('none')
    #Filling missing values with the average
    features['high_temp'] = features['high_temp'].fillna(np.mean(features['high_temp']))

    #No longer need the date since we have all the information we need (day of week and month)
    features = features.drop('date',axis=1)
    #Using one hot encoding for categorical variables. Ask me if you want me to explain this further.
    features = pd.get_dummies(features)
    

    
    for feature in feature_list:
        if feature not in features.columns:
            features[feature] = 0
            
    #Making sure the column order is correct
    features = features[feature_list]
    
    return features

def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    rf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return rf
