
from __future__ import print_function

import argparse
import os
import pandas as pd
import json
from io import StringIO
from sklearn.externals import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    #Custom arguments
    parser.add_argument('--model', type=str)
    parser.add_argument('--strip_accents', type=str, default='ascii')
    parser.add_argument('--token_pattern', type=str, default=u'(?ui)\\b\\w*[a-z]+\\w*\\b') 
    parser.add_argument('--lowercase', type=bool, default=True)
    parser.add_argument('--stop_words', type=str, default='english')
    parser.add_argument('--min_df', type=int, default=5)
    parser.add_argument('--ngram_min',type=int, default=1)
    parser.add_argument('--ngram_max',type=int, default=2)

    args = parser.parse_args()
    
    param_dict = {'strip_accents': args.strip_accents, 
                  'token_pattern': args.token_pattern,
                  'lowercase': args.lowercase,
                  'stop_words': args.stop_words,
                  'min_df': args.min_df,
                  'ngram_range': (args.ngram_min, args.ngram_max)}
    
    # Take the set of files and read them all into a single pandas dataframe
    dataloc = os.path.join(args.train, "{0}.json".format(args.model))
    with open(dataloc) as data_file:
        data = json.load(data_file)
    
    train_df = pd.io.json.json_normalize(data['data'])
    
    #Identifying keywords that show up in every department's comments
    for i, group in enumerate(train_df.groupby('fire_department.firecares_id')):
        vectorizer = CountVectorizer(**param_dict)
        vectorizer.fit(group[1]['description.comments'])
        if i==0:
            keywords = set(vectorizer.get_feature_names())
        keywords = keywords & set(vectorizer.get_feature_names())
    
    #Building the pipeline
    nb = make_pipeline(CountVectorizer(**param_dict, vocabulary=keywords), MultinomialNB())
    nb.fit(train_df['description.comments'], train_df['target'])

    #Saving the trained model
    joblib.dump(nb, os.path.join(args.model_dir, "model.joblib"))

def model_fn(model_dir):
    """
    Deserialized and return fitted model
    
    """
    
    nb = joblib.load(os.path.join(model_dir, "model.joblib"))
    return nb

def predict_fn(input_data, model):
    pred_prob = model.predict_proba(input_data)[:,1]
    return pred_prob
