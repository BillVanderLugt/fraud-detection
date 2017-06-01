from __future__ import print_function
from bs4 import BeautifulSoup
from pprint import pprint
from time import time
import logging
import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

#print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.6),
    #'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf__alpha': (0.001, 0.0001),
    #'clf__penalty': ('l2'),
    'clf__n_iter': (5, 50),
    'clf__loss': ('huber', 'log', 'modified_huber', 'squared_hinge')
}

def grid(X, y):
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

def clean_text(df):
    print ('processing text...')
    text = []
    for i in df.description:
        soup = BeautifulSoup(i.strip(), 'lxml')
        text.append(soup.get_text())
    df['text'] = text
    return df

def equalize_halves(df):
    df_fraud = df[df.fraud==1]
    df_legit = df[df.fraud==0]
    num_fraud = df_fraud.shape[0]
    sample = df.sample(n=num_fraud, replace=False, random_state=1)
    eq_df = pd.concat([df_fraud, sample])
    print ("With {} of each, total size is {}".format(num_fraud, eq_df.shape[0]))
    return eq_df

def best_model(X, Y):
    '''
    returns maxtrix of two columns returning probabilities of each outcome
    '''
    vectX = CountVectorizer(max_df=0.5, ngram_range=(1,2)).fit_transform(X)
    TfidfX = TfidfTransformer().fit_transform(vectX)
    probs = SVC(probability=True).fit(TfidfX,Y).predict_proba(TfidfX)
    return probs

    # Best parameters set:
    # 	clf__alpha: 0.0001
    # 	vect__max_df: 0.5
    # 	vect__max_features: 50000
    # 	vect__ngram_range: (1, 2)

if __name__ == '__main__':

    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier

    print ('loading...')
    df = pd.read_json('data/data.json')

    fraud_map = {'fraudster_event': 1, 'fraudster': 1, 'fraudster_att': 1, 'premium': 0, 'spammer_limited': 0, 'spammer_warn': 0, 'tos_warn': 0, 'locked': 0, 'spammer_web': 0, 'spammer': 0, 'spammer_noinvite': 0, 'tos_lock': 0}

    df['fraud'] = df['acct_type'].map(fraud_map)

    print ('equalizing halves...')
    df = equalize_halves(df)
    print ("df down to ", df.shape[0], " rows.")
    # clean out html from description
    "cleaning..."
    df = clean_text(df)

    #perform grid search
    #grid(df.text, df.fraud)
    probs = best_model(df.text, df.fraud)
    print (probs)
'''

{'clf__alpha': (0.0001, 1e-05),
 'clf__n_iter': (10, 50),
 'vect__max_df': (0.3, 0.5),
 'vect__ngram_range': ((1, 2), (1, 3))}
Fitting 3 folds for each of 16 candidates, totalling 48 fits
[Parallel(n_jobs=-1)]: Done  48 out of  48 | elapsed:  1.3min finished
done in 79.272s

Best score: 0.831
Best parameters set:
	clf__alpha: 0.0001
	clf__n_iter: 50
	vect__max_df: 0.5
	vect__ngram_range: (1, 2)

    {'clf__alpha': (0.001, 0.0001),
 'vect__max_df': (0.5, 0.6),
 'vect__max_features': (None, 5000, 10000, 50000),
 'vect__ngram_range': ((1, 1), (1, 2))}
Fitting 3 folds for each of 32 candidates, totalling 96 fits
[Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   29.3s
[Parallel(n_jobs=-1)]: Done  96 out of  96 | elapsed:  1.1min finished
done in 69.861s

###########################################
Best score: 0.832
Best parameters set:
	clf__alpha: 0.0001
	vect__max_df: 0.5
	vect__max_features: 50000
	vect__ngram_range: (1, 2)
'''
