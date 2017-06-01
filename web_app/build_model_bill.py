import random
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import SVC
from bs4 import BeautifulSoup
from sklearn.model_selection import cross_val_score
import logging
from time import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

class Model(object):
    def __init__(self):
        """A Random Forest classifier
            -Fits a Random Forest model to seven of features predicting fraud (1) or not fraud (0).

        The class implemens fit, predict, score interface.
        """
        self._classifier = RandomForestClassifier(n_estimators=25, random_state=15)

    def _text_classifier(self, X, Y):
        '''
        Parameters
        ----------
        X: features dataframe
        y: labels dataframe

        Returns
        -------
        fraud_prob: returns prob of fraud based on text analysis
        '''
        print('counting words...')
        X = X['text']
        self._vectX = CountVectorizer(max_df=0.5, ngram_range=(1,1))
        self._vectX_fitted = self._vectX.fit(X)
        self._vectX_transformed = self._vectX_fitted.transform(X)
        print('TFIDFing words...')
        self._TfidfX = TfidfTransformer()
        self._TfidfX_fitted = self._TfidfX.fit(self._vectX_transformed)
        self._TfidfX_transformed = self._TfidfX_fitted.transform(self._vectX_transformed)
        t0 = time()
        print('fitting model...')
        self._text_model = SVC(probability=True)
        self._text_model_fitted = self._text_model.fit(self._TfidfX_transformed,Y)
        print("done in %0.3fs" % (time() - t0))
        print('getting probabilities...')
        self._probs = self._text_model.predict_proba(self._TfidfX_transformed)[:, 1]
        return self._probs

    def fit(self, X, y):
        """Fit the Random Forest Model by implementing under-sampling to create balanced classes.

        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.
        y: A numpy array or python list of labels, to be used as responses.

        Returns
        -------
        self: The fit model object.
        """
        # join to maintain relativity of features to labels when sampling from classes
        X['text_prob'] = self._text_classifier(X, y)
        df = pd.concat([X, y], axis=1)
        del df['text']
        print(X.head())

        # under sample to create balanced classes
        fraud = df.loc[df.fraud == 1]
        n_samps = fraud.shape[0]
        not_fraud = df.loc[df.fraud == 0].sample(n_samps, random_state=15)
        df = pd.concat([fraud, not_fraud])

        y = df.pop('fraud')
        X = df

        self._classifier.fit(X, y)
        return self

    def predict(self, X):
        """Make probability predictions on new data.

        Parameters
        ----------
        X: A pandas dataframe of features to predict on.

        Returns
        -------
        probs: A (n_obs, n_classes) numpy array of predicted class probabilities.
        """

        return self._classifier.predict_proba(X)

    def _predict_text(self, text):
        print ("Whole input text:")
        print (text)
        text = text.split(' ')
        vocab = model._vectX.vocabulary_
        overlap = [(word, vocab[word]) for word in text if word in vocab]
        print ("Overlapping terms:", overlap)
        vectX = CountVectorizer(vocabulary=vocab).transform(text)
        #vectX = self._vectX_fitted.transform(text).todense()
        print ('vectX', vectX)
        print ('overlap', overlap)

        print('finding 3 most similar...')
        training = self._TfidfX_transformed
        dense = training.todense()
        # similarities = cosine_similarity(training, vectX)
        # print ('cols in training', dense.shape[0])
        # print ('length of new vector', len(vectX))
        # print ('outcome', similarities)
        similarities = None
        # TfidfX = self._TfidfX_fitted.transform(vectX).todense()
        # print ('TfidfX: ', TfidfX)
        # print ('stopwords', TfidfX.stop_words_)
        # print ('idf', TfidfX.idf_)
        return similarities, dense

    def score(self, X, y):
        print('scoring model...')
        return self._classifier.score(X, y)

def get_data(datafile):
    """Load raw data from a file and return training data and responses.

    Parameters
    ----------
    filename: The path to a json file containing the features and target data.

    Returns
    -------
    X: A pandas dataframe containing features used in the model.
    y: A pandas containing labels, used for model response.
    """
    df = pd.read_json(datafile)

    # map acct_type variable to target (1 for fraud; 0 for not)
    fraud_map = {'fraudster_event': 1, 'fraudster': 1, 'fraudster_att': 1, 'premium': 0, 'spammer_limited': 0, 'spammer_warn': 0, 'tos_warn': 0, 'locked': 0, 'spammer_web': 0, 'spammer': 0, 'spammer_noinvite': 0, 'tos_lock': 0}
    df['fraud'] = df['acct_type'].map(fraud_map)

    # feature engineering
    df['payee_ind'] = df['payee_name'].map(lambda x: 1*(len(x) > 0)) # 1 if there's a payee id listed

    # clean text
    print ('processing text...')
    t0 = time()
    text = []
    for i in df.description:
        soup = BeautifulSoup(i.strip(), 'lxml')
        text.append(soup.get_text())
    df['text'] = text
    print("done in %0.3fs" % (time() - t0))
    print()

    # features included in Random Forest
    model_feats = ['body_length', 'sale_duration2', 'user_age', 'name_length', 'payee_ind', 'user_type', 'fb_published', 'text']

    y = df.pop('fraud')
    X = df[model_feats]

    return X, y

def load_model():
    '''
    returns unpickled model
    '''
    with open("final_model.pkl", 'rb') as f_un:
        file_unpickled = pickle.load(f_un)
    return file_unpickled

def toy(X, y):
    tfidf = TfidfVectorizer(max_features=10)
    tfidf.fit(X)
    X_vecs = tfidf.transform(X)
    mod = SVC(probability=True)
    mod.fit(X_vecs, y)
    #probs = mod.predict_proba(X_vecs)
    #print (probs)
    return X_vecs, tfidf

if __name__ == '__main__':
    #X, y = get_data('data/data.json')
    model = load_model()
    #model = Model()

    # t0 = time()
    # model.fit(X, y)
    # print("done in %0.3fs" % (time() - t0))
    # print()


    # t0 = time()
    # print("Training score:")
    # print(model.score(X, y))
    # print("done in %0.3fs" % (time() - t0))
    # print()

    # print ('pickling model...')
    # del X['text']
    # with open('final_model.pkl', 'wb') as f:
    #     pickle.dump(model, f)

    dummy_text = 'This is simply an inexplicable, ineluctable test party text for testing without much test or planning.'
    X = ['fraud shady sketchy',
             'clean fine ok',
             'shady tree blah']
    y = [1, 0, 0]

    #sims, dense = model._predict_text(dummy_text)
    X_vecs, tfidf = toy(X, y)
    print (X_vecs.todense())
    print (tfidf.vocabulary_)
    print (tfidf.idf_)
