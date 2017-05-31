import random
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

class Model(object):
    def __init__(self):
        """A Random Forest classifier
            -Fits a Random Forest model to seven of features predicting fraud (1) or not fraud (0).

        The class implemens fit, predict, score interface.
        """
        self._classifier = RandomForestClassifier(n_estimators=25, random_state=15)

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
        df = pd.concat([X, y], axis=1)

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

    def score(self, X, y):
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

    # features included in Random Forest
    model_feats = ['body_length', 'sale_duration2', 'user_age', 'name_length', 'payee_ind', 'user_type', 'fb_published']

    y = df.pop('fraud')
    X = df[model_feats]

    return X, y


if __name__ == '__main__':
    X, y = get_data('../data/data.json')
    model = Model()
    model.fit(X, y)
    with open('final_model.pkl', 'wb') as f:
        pickle.dump(model, f)
