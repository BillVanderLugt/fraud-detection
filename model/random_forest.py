import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def run_forest(X_train, X_test, y_train, y_test):
    print('Training Random Forest...')
    rf = RandomForestClassifier(n_estimators=25, criterion='gini', random_state=15)
    rf.fit(X_train, y_train)

    # print('Train accuracy: {}'.format(rf.score(X_train, y_train)))
    # print('Test accuracy: {}'.format(rf.score(X_test, y_test)))
    #
    # y_pred_train = rf.predict(X_train)
    # print('Train f1: {}'.format(f1_score(y_train, y_pred_train)))
    # y_pred_test = rf.predict(X_test)
    # print('Test f1: {}'.format(f1_score(y_test, y_pred_test)))

    print('Cross val scores:')
    print('accuracy', np.mean(cross_val_score(rf, X_train, y_train, scoring='accuracy')))
    print('precision', np.mean(cross_val_score(rf, X_train, y_train, scoring='precision')))
    print('recall', np.mean(cross_val_score(rf, X_train, y_train, scoring='recall')))
    print('f1', np.mean(cross_val_score(rf, X_train, y_train, scoring='f1')))

    print('feature importances', rf.feature_importances_)

    # test prediciton
    y_test_pred = rf.predict(X_test)
    print('f1 test', f1_score(y_test, y_test_pred))




if __name__ == '__main__':
    df = pd.read_json('data/data.json')

    fraud_map = {'fraudster_event': 1, 'fraudster': 1, 'fraudster_att': 1, 'premium': 0, 'spammer_limited': 0, 'spammer_warn': 0, 'tos_warn': 0, 'locked': 0, 'spammer_web': 0, 'spammer': 0, 'spammer_noinvite': 0, 'tos_lock': 0}

    df['fraud'] = df['acct_type'].map(fraud_map)

    # under sample for even classes
    fraud = df.loc[df.fraud == 1]
    n_samps = fraud.shape[0]
    not_fraud = df.loc[df.fraud == 0].sample(n_samps, random_state=15)
    df = pd.concat([fraud, not_fraud])

    # some feature engineering
    df['payee_ind'] = df['payee_name'].map(lambda x: 1*(len(x) > 0)) # 1 if there's a payee id listed

    # is listed
    df['listed_ind'] = df['listed'].map({'y': 1, 'n': 0})

    # has org description
    df['has_org_desc'] = df['org_desc'].map(lambda x: 1*(len(x) > 0))

    # number of types of tickets sold
    df['num_types_tickets'] = df['ticket_types'].map(lambda d: len(d))

    # create features and target
    y = df.pop('fraud')

    # numeric columns to be imputed
    has_nans = ['delivery_method', 'sale_duration', 'has_header', 'delivery_method',]

    sig_num_columns = ['body_length', 'sale_duration2', 'user_age', 'name_length', 'payee_ind', 'user_type', 'fb_published']

    X = df[sig_num_columns]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=15)

    print('Training on {} samples'.format(X_train.shape[0]))
    print('Testing on {} samples'.format(X_test.shape[0]))

    run_forest(X_train, X_test, y_train, y_test)
