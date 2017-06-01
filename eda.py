import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

df = pd.read_json('data/data.json')

def make_target(df):
    fraud_map = {'fraudster_event': 1, 'fraudster': 1, 'fraudster_att': 1, \
                 'premium': 0, 'spammer_limited': 0, 'spammer_warn': 0, \
                 'tos_warn': 0, 'locked': 0, 'spammer_web': 0, 'spammer': 0}
    df['fraud'] = df['acct_type'].map(fraud_map)
    return df

def plot_(df):
    plt.close()
    df['age_bins'] = pd.cut(df['user_age'].values, 100)
    age_dic = {}
    for bins in df['age_bins'].unique():
        age_dic[bins[1:5]] = df[(df['age_bins'] == bins) & (df['fraud'] == 1)]['fraud'].count()\
                        / df['fraud'].count()
    x = [float(k) for k, v in age_dic.items()]
    y = [v for k, v in age_dic.items()]
    sort_lis = sorted(list(zip(x, y)))
    x = [a for a, b in sort_lis]
    y = [b for a, b in sort_lis]
    plt.plot(x, y, color='b')
    plt.title('Percent Fraud over Account Age')
    plt.xlabel('Age of Account')
    plt.ylabel('Percent Fraud')
    plt.ylim((-5, 2500))
    plt.savefig('fraud_over_age.png')


if __name__ == '__main__':
    df = make_target(df)
    fraud_df = df[df['fraud'] == 1]
    pickle.dumps(df, 'df.pkl')
    plot_(df)
