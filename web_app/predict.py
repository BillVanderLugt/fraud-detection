from build_model import Model, get_data
import pickle
import pandas as pd

raw_data = pd.read_json('data/test_script_examples.json')
line_num = 0
# X, y = X.iloc[[line_num]], y.iloc[[line_num]]

with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)


prob_of_fraud = model.predict(X)[0][1]
print('Probability of fraud:', prob_of_fraud)
