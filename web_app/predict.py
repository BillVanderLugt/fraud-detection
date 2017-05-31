from build_model import Model, get_data
import pickle



with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)

X, y = get_data('data/test_script_examples.json')

print("Accuracy:", model.score(X, y))
print("Predictions:", model.predict(X))
