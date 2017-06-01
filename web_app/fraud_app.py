from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hello')
def hello():
    return render_template('hello.html')

@app.route('/problem')
def problem():
    return render_template('problem.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/score', methods=['POST'])
def model_predict():
    body_length = request.form['body_length']
    sale_duration2 = request.form['sale_duration2']
    user_age= request.form['user_age']
    name_length = request.form['name_length']
    payee_name = request.form['payee_name']
    user_type = request.form['user_type']
    fb_published = request.form['fb_published']

    X = 0
    y = model.predict(X)
    
    return y

if __name__ == '__main__':
    with open('../model/final_model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    conn = psycopg2.connect(dbname='eventdata', user='postgres', password='password', host='localhost')
    c = conn.cursor()

    app.run(host='0.0.0.0', port=8080, debug=True)
