from flask import Flask, render_template, request
import pickle
import sys
sys.path.append('../model')
import final_model
import psycopg2
from predict import predict_and_store

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
def score():
    body_length = request.form['body_length']
    sale_duration2 = request.form['sale_duration2']
    user_age= request.form['user_age']
    name_length = request.form['name_length']
    payee_name = request.form['payee_name']
    user_type = request.form['user_type']
    fb_published = request.form['fb_published']

    record = (body_length,sale_duration2,user_age,name_length,payee_name,user_type,fb_published)

    y = predict_and_store(record,model,conn)
    return render_template('score.html', predicted=y)

if __name__ == '__main__':
    # unpickle model before app run to establish 'model' in global namespace
    with open('../model/final_model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    # ...and do the same with psql connection 'conn'
    conn = psycopg2.connect(dbname='eventdata', user='postgres', host='localhost',password='password')


    app.run(host='0.0.0.0', port=8080, debug=True)
