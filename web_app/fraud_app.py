from flask import Flask, render_template, request
import pickle
import sys
from build_model import Model
import psycopg2
import socket
from predict import predict_and_store
from ping_server import ping
import requests

app = Flask(__name__)
PORT = 8080
REGISTER_URL = "http://galvanize-case-study-on-fraud.herokuapp.com/data_point"

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

@app.route('/score', methods=['GET','POST'])
def score():
    record = ping()
    y = predict_and_store(record,model,conn)
    return render_template('score_prompt.html', predicted=y)

@app.route('/score_prompt', methods=['POST'])
def score_prompt():
    body_length = request.form['body_length']
    sale_duration2 = request.form['sale_duration2']
    user_age= request.form['user_age']
    name_length = request.form['name_length']
    payee_name = request.form['payee_name']
    user_type = request.form['user_type']
    fb_published = request.form['fb_published']

    if len(payee_name) > 0:
        payee_ind = 1
    else:
        payee_ind = 0

    record = (body_length,sale_duration2,user_age,name_length,payee_ind,user_type,fb_published)

    y = predict_and_store(record,model,conn)
    return render_template('score_prompt.html', predicted=y)

@app.route('/check')
def check():
    line1 = "Number of data points: {0}".format(len(DATA))
    if DATA and TIMESTAMP:
        dt = datetime.fromtimestamp(TIMESTAMP[-1])
        data_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        line2 = "Latest datapoint received at: {0}".format(data_time)
        line3 = DATA[-1]
        output = "{0}\n\n{1}\n\n{2}".format(line1, line2, line3)
    else:
        output = line1
    return output, 200, {'Content-Type': 'text/css; charset=utf-8'}

def register_for_ping(ip, port):
    registration_data = {'ip': ip, 'port': port}
    requests.post(REGISTER_URL, data=registration_data)


if __name__ == '__main__':
    # unpickle model before app run to establish 'model' in global namespace
    with open('data/pure_rf_model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    # ...and do the same with psql connection 'conn'
    conn = psycopg2.connect(dbname='eventdata', user='postgres', host='localhost',password='password')

    # Register for pinging service
    ip_address = socket.gethostbyname(socket.gethostname())
    print("attempting to register %s:%d" % (ip_address, PORT))
    register_for_ping(ip_address, str(PORT))

    app.run(host='0.0.0.0', port=8080, debug=True)
