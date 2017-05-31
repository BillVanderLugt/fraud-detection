from flask import Flask, render_template
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
