import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import rec_model

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_name = request.form.get("title")
    df = model(int_name)

    return render_template('index.html', tables=[df.to_html(classes='data')], titles=df.columns.values)

if __name__ == "__main__":
    app.run(debug=True)