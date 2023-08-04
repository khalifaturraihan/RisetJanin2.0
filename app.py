from flask import Flask, render_template, request, redirect
import pickle

import numpy as np                        # numpy==1.24.3

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':

        with open('rf_pickle.pkl', 'rb') as r:
            model = pickle.load(r)

        accelerations = float(request.form['accelerations'])
        movement = float(request.form['movement'])
        uterine = float(request.form['uterine'])
        light = float(request.form['light'])
        severe = float(request.form['severe'])
        prolongued = float(request.form['prolongued'])
        abnormal = float(request.form['abnormal'])
        percentage = float(request.form['percentage'])

        datas = np.array((accelerations, movement, uterine, light, severe, prolongued, abnormal, percentage))
        datas = np.reshape(datas, (1, -1))

        isJanin = model.predict(datas)

        return render_template('hasil.html', finalData=isJanin)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
