from flask import Flask, request, make_response, jsonify, Response
from flask_cors import CORS
import urllib.request, json
import datetime as dt
import pandas as pd
import numpy as np
import pickle as p
import re
import string

from utils import *

app = Flask(__name__)
CORS(app)
# app.listen(process.env.PORT || 3000)

@app.route("/")
def hello():
    return ('hello world')

@app.route("/predict", methods=['POST'])
def predict_sign():
    xray_instance = request.get_json(silent=True, force=True)

    unseen_base64 = xray_instance['base64_str']

    predicted_status = save_and_predict(unseen_base64)

    prediction = {'prediction': str(predicted_status)}
    prediction = json.dumps(prediction)
    print (prediction)

    resp = Response(prediction, status=200, mimetype='application/json')
    return resp

if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 5000))
    # app.run(host='0.0.0.0', port=port)
    # get_specific_invoice('1211')
    # vec_loc = 'models/vectorizer.pickle'
    # vec = p.load(open(vec_loc, 'rb'))
    app.debug = True
    app.run()
