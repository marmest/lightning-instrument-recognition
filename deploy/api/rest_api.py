# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:25:06 2023

@author: Francek
"""

from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model 
import numpy as np
from data_preprocessing import extract_from_file

ALLOWED_EXTENSIONS = set(['wav'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1).lower() in ALLOWED_EXTENSIONS

step_perc = 1.0 #koliko nam je step kad segmentiramo spektrogram - default 100%
predictions_map = {0 : "cel", 1 : "cla", 2 : "flu", 3 : "gac", 4 : "gel", 5 : "org",
             6 : "pia", 7 : "sax", 8 : "tru", 9 : "vio", 10 : "voi"}
label_map = {"cel": 0, "cla": 1, "flu": 2, "gac": 3, "gel": 4, "org": 5,
             "pia": 6, "sax": 7, "tru": 8, "vio": 9, "voi": 10}

model_mel = load_model("C:/Users/Fran/novi_lumen/models/cnn_mel_85_23.h5")
model_modgd = load_model("C:/Users/Fran/novi_lumen/models/cnn_modgd_85_25.h5")
model_pitch = load_model("C:/Users/Fran/novi_lumen/models/cnn_pitch_85_12.h5")

def predict_instrument(audio_grams, type):
    val = np.reshape(audio_grams, (audio_grams.shape[0], audio_grams.shape[1], audio_grams.shape[2], 1))
    if type == 'mels':
        prediction = model_mel.predict(val)
    elif type == 'modgds':
        prediction = model_modgd.predict(val)
    else:
        prediction = model_pitch.predict(val)
    prediction = np.sum(prediction, axis = 0)
    m = np.max(prediction)
    prediction /= m
    return prediction

# -----------------------------------------

app = Flask(__name__)

@app.route('/', methods=['GET'])
def render():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # TO-DO: dodati slučaj kada file nije ispravne ekstenzije i kada uopće nije uploadan file
    audio_file = request.files['audiofile']
    audio_grams = extract_from_file(audio_file, api = True)

    prediction_mels = predict_instrument(audio_grams['mels'], 'mels')
    prediction_modgds = predict_instrument(audio_grams['modgds'], 'modgds')
    prediction_pitchs = predict_instrument(audio_grams['pitchs'], 'pitchs')

    predictions_all = []
    final_instruments = []
    for i in range(11):
        predictions_all.append((prediction_mels[i] + prediction_modgds[i] + prediction_pitchs[i]) / 3)
        if predictions_all[i] > 0.5:
            final_instruments.append(predictions_map[i])

    return render_template('index.html', predictions = final_instruments)

@app.route('/predict_test', methods=['POST'])
def predict_test():
    # TO-DO: dodati slučaj kada file nije ispravne ekstenzije i kada uopće nije uploadan file
    
    audio_file = request.files['audiofile']

    audio_grams = extract_from_file(audio_file, api = True)

    prediction_mels = predict_instrument(audio_grams['mels'], 'mels')
    prediction_modgds = predict_instrument(audio_grams['modgds'], 'modgds')
    prediction_pitchs = predict_instrument(audio_grams['pitchs'], 'pitchs')

    dict = {}
    i = 0
    for key in label_map.keys():   
        if (prediction_mels[i] + prediction_modgds[i] + prediction_pitchs[i]) / 3 > 0.5:
            dict[key] = 1
        else:
            dict[key] = 0
        i = i + 1
            
    return dict

if __name__ == '__main__':
    app.run(port=5050, debug=True)