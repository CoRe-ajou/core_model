import tensorflow as tf
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from get_elmo_vector import GetELMoVector
import random
import pandas as pd
import numpy as np
import csv

def create_model() :
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(10,return_sequences=True), input_shape = (256,256)))
    model.add(tf.keras.layers.Conv1D(filters=250, kernel_size=3, padding='same'))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(250, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def predict_model():
    with graph.as_default():

        #model._make_predict_function()
        preds = model.predict(input_array)
        #tf.keras.backend.clear_session()
        print(preds)

global input_array
input_method = GetELMoVector()
input_array = input_method.get_elmo_vector("ㅈㄴ 울먹이네 우리가 잘못한건줄 알겟다 ㅋㅋ")

#instantiate flask
#app = flask.Flask(__name__)

# to load the model and save it for the entire environment use graph
global graph
graph = tf.get_default_graph()

model = create_model()
model.load_weights('../data/core-model.h5')

#model predict
predict_model()



