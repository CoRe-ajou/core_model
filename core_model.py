import tensorflow as tf
from get_elmo_vector import GetELMoVector
import random
import pandas as pd
import numpy as np
import csv

#embedded model input
test_x = np.load('elmo/pad_vector_test.npy')
train_x = np.load('elmo/pad_vector_train.npy')

test_y = np.load('elmo/y_test.npy')
train_y = np.load('elmo/y_train.npy')

#match shape
train_y = np.asarray(train_y).astype('float32').reshape((-1,1))
test_y = np.asarray(test_y).astype('float32').reshape((-1,1))

print(train_y.shape, test_y.shape)
print(train_x.shape, test_x.shape)

#core_cnn_layer
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=250, kernel_size=3, padding='same'))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(250, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=10, batch_size=32)

#model summary
model.summary()

#model predict
input_method = GetELMoVector()
input_array = input_method.get_elmo_vector("ㅈㄴ 울먹이네 우리가 잘못한건줄 알겟다 ㅋㅋ")
result = model.predict(input_array)
print(result)