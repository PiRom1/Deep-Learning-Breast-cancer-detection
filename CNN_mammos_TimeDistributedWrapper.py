import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import numpy as np
from tensorflow.keras import layers
import random as rd
from PIL import Image

import os


os.chdir('/Users/rp2/Documents/Python')

from my_functions import *

## Chargement des données



x_ddsm = import_data("DDSM_Dataset")
x_mias = import_data("MIAS Dataset")
x_inbreast = import_data("INbreast Dataset")


plot_subset_images(x_ddsm,3,4, cmap = 'magma')
plot_subset_images(x_mias,3,4)
plot_subset_images(x_inbreast,3,4)


## Création des séries temporelles


x_serie_ddsm, y_serie_ddsm = build_temporal_series(x_ddsm)
x_serie_mias, y_serie_mias = build_temporal_series(x_mias)
x_serie_inbreast, y_serie_inbreast = build_temporal_series(x_inbreast)


## Mise en forme des données (tenseurs, ... )

x_serie_ddsm_train, x_serie_ddsm_test, y_serie_ddsm_train, y_serie_ddsm_test = build_train_test_serie(x_serie_ddsm, y_serie_ddsm)
x_serie_mias_train, x_serie_mias_test, y_serie_mias_train, y_serie_mias_test = build_train_test_serie(x_serie_mias, y_serie_mias)
x_serie_inbreast_train, x_serie_inbreast_test, y_serie_inbreast_train, y_serie_inbreast_test = build_train_test_serie(x_serie_inbreast, y_serie_inbreast)


## Modèle

inputs = tf.keras.Input(shape = (5,227,227,1))

conv_2d_layer = tf.keras.layers.Conv2D(8, (3, 3))
outputs = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs)
outputs = layers.TimeDistributed(layers.MaxPooling2D((2,2)))(outputs)
outputs = layers.TimeDistributed(layers.BatchNormalization())(outputs)
'''
outputs = tf.keras.layers.Conv2D(8,(3,3))(outputs)
outputs = layers.MaxPooling2D((2,2))(outputs)
outputs = layers.BatchNormalization(outputs)

outputs = tf.keras.layers.Conv2D(8,(3,3))(outputs)
outputs = layers.MaxPooling2D((2,2))(outputs)
outputs = layers.BatchNormalization(outputs)
'''
outputs = layers.TimeDistributed(layers.Flatten())(outputs)

#outputs = layers.Flatten(outputs)
outputs = layers.LSTM(256, activation='relu', return_sequences=False)(outputs)

#outputs = layers.Dense(64,activation = 'gelu')(outputs)
outputs = layers.Dense(32,activation = 'gelu')(outputs)
#outputs = layers.Dense(8,activation = 'gelu')(outputs)
outputs = layers.Dense(2,activation = 'softmax')(outputs)



model = tf.keras.Model(inputs,outputs)

model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy','AUC'])


history_model = model.fit(x_serie_ddsm_train,y_serie_ddsm_train,epochs = 8, validation_split = 0.2)


model.evaluate(x_serie_ddsm_test, y_serie_ddsm_test)
model.evaluate(x_serie_mias_test, y_serie_mias_test)
model.evaluate(x_serie_inbreast_test, y_serie_inbreast_test)


plot_confusion_matrix(x_serie_ddsm_test, y_serie_ddsm_test, model)
plot_confusion_matrix(x_serie_mias_test, y_serie_mias_test, model)
plot_confusion_matrix(x_serie_inbreast_test, y_serie_inbreast_test, model)


























