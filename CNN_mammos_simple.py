import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
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


plot_subset_images(x_ddsm,3,4,title = "DDSM")
plot_subset_images(x_mias,3,4, title = "MIAS")
plot_subset_images(x_inbreast,3,4, title = "InBreast")



 ## Mise en forme des données


x_ddsm_train, x_ddsm_test, y_ddsm_train, y_ddsm_test = build_train_test_data(x_ddsm)
x_mias_train, x_mias_test, y_mias_train, y_mias_test = build_train_test_data(x_mias)
x_inbreast_train, x_inbreast_test, y_inbreast_train, y_inbreast_test = build_train_test_data(x_inbreast)


x_ddsm_mias_train = tf.concat([x_ddsm_train, x_mias_train], axis = 0)
x_ddsm_mias_test = tf.concat([x_ddsm_test, x_mias_test], axis = 0)
y_ddsm_mias_train = tf.concat([y_ddsm_train, y_mias_train], axis = 0)
y_ddsm_mias_test = tf.concat([y_ddsm_test, y_mias_test], axis = 0)

x_tout_train = tf.concat([x_ddsm_train,x_mias_train,x_inbreast_train], axis = 0)
x_tout_test = tf.concat([x_ddsm_test,x_mias_test,x_inbreast_test], axis = 0)
y_tout_train = tf.concat([y_ddsm_train,y_mias_train,y_inbreast_train], axis = 0)
y_tout_test = tf.concat([y_ddsm_test,y_mias_test,y_inbreast_test], axis = 0)



## Normalisation

x_ddsm_train = tf.keras.utils.normalize(x_ddsm_train)
x_mias_train = tf.keras.utils.normalize(x_mias_train)
x_inbreast_train = tf.keras.utils.normalize(x_inbreast_train)
x_ddsm_mias_train = tf.keras.utils.normalize(x_ddsm_mias_train)

x_ddsm_test = tf.keras.utils.normalize(x_ddsm_test)
x_mias_test = tf.keras.utils.normalize(x_mias_test)
x_inbreast_test = tf.keras.utils.normalize(x_inbreast_test)
x_ddsm_mias_test = tf.keras.utils.normalize(x_ddsm_mias_test)





## Modèle

model = tf.keras.Sequential()

model.add(layers.Conv2D(10,(3,3),input_shape = (227,227,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(20,(3,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(30,(3,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.BatchNormalization())

model.add(layers.Flatten())

'''
model.add(layers.Dense(128,activation = 'gelu'))
model.add(layers.BatchNormalization())
#model.add(layers.Dropout(0.2))
model.add(layers.Dense(128,activation = 'gelu'))
model.add(layers.BatchNormalization())
#model.add(layers.Dropout(0.2))
'''
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(32,activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(16,activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.05))
model.add(layers.Dense(8,activation = 'relu'))
model.add(layers.BatchNormalization())

model.add(layers.Dense(4,activation = 'relu'))

model.add(layers.Dense(2,activation = 'softmax'))

model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['AUC','accuracy'])

history_model = model.fit(x_ddsm_train,y_ddsm_train,epochs = 50, validation_split = 0.2)

history_model = model.fit(x_mias_train,y_mias_train,epochs = 10)

history_model = model.fit(x_ddsm_mias_train, y_ddsm_mias_train, epochs = 10, validation_split = 0.2)

history_model = model.fit(x_tout_train,y_tout_train,epochs = 6, validation_split = 0.2)


model.evaluate(x_ddsm_test,y_ddsm_test)

model.evaluate(x_mias_test,y_mias_test)

model.evaluate(x_inbreast_test,y_inbreast_test)

model.evaluate(x_ddsm_mias_test, y_ddsm_mias_test)

model.evaluate(x_mias_train,y_mias_train)

model.evaluate(x_tout_test, y_tout_test)

# Plot accuracy :
plt.plot(history_model.history['accuracy'], label = 'Train')
plt.plot(history_model.history['val_accuracy'], label = 'Validation')
plt.title("Précision")
plt.legend()
plt.show()


# Plot loss :
plt.plot(history_model.history['loss'], label = 'Train')
plt.plot(history_model.history['val_loss'], label = 'Validation')
plt.title("Loss")
plt.legend()
plt.show()


## Matrices de confusion


plot_confusion_matrix(x_ddsm_test, y_ddsm_test, model)
plot_confusion_matrix(x_ddsm_train, y_ddsm_train, model)

plot_confusion_matrix(x_mias_test, y_mias_test, model)
plot_confusion_matrix(x_inbreast_test, y_inbreast_test, model)



## Utilisation d'un modèle pré entraîné

# Chargement du modèle ResNet
modelResNet = tf.keras.applications.ResNet50(weights = 'imagenet', include_top = False, input_shape = (227,227,3))

# imputation d'une couche de classification finale de taille 2
model2 = tf.keras.Sequential([modelResNet, tf.keras.layers.Flatten(),tf.keras.layers.Dense(2, activation = 'softmax')])

# Le modèle travaille sur des images RGB. Les nôtres sont GreyScale. On va les répliquer trois fois pour simuler des images RGB.
x_ddsm_train_rgb = make_rgb_from_grayscale(x_ddsm_train)
x_ddsm_test_rgb = make_rgb_from_grayscale(x_ddsm_test)

x_mias_train_rgb = make_rgb_from_grayscale(x_mias_train)
x_mias_test_rgb = make_rgb_from_grayscale(x_mias_test)

# Compilation
model2.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['AUC','accuracy'])

# Entraînement
model2.fit(x_ddsm_train_rgb, y_ddsm_train, epochs = 5)

# Evaluation des performances
model2.evaluate(x_ddsm_test_rgb, y_ddsm_test)
model2.evaluate(x_mias_test_rgb, y_mias_test)
model2.evaluate(x_ddsm_test_rgb, y_ddsm_test)













## Visualisation des filtres


model2 = tf.keras.Sequential()

a = model.layers[0:9]

for layer in a:
    model2.add(layer)



num_image = rd.randint(0,len(X))
img = model2(X)[num_image]

img2 = np.zeros((26,26,3))

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        sum = 0
        for channel in range(img.shape[2]):
            sum += img[x][y][channel]

            if channel == 9:
                img2[x][y][0] = sum/10
                sum = 0

            if channel == 19:
                img2[x][y][1] = sum/10
                sum = 0

            if channel == 29:
                img2[x][y][2] = sum/10
                sum = 0



img = img2



rows,cols = 1,2

fig = plt.figure()



fig.add_subplot(rows,cols,1)
plt.imshow(img)

fig.add_subplot(rows,cols,2)
plt.imshow(X[num_image])

plt.show()





















