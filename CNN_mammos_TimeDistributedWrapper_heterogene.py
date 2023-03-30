import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import numpy as np
from tensorflow.keras import layers
import random as rd
from PIL import Image

import os

os.chdir('/Users/rp2/Documents/Stage_3A/Données/DDSM/DDSM_Dataset')

from my_functions import *


## Chargement des images

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

## Création de séries temporelles (hétérogènes)

#Technique 1) Padding à 0

x_serie_padd_ddsm = build_serie_hetero_padding(x_serie_ddsm)
x_serie_padd_mias = build_serie_hetero_padding(x_serie_mias)
x_serie_padd_inbreast = build_serie_hetero_padding(x_serie_inbreast)

# Technique 2) Suppression de données au sein des séries

x_serie_deleted_ddsm = build_serie_hetero_delete(x_serie_ddsm)
x_serie_deleted_mias = build_serie_hetero_delete(x_serie_mias)
x_serie_deleted_inbreast = build_serie_hetero_delete(x_serie_inbreast)


for i,serie in enumerate(x_serie_deleted_ddsm): # Afficher le nombre de mammographies au sein de chaque série

    print('Taille de la série '+str(i)+' : '+str(len(serie)))


# Technique 3) Création de 5 tenseurs chacun hétérogène, prenant un nombre constant d'images par série.
# list_series.shape : [(50,1,227,227) / (50,2,227,227) / (50,3,227,227) / (50,4,227,227) / (50,5,227,227)]

x_serie_list_ddsm, y_serie_list_ddsm = build_serie_hetero_list(x_ddsm, nb_serie = 50)
x_serie_list_mias, y_serie_list_mias = build_serie_hetero_list(x_mias, nb_serie = 50)
x_serie_list_inbreast, y_serie_list_inbreast = build_serie_hetero_list(x_inbreast, nb_serie = 50)


## Chargement inputs en format tenseurs et formatage outputs


# Méthode 1) padding
x_serie_padd_ddsm_train, x_serie_padd_ddsm_test, y_serie_padd_ddsm_train, y_serie_padd_ddsm_test = build_train_test_serie(x_serie_padd_ddsm, y_serie_ddsm)
x_serie_padd_mias_train, x_serie_padd_mias_test, y_serie_padd_mias_train, y_serie_padd_mias_test = build_train_test_serie(x_serie_padd_mias, y_serie_mias)
x_serie_padd_inbreast_train, x_serie_padd_inbreast_test, y_serie_padd_inbreast_train, y_serie_padd_inbreast_test = build_train_test_serie(x_serie_inbreast, y_serie_padd_inbreast)

# Méthode 2) délétion
''' # Ne fonctionne pas : un tenseur ne peut pas être hétérogène
x_serie_deleted_ddsm_train, x_serie_deleted_ddsm_test, y_serie_deleted_ddsm_train, y_serie_deleted_ddsm_test = build_train_test_serie(x_serie_deleted_ddsm, y_serie_ddsm)
x_serie_deleted_mias_train, x_serie_deleted_mias_test, y_serie_deleted_mias_train, y_serie_deleted_mias_test = build_train_test_serie(x_serie_deleted_mias, y_serie_mias)
x_serie_deleted_inbreast_train, x_serie_deleted_inbreast_test, y_serie_deleted_inbreast_train, y_serie_deleted_inbreast_test = build_train_test_serie(x_serie_inbreast, y_serie_deleted_inbreast)
'''

# Méthode 3) sous-listes
nb_serie = 50

keep = np.random.choice([True,False],nb_serie, p = [0.8,0.2])
id_train = np.arange(0, nb_serie)[keep]
id_test = np.arange(0, nb_serie)[keep == False]

x_serie_list_ddsm_train = [x_serie_list_ddsm[0][id_train], x_serie_list_ddsm[1][id_train],x_serie_list_ddsm[2][id_train],x_serie_list_ddsm[3][id_train],x_serie_list_ddsm[4][id_train]]

x_serie_list_ddsm_test = [x_serie_list_ddsm[0][id_test], x_serie_list_ddsm[1][id_test],x_serie_list_ddsm[2][id_test],x_serie_list_ddsm[3][id_test],x_serie_list_ddsm[4][id_test]]

le = preprocessing.LabelEncoder()

y_serie_list_ddsm_train = [tf.keras.utils.to_categorical((le.fit_transform(y_serie_list_ddsm[0][id_train])),num_classes = 2, dtype = 'float32'),
                           tf.keras.utils.to_categorical((le.fit_transform(y_serie_list_ddsm[1][id_train])),num_classes = 2, dtype = 'float32'),
                           tf.keras.utils.to_categorical((le.fit_transform(y_serie_list_ddsm[2][id_train])),num_classes = 2, dtype = 'float32'),
                           tf.keras.utils.to_categorical((le.fit_transform(y_serie_list_ddsm[3][id_train])),num_classes = 2, dtype = 'float32'),
                           tf.keras.utils.to_categorical((le.fit_transform(y_serie_list_ddsm[4][id_train])),num_classes = 2, dtype = 'float32')]

y_serie_list_ddsm_test = [tf.keras.utils.to_categorical((le.fit_transform(y_serie_list_ddsm[0][id_test])),num_classes = 2, dtype = 'float32'),
                          tf.keras.utils.to_categorical((le.fit_transform(y_serie_list_ddsm[1][id_test])),num_classes = 2, dtype = 'float32'),
                          tf.keras.utils.to_categorical((le.fit_transform(y_serie_list_ddsm[2][id_test])),num_classes = 2, dtype = 'float32'),
                          tf.keras.utils.to_categorical((le.fit_transform(y_serie_list_ddsm[3][id_test])),num_classes = 2, dtype = 'float32'),
                          tf.keras.utils.to_categorical((le.fit_transform(y_serie_list_ddsm[4][id_test])),num_classes = 2, dtype = 'float32')]



## Construction du modèle

inputs = tf.keras.Input(shape = (None,227,227,1))

# Masking des valeurs à 0 (pas nécessaire ? --> précision 90% sans masking sur 2000 mammos)

#inputs = layers.TimeDistributed(tf.keras.layers.Masking(mask_value = 0, input_shape = (None, 5,227,227)))(inputs)     Conv2D ne supporterait pas le masking ?

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



# Fit pour technique 1) : padding à 0

history_model = model.fit(x_serie_padd_ddsm_train,y_serie_padd_ddsm_train,epochs = 8)
model.evaluate(x_serie_padd_ddsm_test,y_serie_padd_ddsm_test)


# Fit pour technique 3) : séries de taille différente dans différents tenseurs

model3 = tf.keras.models.clone_model(model)
model3.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy','AUC'])


nb_epochs = 100
for epoch in range(nb_epochs):

    serie = rd.randint(0,4)
    print("Entraînement n°" + str(epoch+1) + " avec la série de " + str(serie+1) + " images.")
    history_model3 = model3.fit(list_serie_train[serie],list_diagnostic_train[serie],epochs = 5)

model3.evaluate(list_serie_test[0], list_diagnostic_test[0])
model3.evaluate(list_serie_test[1], list_diagnostic_test[1])
model3.evaluate(list_serie_test[2], list_diagnostic_test[2])
model3.evaluate(list_serie_test[3], list_diagnostic_test[3])
model3.evaluate(list_serie_test[4], list_diagnostic_test[4])




pred = model3(list_serie_test[4])
pred = np.array(tf.argmax(pred,axis = 1))

labels=tf.argmax(list_diagnostic_test[4],axis = 1)


confusion_matrix = tf.math.confusion_matrix(labels=labels, predictions=pred, num_classes = 2 )

print(confusion_matrix)

confusion_matrix = np.array(confusion_matrix)

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay(confusion_matrix).plot(cmap = 'Blues')

plt.show()





from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

pred = model(x_padd_test)
pred = np.array(tf.argmax(pred,axis = 1))

labels=tf.argmax(y_test,axis = 1)


confusion_matrix = tf.math.confusion_matrix(labels=tf.argmax(y_test,axis = 1), predictions=pred, num_classes = 2 )

print(confusion_matrix)

confusion_matrix = np.array(confusion_matrix)



ConfusionMatrixDisplay(confusion_matrix).plot(cmap = 'Blues')

plt.show()




















