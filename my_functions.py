import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import random as rd

from PIL import Image

import os

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import preprocessing

'''
# Récupère le nom de la variable. Get string from variable name
def get_var_name(var):

    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return ([var_name for var_name, var_val in callers_local_vars if var_val is var])


def foo(bar):
    return get_var_name(bar)
'''

## Data relative functions


# Importe des images du jeu de données renseigné. Le dossier indiqué doit contenir deux sous dossiers : bening et malignant.
def import_data(dataset, nb_images = 500, path = '/Users/rp2/Documents/Stage_3A/Données/DDSM/'):


    os.chdir(path + dataset)

    datasets_name = os.listdir()  # Benign et Malignant

    path = path + dataset


    x = []  # Liste stockant les données


    for case_dataset in datasets_name:   # Chargement données dans X. X de taille 2. Premier élement contient n images du dossier bening. Le deuxième contient n images de malignant.

        new_path = path + '/' + case_dataset

        data = os.listdir(case_dataset)
        data = np.array(data)

        if nb_images > len(data):
            print("Plus d'images demandées que d'images dans le dataset. Demandez moins d'images")
            return()

        a_retenir = np.arange(0,len(data))
        a_retenir = np.random.choice(a_retenir,nb_images,replace = False)

        data = data[a_retenir]

        list_images = []

        c = 1
        for image in data:
            img = Image.open(path+'/'+case_dataset+'/'+image)
            img = np.asarray(img)
            list_images.append(img)
            print("Chargement image n°"+str(c))
            c += 1




        x.append(list_images)

    return x    # x de taille 2 : n images bening et n images malignant.


# Construit des séries temporelles de données à partir d'un jeu de données de même forme (2,nb_images)
def build_temporal_series(data, nb_par_serie = 5, labels = ['Benign','Malignant']):

    nb_benign = nb_malignant = len(data[0])//5

    X = []
    Y = []

    for dataset in range(len(data)):
        for i in range(nb_benign):
            X.append(data[dataset][(i*5):((i*5)+nb_par_serie)])
            Y.append(labels[dataset])

    return np.array(X), np.array(Y)

# Créé l'hétérogénéité au sein des séries. Méthode 1 = padding
def build_serie_hetero_padding(data, taux_absence = 0.2):

    data_padd = np.zeros_like(data)

    for num_serie in range(len(data)):
        for num_mammo in range(len(data[num_serie])):

            if rd.random() < 1- taux_absence:
                data_padd[num_serie][num_mammo] = data[num_serie][num_mammo]

    return data_padd

# Créé l'hétérogénéité au sein des séries. Méthode 2 = suppression aléatoire de mammographies
def build_serie_hetero_delete(data, taux_absence = 0.2):

    data_het = []

    for num_serie in range(len(data)):
        data_het.append([])

        for num_mammo in range(len(data[num_serie])):

            if rd.random() < 1 - taux_absence :

                data_het[num_serie].append(data[num_serie][num_mammo])


    return np.array(data_het,dtype = 'object')

# Créé l'hétérogénéité au sein des séries. Méthode 3 = création de 5 sous-listes, chacune comportant n images par série.
def build_serie_hetero_list(data, nb_serie = 50, taille_serie = [1,2,3,4,5]): # taille_serie = liste contenant le nombre d'images par série.

    list_series = []    # Liste contenant les tenseurs homogènes.
    list_diagnostic = []

    data = np.array(data)   # shape = (2, nb_images, width, height) avec premier élément les mammos benign, et en deuxième les mammos malignant.

    for size in taille_serie:
        mammos = []
        diagnostic = []

        for num_serie in range(nb_serie):

            if rd.random() < 0.5: # Alors série bening
                keep = np.random.choice(np.arange(0,len(data[0])),size, replace = False)
                mammos.append(data[0][keep])
                diagnostic.append("Benign")

            else: # Sinon série cancer
                keep = np.random.choice(np.arange(0,len(data[1])),size, replace = False)
                mammos.append(data[1][keep])
                diagnostic.append("Malignant")

        list_series.append(np.array(mammos))
        list_diagnostic.append(np.array(diagnostic))

    return tf.convert_to_tensor(list_series), list_diagnostic



# Construit x_train, x_test, y_train et y_test à partir d'un jeu de données (toujours orgganisé pareil : (2,n) avec chacune des sous listes contenant n images.
def build_train_test_data(data, labels = ['Benign','Malignant'], test_size = 0.2):

    X = []
    Y = []

    for dataset in range(len(data)):    # X : inputs / Y : Labels
        for img in data[dataset]:
            X.append(img)
            Y.append(labels[dataset])


    le = preprocessing.LabelEncoder()   # Encodage de la liste Y : passage de string à nombre, et de nombres à one-hot-encoder.
    Y = le.fit_transform(Y)
    Y = tf.keras.utils.to_categorical(Y, num_classes = 2, dtype = 'float32')  # num_classes = 2 : benign et malignant



    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = test_size)

    X_train = tf.convert_to_tensor(X_train)  # Passage de np.array à tenseurs
    X_test = tf.convert_to_tensor(X_test)


    return X_train, X_test, Y_train, Y_test


def build_train_test_serie(data_x, data_y, taux_train = 0.8):  # Sépare le jeu de données de forme série en train et test.

    le = preprocessing.LabelEncoder()
    data_y = le.fit_transform(data_y)
    data_y = tf.keras.utils.to_categorical(data_y, num_classes = 2, dtype = 'float32')   # Convertir le vecteur label de string à nombres.


    id_train = []
    id_test = []


    for i in range(len(data_x)):    # Sélectionne les id des futurs jeux de test et train
        if rd.random() < taux_train:
            id_train.append(i)
        else:
            id_test.append(i)

    x_train = data_x[id_train]
    x_test = data_x[id_test]
    y_train = data_y[id_train]
    y_test = data_y[id_test]

    x_train = tf.convert_to_tensor(x_train)  # Conversion en tenseurs
    x_test = tf.convert_to_tensor(x_test)

    return x_train, x_test, y_train, y_test



## Image relative functions


# Prend en entrée un jeu de données d'images grayscale de shape (nb_images, taille_image_taille_image) et renvoie un jeu de données de même taille, avec des images RGB
def make_rgb_from_grayscale(data):
    datargb = []
    for i in range(len(data)):
        img = np.stack((data[i],)*3,axis = -1)
        datargb.append(img)

    return(tf.convert_to_tensor(datargb))



## Plot relative functions


# Affiche un échantillon d'images provenant de data. Data doit être de forme (2,n) avec n étant le nombre d'images dans chacun des deux sous listes de data (benign et malignant).
def plot_subset_images(data, rows, cols, title = '', cmap = 'viridis'):


    im = np.random.choice(np.arange(0,len(data[0])),rows*cols,replace = False)

    labels = ['Benign','Malignant']

    fig = plt.figure()

    for i in range(rows*cols):

        a = rd.randint(0,1)

        image = data[a][im[i]]
        fig.add_subplot(rows,cols,i+1)
        plt.imshow(image, cmap = 'viridis')



        plt.title(labels[a])
        fig.suptitle(title)

    plt.show()


# Plot la matrice de confusion d'un modèle. x et y sont ce que l'on fournit au modèle (inputs et labels attendus)
def plot_confusion_matrix(x, y, model):

    pred = model(x)  # Récupère les prédictions du modèle
    pred = np.array(tf.argmax(pred,axis = 1))  # Récupère l'argmax (classe prédite)

    confusion_matrix = tf.math.confusion_matrix(labels=tf.argmax(y,axis = 1), predictions=pred, num_classes = 2 ) # Construit la matrice de confusion
    confusion_matrix = np.array(confusion_matrix)

    ConfusionMatrixDisplay(confusion_matrix).plot(cmap = 'Blues')

    accuracy = (confusion_matrix[0,0] + confusion_matrix[1,1]) / np.sum(confusion_matrix)

    plt.title("Précision = " + str(round(accuracy,2)))

    plt.show()

    #return confusion_matrix









