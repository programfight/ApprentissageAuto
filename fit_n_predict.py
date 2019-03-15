#PIL pour manipulation d'images
from PIL import Image
from PIL import ImageStat

#NumPy cool
import numpy as np

#Gestion des dossiers/fichiers
import os

#Modèle de persistence pour enregistrer le classifieur
from joblib import dump, load

#Gestion de Logs
from output_stats import clear_file
from output_stats import add_stats

#Nos fonctions d'apprentissage
import Algorithms.NB_simple as algoNB

#Warnings
import warnings
warnings.filterwarnings("ignore")


"""
Fonction d'apprentissage. Prend un chemin en entrée, et peut éventuellement
écrire les stats d'apprentissage dans un fichier.cvs.
Enregistre toutes les images des sous-dossiers "Mer" et "Ailleurs"  dans images_classes
et leur attribue une classe {-1, 1} selon leur sous-dossier.
"""
def fit(train_path, print_stats = False):

    #Classes' paths
    path_mer = train_path + 'Mer'
    path_ailleurs = train_path + 'Ailleurs'

    #Clear/make stats output file
    if(print_stats):
        clear_file()

    images_class = []

    #Loading SEA class
    for path, dirs, files in os.walk(path_mer):
        for filename in files:
            im = Image.open(path + '/' + filename)
            images_class.append([im,1])

    #Loading ELSEWHERE class
    for path, dirs, files in os.walk(path_ailleurs):
        for filename in files:
            im = Image.open(path + '/' + filename)
            images_class.append([im,-1])

    """ /!\ Pour changer l'algorithme à utiliser, changer cette ligne : """
    classifier = algoNB.NBGS_fit(images_class)[0]

    #Sauvegarde pour utilisation
    dump(classifier, './Models/saved_classifier')

"""
Fonction de prédiction, prend en entrée un chemin et renvoit
la classe prédite pour chaque image dans ce dossier.
"""
def predict(predict_path, print_results = True, print_stats = False):
    classifier = load('./Models/saved_classifier')
    images_pred = []

    for path, dirs, files in os.walk(predict_path):
        for filename in files:
            image = Image.open(path + '/' + filename)
            """ /!\ Pour changer l'algorithme de prédiction, changer cette ligne : """
            pred = algoNB.NBGS_predict(classifier, image)
            images_pred.append([filename, pred[0]])

    if(print_results):
        for image_class in images_pred:
            print("{\""+str(image_class[0])+"\" : "+str(image_class[1])+"}")
