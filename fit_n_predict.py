#PIL pour manipulation d'images
from PIL import Image
from PIL import ImageStat

#Sklearn pour score et split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts

#Random
import random

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
from Algorithms.current import current_algorithm_fit
from Algorithms.current import current_algorithm_pred
from Algorithms.current import current_algorithm_pred_array
from Algorithms.current import current_algorithm_cross_validate

#Warnings
import warnings
warnings.filterwarnings("ignore")


####################################################################################################
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

    data = []
    target = []

    #Loading SEA class
    for path, dirs, files in os.walk(path_mer):
        for filename in files:
            im = Image.open(path + '/' + filename)
            data.append(im)
            target.append(1)

    #Loading ELSEWHERE class
    for path, dirs, files in os.walk(path_ailleurs):
        for filename in files:
            im = Image.open(path + '/' + filename)
            data.append(im)
            target.append(-1)

    classifier = current_algorithm_fit(data, target)[0]

    #Sauvegarde pour utilisation
    dump(classifier, './Models/saved_classifier')

####################################################################################################
"""
Fonction de prédiction, prend en entrée un chemin et renvoit
la classe prédite pour chaque image dans ce dossier.
"""
def predict(predict_path, print_results = True, print_stats = False):
    classifier = load('./Models/saved_classifier')
    images = []

    for path, dirs, files in os.walk(predict_path):
        for filename in files:
            image = Image.open(path + '/' + filename)
            images.append(image)

    prediction = current_algorithm_pred_array(classifier, images)

    if(print_results):
        for result in prediction:
            print("{\""+str(result[0])+"\" : "+str(result[1])+"}")

    return prediction

####################################################################################################
"""
Fonction de calcul de score, ne peut être utilisée
que sur un dossier qui possède 2 sous-dossiers "Mer" et "Ailleurs".
"""
def calc_score(score_path, scoreType, print_results = True, print_stats = False):
	classifier = load('./Models/saved_classifier')

	path_mer = score_path + 'Mer'
	path_ailleurs = score_path + 'Ailleurs'

	#The result of our prediction algorithm
	results = []

	#The target datasets
	target = []

	#Loading SEA class
	for path, dirs, files in os.walk(path_mer):
		for filename in files:
			image = Image.open(path + '/' + filename)
			pred = current_algorithm_pred(classifier, image)
			results.append(pred)
			target.append(1)

	#Loading ELSEWHERE class
	for path, dirs, files in os.walk(path_ailleurs):
		for filename in files:
			image = Image.open(path + '/' + filename)
			pred = current_algorithm_pred(classifier, image)
			results.append(pred)
			target.append(-1)

	if print_results:
            if scoreType == "accuracy":
                print("\n> Score Total : "+str(accuracy_score(target, results))+'\n')

####################################################################################################
"""
Fonction qui divise l'ensemble d'images en 2. Le classifieur va être entraîné sur le 1er ensemble obtenu
et effectuera ses prédicitons sur le 2ème. Affiche le score.
"""
def score_split(split_path, test_percent, scoreType,print_results = True, print_stats = False):

    #Classes' paths
    path_mer = split_path + 'Mer'
    path_ailleurs = split_path + 'Ailleurs'

    #Clear/make stats output file
    if(print_stats):
        clear_file()

    data = []
    target = []

    #Loading SEA class
    for path, dirs, files in os.walk(path_mer):
        for filename in files:
            im = Image.open(path + '/' + filename)
            data.append(im)
            target.append(1)

    #Loading ELSEWHERE class
    for path, dirs, files in os.walk(path_ailleurs):
        for filename in files:
            im = Image.open(path + '/' + filename)
            data.append(im)
            target.append(-1)

    if print_results:
            if scoreType == "accuracy":
                data_train, data_test, target_train, target_test = tts(data, target, test_size = test_percent)
                results = []
                classifier = current_algorithm_fit(data_train, target_train)[0]
                results = [element[1] for element in current_algorithm_pred_array(classifier, data_test)]
                print("\n> Accuracy Score : "+str(accuracy_score(target_test, results))+'\n')
            if scoreType == "cross-validation":
                scores = current_algorithm_cross_validate(data, target, 5)
                print("\n> Cross-Validation : \n\t> Accuracy : "+str(scores.mean())+"\n\t> StdDev : "+str(scores.std() * 2))
