#PIL pour manipulation d'images
from PIL import Image
from PIL import ImageStat

#Score pour calc_score
from sklearn.metrics import accuracy_score

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
    classifier = current_algorithm_fit(images_class)[0]

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
            pred = current_algorithm_pred(classifier, image)
            images_pred.append([filename, pred[0]])

    if(print_results):
        for image_class in images_pred:
            print("{\""+str(image_class[0])+"\" : "+str(image_class[1])+"}")

"""
Fonction de calcul de score, ne peut être utilisée
que sur un dossier qui possède 2 sous-dossiers "Mer" et "Ailleurs".
"""
def calc_score(score_path, print_results = True, print_stats = False):
	classifier = load('./Models/saved_classifier')
	
	path_mer = score_path + 'Mer'
	path_ailleurs = score_path + 'Ailleurs'

	#The result of our prediction algorithm
	results = []
	results_mer = []
	results_ailleurs = []
	
	#The target datasets
	target = []
	target_mer = []
	target_ailleurs = []
	
	#Loading SEA class
	for path, dirs, files in os.walk(path_mer):
		for filename in files:
			image = Image.open(path + '/' + filename)
			""" /!\ Pour changer l'algorithme de prédiction, changer cette ligne : """
			pred = current_algorithm_pred(classifier, image)
			results.append(pred[0])
			results_mer.append(pred[0])
			target.append(1)
			target_mer.append(1)

	#Loading ELSEWHERE class
	for path, dirs, files in os.walk(path_ailleurs):
		for filename in files:
			image = Image.open(path + '/' + filename)
			""" /!\ Pour changer l'algorithme de prédiction, changer cette ligne : """
			pred = current_algorithm_pred(classifier, image)
			results.append(pred[0])
			results_ailleurs.append(pred[0])
			target.append(-1)
			target_ailleurs.append(-1)

	if(print_results):
		print("------------------------------------")
		print("> Score Total :    "+str(accuracy_score(target, results)))
		print("------------------------------------")
		print("  > Score Mer :      "+str(accuracy_score(target_mer, results_mer)))
		print("  > Score Ailleurs : "+str(accuracy_score(target_ailleurs, results_ailleurs)))
