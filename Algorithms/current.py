import Algorithms.NB_simple as NB_simple
import Algorithms.SVCs as SVC

"""
Permet de diminuer la réécriture quand on veut changer d'algorithme à utiliser.
Pour changer l'algorithme actuel, modifier le corps de ces 2 fonctions :
"""
def current_algorithm_fit(raw_data, labels):
	return SVC.SVC_fit(raw_data, labels)

def current_algorithm_pred(classifier, image):
	return SVC.SVC_predict(classifier, image)

"""
Permet de diminuer la réécriture pour la cross-validation :
"""
def current_algorithm_cross_validate(raw_data, labels, nb_cv = 5):
	return SVC.SVC_cross_validate(raw_data, labels, nb_cv)

"""
Une boucle sur current_algorithm_pred qui retourne un tableau de résultats {-1, 1}.
"""
def current_algorithm_pred_array(classifier, images):
	results = []
	for image in images:
		results.append([image.filename, current_algorithm_pred(classifier, image)])
	return results
