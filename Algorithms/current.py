import Algorithms.NB_simple as NB_simple

"""
Permet de diminuer la réécriture quand on veut changer d'algorithme à utiliser.
Pour changer l'algorithme actuel, modifier le corps de ces 2 fonctions :
"""
def current_algorithm_fit(raw_data, target):
	return NB_simple.NBGS_fit(raw_data, target)

def current_algorithm_pred(classifier, image):
	return NB_simple.NBGS_predict(classifier, image)

"""
Une boucle sur current_algorithm_pred qui retourne un tableau de résultats {-1, 1}.
"""
def current_algorithm_pred_array(classifier, images):
	results = []
	for image in images:
		results.append([image.filename, current_algorithm_pred(classifier, image)])
	return results
