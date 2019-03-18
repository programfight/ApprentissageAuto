import Algorithms.NB_simple as NB_simple

"""
Permet de diminuer la réécriture quand on veut changer d'algorithme à utiliser.
Pour changer l'algorithme actuel, modifier le corps de ces 2 fonctions :
"""
def current_algorithm_fit(images_class):
	return NB_simple.NBGS_fit(images_class)
	
def current_algorithm_pred(classifier, image):
	return NB_simple.NBGS_predict(classifier, image)