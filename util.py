#Sklearn pour l'apprentissage
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#Time
import time

"""
Fonctions utilitaires pour simplifier l'écriture des fonctions
d'apprentissage.
"""

"""
Fait apprendre les données 'data' à 'classifier' à l'aide des étiquettes 'target'.
Renvoie le temps que met l'algorithme à apprendre ces données ('tpsApp').
"""
def util_fit(classifier, data, target):
    tpsApp = time.time()
    classifier.fit(data, target)
    tpsApp = time.time() - tpsApp
    return tpsApp


"""
Calcule le score d'un classifieur 'classifier' sur des données 'test' en
se basant sur les étiquettes 'target'.
Renvoie un tuple contenant ce score et le temps passé à prédire les classes
des données ('tpsPred').
"""
def util_score(classifier, test, target):
    tpsPred = time.time()
    predictions = classifier.predict(test)
    score = accuracy_score(predictions,target)
    tpsPred = time.time() - tpsPred
    return (score, tpsPred)

