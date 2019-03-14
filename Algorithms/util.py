#Sklearn pour l'apprentissage
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#Time
import time

"""
Fonctions utilitaires pour simplifier l'écriture des fonctions
d'apprentissage.
"""

def util_fit(classifier, data, target):
    tpsApp = time.time()
    classifier.fit(data, target)
    tpsApp = time.time() - tpsApp
    return tpsApp

def util_score(classifier, test, target):
    tpsPred = time.time()
    predictions = classifier.predict(test)
    score = accuracy_score(predictions,target)
    tpsPred = time.time() - tpsPred
    return (score, tpsPred)
