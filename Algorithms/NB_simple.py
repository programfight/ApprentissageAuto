#PIL pour manipulation d'images
from PIL import Image
from PIL import ImageStat

#Sklearn pour l'apprentissage
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

#Util
from Algorithms.util import util_fit
from Algorithms.util import util_score

#Nos fonctions de manipulation d'images
import imgmanip

"""
Plusieurs algorithmes d'apprentissage basés sur Naive Bayes,
qui regardent des paramètres simples pour chaque image.
"""

####################################################################################################
# NB Gaussien sur histogramme normalisé
"""
Naive Bayes Gaussien (NBG) qui regarde l'histogramme de chaque image.
L'histogramme est réduit pour éviter le surplus de données.
[ SCORE ~= ? ]
"""

"""
Fonction d'apprentissage pour NBG avec histogramme 'réduit.
Renvoie le classifieur, son nom, ses paramètres et le temps d'apprentissage dans un tuple.
"""
def NBGH_red_fit(raw_data, labels):
    data    = []
    target  = []

    for image, label in zip(raw_data, labels):
        h = imgmanip.histogrammeReduit(image)
        data.append(h)
        target.append(label)

    classifier = GaussianNB()
    tpsApp = util_fit(classifier, data, target)

    return (classifier, "NBG", "Histogramme Réduit", tpsApp)

"""
Fonction de prédiction pour le NBGH normalisé.
"""
def NBGH_red_predict(classifier, image):
    data = imgmanip.histogrammeReduit(image)
    return classifier.predict([data])[0]

####################################################################################################
# NB Gaussien sur stats : moyenne, médiane, écart type
"""
Naive Bayes Gaussien (NBG) qui regarde les statistiques de chaque image
(moyenne, médiane, et écart type de chaque bande de l'image).
[ SCORE ~= 77% ]
"""

"""
Fonction d'apprentissage pour NBG sur stats.
Renvoie le classifieur, son nom, ses paramètres et le temps d'apprentissage dans un tuple.
"""
def NBGS_fit(raw_data, labels):
    data    = []
    target  = []

    for image, label in zip(raw_data, labels):
        image_stats = ImageStat.Stat(image)
        stats = image_stats.mean + image_stats.median + image_stats.stddev
        data.append(stats)
        target.append(label)

    classifier = GaussianNB()
    tpsApp = util_fit(classifier, data, target)

    return (classifier, "NBG", "Histo Normalisé", tpsApp)

"""
Fonction de prédiction pour le NBGS
"""
def NBGS_predict(classifier, image):
    image_stats = ImageStat.Stat(image)
    data = image_stats.mean + image_stats.median + image_stats.stddev
    return classifier.predict([data])[0]

"""
Fonction de Cross-Validation pour le NBGS
"""
def NBGS_cross_validate(raw_data, labels, nb_cv = 5):
    data    = []
    target  = []

    for image, label in zip(raw_data, labels):
        image_stats = ImageStat.Stat(image)
        stats = image_stats.mean + image_stats.median + image_stats.stddev
        data.append(stats)
        target.append(label)

    # TRAINING
    classifier = GaussianNB()
    CV_score = cross_val_score(classifier, data, target, cv = nb_cv)

    return CV_score
