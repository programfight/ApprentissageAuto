#PIL pour manipulation d'images
from PIL import Image
from PIL import ImageStat

#Sklearn pour l'apprentissage
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

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
# NB Gaussien sur histogramme simple
"""
Naive Bayes Gaussien (NBG) qui regarde simplement l'histogramme de chaque image.
On a donc 768 attributs (3 * 256) par image. Les variations de taille entre les images
rendent cette technique quasiment aléatoire.
[ SCORE ~= 58% ]
"""

"""
Fonction d'apprentissage pour NBG avec histogramme.
Renvoie le classifieur, son nom, ses paramètres et le temps d'apprentissage dans un tuple.
"""
def NBGH_fit(raw_data, labels):
    data    = []
    target  = []

    for image, label in zip(raw_data, labels):
        h = image.histogram()
        data.append(h)
        target.append(label)

    # TRAINING
    classifier = GaussianNB()
    tpsApp = util_fit(classifier, data, target)

    return (classifier, "NBG", "Histo Simple", tpsApp)

"""
Fonction de prédiction pour le NBGH.
"""
def NBGH_predict(classifier, image):
    data = image.histogram()
    return classifier.predict([data])[0]

####################################################################################################
# NB Gaussien sur histogramme normalisé
"""
Naive Bayes Gaussien (NBG) qui regarde l'histogramme de chaque image.
L'histogramme est réduit pour éviter le biais à cause
la taille d'une image.
[ SCORE ~= 52% ]
"""

"""
Fonction d'apprentissage pour NBG avec histogramme 'normalisé'.
Renvoie le classifieur, son nom, ses paramètres et le temps d'apprentissage dans un tuple.
"""
def NBGH_norm_fit(raw_data, labels):
    data    = []
    target  = []

    for image, label in zip(raw_data, labels):
        h = imgmanip.histProfondeurCouleurReduite(image)
        data.append(h)
        target.append(label)

    # TRAINING
    classifier = GaussianNB()
    tpsApp = util_fit(classifier, data, target)

    return (classifier, "NBG", "Histogramme Normalisé", tpsApp)

"""
Fonction de prédiction pour le NBGH normalisé.
"""
def NBGH_norm_predict(classifier, image):
    data = imgmanip.histProfondeurCouleurReduite(image)
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

    # TRAINING
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
