#PIL pour manipulation d'images
from PIL import Image
from PIL import ImageStat

#Sklearn pour l'apprentissage
from sklearn.model_selection import train_test_split as tts
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
[ 30 < SCORE < 60 ]
"""

"""
Fonction d'apprentissage pour NBG avec histogramme.
Renvoie le classifieur, son nom, ses paramètres et le temps d'apprentissage dans un tuple.
"""
def NBGH_fit(images_class):
    data    = []
    target  = []

    for im in images_class:
        h = im[0].histogram()
        data.append(h)
        target.append(im[1])

    # TRAINING
    classifier = GaussianNB()
    tpsApp = util_fit(classifier, data, target)

    return (classifier, "NBG", "Histo Simple", tpsApp)

"""
Fonction de prédiction pour le NBGH.
"""
def NBGH_predict(classifier, image):
    data = image.histogram()
    return classifier.predict([data])

####################################################################################################
# NB Gaussien sur histogramme normalisé
"""
Naive Bayes Gaussien (NBG) qui regarde l'histogramme de chaque image.
L'histogramme est réduit pour éviter le biais à cause
la taille d'une image.
[ ? < SCORE < ? ]
"""

"""
Fonction d'apprentissage pour NBG avec histogramme 'normalisé'.
Renvoie le classifieur, son nom, ses paramètres et le temps d'apprentissage dans un tuple.
"""
def NBGH_norm_fit(images_class):
    data    = []
    target  = []

    for im in images_class:
        h = imgmanip.histProfondeurCouleurReduite(im[0])
        data.append(h)
        target.append(im[1])

    # TRAINING
    classifier = GaussianNB()
    tpsApp = util_fit(classifier, data, target)

    return (classifier, "NBG", "Histo Normalisé", tpsApp)

"""
Fonction de prédiction pour le NBGH normalisé.
"""
def NBGH_norm_predict(classifier, image):
    data = imgmanip.histProfondeurCouleurReduite(image)
    return classifier.predict([data])

####################################################################################################
# NB Gaussien sur stats : moyenne, médiane, écart type
"""
Naive Bayes Gaussien (NBG) qui regarde les statistiques de chaque image
(moyenne, médiane, et écart type de chaque bande de l'image).
[ ? < SCORE < ? ]
"""

"""
Fonction d'apprentissage pour NBG sur stats.
Renvoie le classifieur, son nom, ses paramètres et le temps d'apprentissage dans un tuple.
"""
def NBGS_fit(images_class):
    data    = []
    target  = []

    for im in images_class:
        image_stats = ImageStat.Stat(im[0])
        stats = image_stats.mean + image_stats.median
        data.append(stats)
        target.append(im[1])

    # TRAINING
    classifier = GaussianNB()
    tpsApp = util_fit(classifier, data, target)

    return (classifier, "NBG", "Histo Normalisé", tpsApp)

"""
Fonction de prédiction pour le NBGS
"""
def NBGS_predict(classifier, image):
    image_stats = ImageStat.Stat(image)
    data = image_stats.mean + image_stats.median
    return classifier.predict([data])
