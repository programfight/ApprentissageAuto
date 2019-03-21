#PIL pour manipulation d'images
from PIL import Image
from PIL import ImageStat

#Sklearn pour l'apprentissage
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

#Chi2 Kernel
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Train Test Split
from sklearn.model_selection import train_test_split as tts

#Util
from Algorithms.util import util_fit
from Algorithms.util import util_score

#Preprocessing
from sklearn import preprocessing

#Nos fonctions de manipulation d'images
import imgmanip

"""

"""

####################################################################################################
# SVC kernel linéaire sur histogramme réduit
def SVC_fit(raw_data, labels):
    data    = []
    target  = []

    for image, label in zip(raw_data, labels):
        stats = imgmanip.histogrammeReduit(image)
        data.append(stats)
        target.append(label)

    data = preprocessing.scale(data)

    classifier = SVC(gamma = 'auto', kernel = 'linear')
    tpsApp = util_fit(classifier, data, target)

    return (classifier, "SVC", " ", tpsApp)

"""
Fonction de prédiction
"""
def SVC_predict(classifier, image):
    data = imgmanip.histogrammeReduit(image)
    return classifier.predict([data])[0]

"""
Fonction de Cross-Validation
"""
def SVC_cross_validate(raw_data, labels, nb_cv = 5):
    data    = []
    target  = []

    for image, label in zip(raw_data, labels):
        stats = imgmanip.hogimage(image)
        data.append(stats)
        target.append(label)

    classifier = SVC(gamma = 'scale', kernel = 'rbf', verbose = 0)
    CV_score = cross_val_score(classifier, data, target, cv = nb_cv, verbose = 2, n_jobs = -1)

    return CV_score
