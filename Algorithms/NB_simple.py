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

"""
Plusieurs algorithmes d'apprentissage basés sur Naive Bayes,
qui regardent des paramètres simples pour chaque image.
"""

# NB Gaussien sur histogramme
"""
Naive Bayes Gaussien qui regarde simplement l'histogramme de chaque image.
On a donc 768 attributs (3 * 256) par image. Les variations de taille rendent
cette technique
"""
def NB_gaussien_histogramme(images_n_type, set_size = 0.20 ):
    data    = []
    target  = []

    for im in images_n_type:
        h = im[0].histogram()
        data.append(h)
        target.append(im[1])

    D_train, D_test, target_train, target_test = tts(data, target, test_size = set_size)

    # TRAIN
    classifier = GaussianNB()
    tpsApp = util_fit(classifier, D_train, target_train)

    #PREDICT
    score, tpsPred = util_score(classifier, D_test, target_test)

    return ("NBG Histogramme", "Simple", tpsApp, tpsPred, score)
