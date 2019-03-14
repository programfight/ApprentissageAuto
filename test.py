#PIL pour manipulation d'images
from PIL import Image
from PIL import ImageFilter
from PIL import ImageStat

#Sklearn pour l'apprentissage
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

#Time
import time

#NumPy cool
import numpy as np

#Gestion des dossiers/fichiers
import os

#stats
from output_stats import clear_file
from output_stats import add_stats

#clear/make stats output file
clear_file()

nbIter = int(input("Combien d'itérations ?\n"))

for i in range(nbIter):

    D = []
    target = []

    print("\n--> Loading MER ")

    for path, dirs, files in os.walk('./Data/Mer'):
        for filename in files:
            im = Image.open(path + '/' + filename)
            im_stat = ImageStat.Stat(im)
            D.append(im.histogram())
            target.append(1)

    print("\n--> Loading AILLEURS")

    for path, dirs, files in os.walk('./Data/Ailleurs'):
        for filename in files:
            im = Image.open(path + '/' + filename)
            im_stat = ImageStat.Stat(im)
            D.append(im.histogram())
            target.append(-1)

    # TRAIN
    print("\nBEGIN TRAIN")
    tTrain_begin = time.time()

    D_train, D_test, target_train, target_test = tts(D, target, test_size = 0.20)

    C = GaussianNB()
    C.fit(D_train, target_train)

    tTrain_time = time.time() - tTrain_begin
    print("\nEND TRAIN")

    #PREDICT

    print("\nBEGIN PREDICT")
    tPredict_begin = time.time()

    y_pred = C.predict(D_test)
    score = accuracy_score(y_pred,target_test)

    tPredict_time = time.time() - tPredict_begin
    print("\nEND TRAIN")


    add_stats("Naive Bayes", "Histogramme", tTrain_time, tPredict_time, score)
