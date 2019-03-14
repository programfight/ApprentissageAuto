from PIL import Image
from PIL import ImageFilter
from PIL import ImageStat

from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

import numpy as np

import os

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

print("\nBEGIN TRAIN")

D_train, D_test, target_train, target_test = tts(D, target, test_size = 0.20)

C = GaussianNB()
C.fit(D_train, target_train)

print("\nEND TRAIN")

y_pred = C.predict(D_test)
print(accuracy_score(y_pred,target_test))
