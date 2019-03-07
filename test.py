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

D_mean = []
D_median = []
target = []

print("\n--> Loading MER ")

for path, dirs, files in os.walk('./Data/Mer'):
    for filename in files:
        im = Image.open(path + '/' + filename)
        im_stat = ImageStat.Stat(im)
        D_mean.append(im_stat.mean)
        D_median.append(im_stat.median)
        target.append(1)

print("\n--> Loading AILLEURS")

for path, dirs, files in os.walk('./Data/Ailleurs'):
    for filename in files:
        im = Image.open(path + '/' + filename)
        im_stat = ImageStat.Stat(im)
        D_mean.append(im_stat.mean)
        D_median.append(im_stat.median)
        target.append(-1)

print("\n|--- TRAINING BEGIN")

D_mean_train, D_mean_test, target_mean_train, target_mean_test = tts(D_mean, target, test_size = 0.20)
D_median_train, D_median_test, target_median_train, target_median_test = tts(D_median, target, test_size = 0.20)

C_mean = GaussianNB()
mean_prediction = C_mean.fit(D_mean_train, target_mean_train)

C_median = GaussianNB()
median_prediction = C_median.fit(D_median_train, target_median_train)

print(C_mean.predict(D_mean) == target)
print(C_median.predict(D_median) == target)
