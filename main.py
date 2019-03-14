#PIL pour manipulation d'images
from PIL import Image
from PIL import ImageStat

#NumPy cool
import numpy as np

#Gestion des dossiers/fichiers
import os

#Gestion de Logs
from output_stats import clear_file
from output_stats import add_stats

#Nos fonctions d'apprentissage
from Algorithms.NB_simple import NB_gaussien_histogramme

#Train Data paths
train_path = './Data/'
path_mer = train_path + 'Mer'
path_ailleurs = train_path + 'Ailleurs'

#clear/make stats output file
clear_file()

nbIter = int(input("Combien d'itérations ?\n"))

for i in range(nbIter):

    images_n_type = []

    print("\n--> Loading MER ")

    for path, dirs, files in os.walk(path_mer):
        for filename in files:
            im = Image.open(path + '/' + filename)
            images_n_type.append([im,1])

    print("\n--> Loading AILLEURS")

    for path, dirs, files in os.walk(path_ailleurs):
        for filename in files:
            im = Image.open(path + '/' + filename)
            images_n_type.append([im,-1])

    results = NB_gaussien_histogramme(images_n_type)

    add_stats(*results)
