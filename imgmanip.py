from PIL import Image
from PIL import ImageFilter
from PIL import ImageStat
from PIL import ImageColor
from PIL import ImageDraw
import numpy as np

from skimage.feature import hog

import os

"""
La fonction histProfondeurCouleurReduite prend une image im et la
resize en 150*150 pixels et réduit sa profondeur
de couleurs à 4 valeurs par couleurs (liste de 12 éléments)
"""
def histProfondeurCouleurReduite(im):
    im = im.point(lambda x: int(x/85)*85) #ou 51
    im = im.resize([150,150])
    histo = im.histogram()
    hred = []
    hred.append(histo[0])
    hred.append(histo[85])
    hred.append(histo[170])
    hred.append(histo[255])
    hred.append(histo[256])
    hred.append(histo[341])
    hred.append(histo[426])
    hred.append(histo[511])
    hred.append(histo[512])
    hred.append(histo[597])
    hred.append(histo[682])
    hred.append(histo[767])
	return hred

def extractStats(image):
    image_stats = ImageStat.Stat(image)
    stats = image_stats.mean + image_stats.stddev + image_stats.median
    return stats

"""
on peu tl' itlsuer
"""
def histogrammeReduit(image):
    histo = np.array(image.histogram());
    h_reduit = (histo - np.amax(histo))/(np.amax(histo) - np.amin(histo))
    return h_reduit.tolist()

"""
La procedure generateNewDataFromExisting genere des nouvelles images en rotationnant les images de -rotaterange à rotaterange
"""
def generateNewDataFromExisting(data,target):
    rotaterange = 4
    lenD = len(data)
    for i in range(lenD):
        #print(str(i)+"/"+str(lenD-1))
        im = data[i]
        for ii in range(-rotaterange,rotaterange):
            if ii==0: 
                continue
            data.append(im.rotate(ii))
			target.append(target[i])


def hogimage(image):
    image = image.resize([250,250])
    _, h = hog(image, orientations=8, pixels_per_cell=(16, 16),
    cells_per_block=(1, 1), visualize=True, multichannel=True)
	return h.flatten()
