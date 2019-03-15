from PIL import Image
from PIL import ImageFilter
from PIL import ImageStat
from PIL import ImageColor
from PIL import ImageDraw
import numpy as np

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
    for i in range(len(histo)):
        if i%85 == 0:
            hred.append(histo[i])
    return hred
