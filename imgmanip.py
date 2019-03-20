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

def histogrammeReduit(image):
    h = image.histogram();
    hred = h[0:len(h)//3]
    hgreen = h[len(h)//3:(len(h)//3)*2]
    hblue = h[(len(h)//3)*2:len(h)]

    m_red = np.amax(hred)
    m_green = np.amax(hgreen)
    m_blue = np.amax(hblue)

    hred = np.array(hred)
    hred = (hred*100)/m_red

    hgreen = np.array(hgreen)
    hgreen = (hgreen*100)/m_green

    hblue = np.array(hblue)
    hblue = (hblue*100)/m_blue


    h_reduit = hred
    h_reduit = np.append(h_reduit, hgreen)
    h_reduit = np.append(h_reduit, hblue)

    return h_reduit
