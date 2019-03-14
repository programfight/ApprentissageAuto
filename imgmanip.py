from PIL import Image
from PIL import ImageFilter
from PIL import ImageStat
from PIL import ImageColor
from PIL import ImageDraw
import numpy as np

import os


#histProfondeurCouleurReduite prend une image im et la resize en 150*150 pixel et réduit sa profondeur
#de couleurs a 4 valeurs par couleurs (liste de 12 elements)
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
