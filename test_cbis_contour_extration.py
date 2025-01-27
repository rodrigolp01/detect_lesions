import pandas as pd
import os
from tqdm import tqdm
from colorama import Fore
import cv2
import re
import shutil
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


test_mask = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset/jpeg/1.3.6.1.4.1.9590.100.1.2.346123264111128140504983520282211141487/1-264.jpg'
image = cv2.imread(test_mask)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50,200)
#plt.imshow(gray)
#plt.show()
contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=cv2.contourArea, reverse= True)
coords = []
for (i,c) in enumerate(sorted_contours):
    x,y,w,h= cv2.boundingRect(c)
    coords = [x,y,w,h]

print(coords)

X = []
Y = []
for (i,c) in enumerate(sorted_contours):
    x,y,w,h= cv2.boundingRect(c)
    X.append(x)
    Y.append(y)

xmin = min(X)
xmax = max(X)
ymin = min(Y)
ymax = max(Y)

print([xmin, ymin, xmax-xmin, ymax-ymin])