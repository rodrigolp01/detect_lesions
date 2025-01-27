import pandas as pd
import os
from tqdm import tqdm
from colorama import Fore
import cv2
import re
import shutil
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


test_mask = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mini_ddsm/MINI-DDSM-Complete-PNG-16/Cancer/0001/C_0001_1.RIGHT_CC_Mask.png'
image = cv2.imread(test_mask)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50,200)
#plt.imshow(gray)
#plt.show()
#contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=cv2.contourArea, reverse= True)
coods = []
X = []
Y = []
for (i,c) in enumerate(sorted_contours):
    x,y,w,h= cv2.boundingRect(c)
    X.append(x)
    Y.append(y)
    #coods.append((x,y))

xmin = min(X)
xmax = max(X)
ymin = min(Y)
ymax = max(Y)

print([xmin, ymin, xmax-xmin, ymax-ymin])

#poly1 = Polygon(coods)
#print(poly1.centroid)
    