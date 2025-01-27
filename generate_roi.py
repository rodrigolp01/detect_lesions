import pandas as pd
import os
from tqdm import tqdm
from colorama import Fore
import cv2
import numpy


info_file = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/mias_dataset/Info.txt'
png_database_path = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/mias_dataset'
mias_roi_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mias/roi'
output_drawn_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mias/roi'

df = pd.read_csv(info_file, sep=" ")

all_classes = df.CLASS.unique()

for cl in all_classes:
    cl_path = os.path.sep.join([mias_roi_path,cl])
    if not os.path.isdir( cl_path ):
        os.makedirs(cl_path)

#print(df.info())

df = df.loc[df['CLASS'] != 'NORM']

#print(all_classes)

for index, row in tqdm(df.iterrows(), desc='mias dataset', position=0, leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
    cl = row['CLASS']
    img_name = row['REFNUM'] + '.png'
    img_path = os.path.sep.join([png_database_path, img_name])
    try:
        x = int(row['X'])
        y = int(1024.0 - row['Y'])
        radius = int(row['RADIUS'])
        img=cv2.imread(img_path)
        bbox_x = x-radius
        bbox_y = y-radius
        #roi_cropped = img[bbox_x:bbox_x+2*radius, bbox_y:bbox_y+2*radius]
        roi_cropped = img[bbox_y:bbox_y+2*radius, bbox_x:bbox_x+2*radius]
        roi_cropped_path = os.path.sep.join([mias_roi_path,cl,img_name])
        cv2.imwrite(roi_cropped_path,roi_cropped)

        #cv2.rectangle(img2, (bbox_x,bbox_y), (bbox_x+2*radius,bbox_y+2*radius), (0, 255, 0))
    except:
        print('imagem ' + img_name + ' sem ROI!')

