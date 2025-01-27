import pandas as pd
import os
from tqdm import tqdm
from colorama import Fore
import cv2
#import numpy


info_file = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/mias_dataset/mias_Info.txt'
png_database_path = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/mias_dataset'
mias_roi_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mias/roi'
output_drawn_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mias/drawn'

df = pd.read_csv(info_file, sep=" ")

all_classes = df.CLASS.unique()

for cl in all_classes:
    cl_path = os.path.sep.join([mias_roi_path,cl])
    if not os.path.isdir( cl_path ):
        os.makedirs(cl_path)

if not os.path.isdir( output_drawn_path ):
    os.makedirs(output_drawn_path)

#print(df.info())

df = df.loc[df['CLASS'] != 'NORM']
df = df[df['X'].notna()]
img_names = df.REFNUM.values.tolist()

#print(all_classes)

for img_name in tqdm(img_names, desc='mias dataset', position=0, leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
    df_image = df.loc[df['REFNUM'] == img_name]
    img_path = os.path.sep.join([png_database_path, img_name + '.png'])
    img=cv2.imread(img_path)
    img2 = img.copy()
    counter = 0
    for index, row in df_image.iterrows():
        cl = row['CLASS']
        x = int(row['X'])
        y = int(1024.0 - row['Y'])
        radius = int(row['RADIUS'])
        bbox_x = x-radius
        bbox_y = y-radius
        roi_cropped = img[bbox_y:bbox_y+2*radius, bbox_x:bbox_x+2*radius]
        roi_cropped_path = os.path.sep.join([mias_roi_path, cl, img_name.split('.')[0]+'_'+str(counter)+'.png'])
        cv2.imwrite(roi_cropped_path,roi_cropped)
        counter += 1
        cv2.rectangle(img2, (bbox_x,bbox_y), (bbox_x+2*radius,bbox_y+2*radius), (0, 255, 0))
    cv2.imwrite(output_drawn_path + '/' + img_name + '.png', img2)