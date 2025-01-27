import pandas as pd
import os
from tqdm import tqdm
from colorama import Fore
import cv2
import re
import shutil


def get_maskes(folder_path, imagname):
    img_masks = []
    #print(imagname)
    for img in os.listdir(folder_path):
        #print('1 ', img)
        if 'Mask' in img or 'MASK' in img:
            #print('2 ', img)
            file_name, file_extension = os.path.splitext(imagname)
            if file_name == '_'.join(img.split('_')[0:-1]):
                #print('3')
                img_masks.append(img)
    return img_masks

def get_patients(x):
    return x.split('_')[1]

def find_contours(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50,200)
    contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse= True)
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
    return xmin, ymin, xmax-xmin, ymax-ymin

def parse_overlay_file(img_path, image_mask, status):
    overlay_file = '_'.join(image_mask.split('_')[0:-1]) + '.OVERLAY'
    #print(overlay_file)
    mask_prefix = image_mask.split('_')[-1].split('.')[0]
    mask_idx = re.findall(r'\d+', mask_prefix)
    #print(mask_idx)
    lesion_type = ''
    if mask_idx:
        abnormality_idx = mask_idx[0]
    else:
        abnormality_idx = 1
    f = open(img_path + '/' + overlay_file, 'r')
    lines = f.readlines()
    #print(lines)
    for i in range(0, len(lines)):
        if 'ABNORMALITY ' + str(abnormality_idx) in lines[i]:
            line2 = lines[i+1]
            #print(line2)
            if 'MASS' in line2:
                lesion_type = 'MASS'
            else:
                lesion_type = 'CALCIFICATION'

    cropped_class = status.upper() + ' ' + lesion_type
    return cropped_class, abnormality_idx    


#http://www.learningaboutelectronics.com/Articles/How-to-crop-an-object-in-an-image-in-Python-OpenCV.php

info_file = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mini_ddsm/Data-MoreThanTwoMasks/Data-MoreThanTwoMasks.xlsx'
png_database_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mini_ddsm/MINI-DDSM-Complete-PNG-16'
roi_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mini_ddsm_rois/roi'
output_drawn_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mini_ddsm_rois/drawn'

df = pd.read_excel(info_file)
all_classes = ['BENIGN MASS', 'BENIGN CALCIFICATION', 'CANCER MASS', 'CANCER CALCIFICATION']

for cl in all_classes:
    cl_path = os.path.sep.join([roi_path,cl])
    if not os.path.isdir( cl_path ):
        os.makedirs(cl_path, exist_ok=True)
    else:
        shutil.rmtree(cl_path)
        os.makedirs(cl_path, exist_ok=True)

if not os.path.isdir( output_drawn_path ):
    os.makedirs(output_drawn_path, exist_ok=True)
else:
    shutil.rmtree(output_drawn_path)
    os.makedirs(output_drawn_path, exist_ok=True)

df = df.loc[df['Status'] != 'Normal']
#df = df.loc[df['Tumour_Contour'] != '-']
df['patient_id'] = df['fileName'].apply(get_patients)

patient_ids = list(set(df.patient_id.values.tolist()))
#print(patient_ids)

for pi in patient_ids:
    df_pi = df[df['patient_id'] == pi]
    #print(df_pi)
    img_path = png_database_path + '/' + df_pi['Status'].iloc[0] + '/' + pi
    #print(img_path)
    for imgname in os.listdir(img_path):
        file_name, file_extension = os.path.splitext(img_path + '/' + imgname)

        if (file_extension == '.png') and not ('Mask' in imgname or 'MASK' in imgname):
            img_maskes = get_maskes(img_path, imgname)
            #print(img_maskes)
            image = cv2.imread(img_path + '/' + imgname)
            img2 = image.copy()
            for img_mask in img_maskes:
                
                image_mask = cv2.imread(img_path + '/' + img_mask)
                x,y,w,h = find_contours(image_mask)
                print(img_mask)
                #x,y,w,h= cv2.boundingRect(contour)
                print(x,y,w,h)
                #breast_image_name = '_'.join(imgname.split('_')[0:-1]) + '.png'
                cropped_contour= image[y:y+h, x:x+w] #roi_cropped = img[bbox_y:bbox_y+2*radius, bbox_x:bbox_x+2*radius]
                cropped_class, index = parse_overlay_file(img_path, img_mask, df_pi['Status'].iloc[0])
                print(cropped_class)
                filename, _ = os.path.splitext(imgname)
                final_cropped_path = roi_path + '/' + cropped_class + '/' + filename + '_' + str(index) + '.png'
                print(final_cropped_path)
                cv2.imwrite(final_cropped_path, cropped_contour)

                cv2.rectangle(img2, (x,y), (x+w,y+h), (0, 255, 0))

            cv2.imwrite(output_drawn_path + '/' + imgname + '.png', img2)
        