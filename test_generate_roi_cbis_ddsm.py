import pandas as pd
import os
from tqdm import tqdm
from colorama import Fore
import cv2
import re
import shutil


def get_patient_ids(x):
    return 'P_'+x.split('_')[2]

def get_image_paths(x):
    return x.replace('CBIS-DDSM/', 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset/')

def find_contours(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50,200)
    contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse= True)
    #print(sorted_contours)
    x=0 
    y=0 
    w=0 
    h=0
    X = []
    Y = []
    for (i,c) in enumerate(sorted_contours):
        x,y,w,h= cv2.boundingRect(c)
        X.append(x)
        Y.append(y)

    if w < 10 or h < 10:
        xmin = min(X)
        xmax = max(X)
        ymin = min(Y)
        ymax = max(Y)

        x = xmin
        y = ymin
        w = xmax-xmin
        h = ymax-ymin

    return x, y, w, h

info_file_path = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset/csv/dicom_info.csv'
info_file_path2 = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset/csv/calc_case_description_train_set.csv'
info_file_path3 = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset/csv/calc_case_description_test_set.csv'
info_file_path4 = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset/csv/mass_case_description_train_set.csv'
info_file_path5 = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset/csv/mass_case_description_test_set.csv'

jpg_database_path = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data'
roi_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/cbis_ddsm_rois/roi'
output_drawn_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/cbis_ddsm_rois/drawn'
output_labels_csv = 'C:/Users/rodri/Doutorado_2023/detect_lesions/cbis_ddsm_rois/labels.csv'

all_classes = ['BENIGN_MASS', 'BENIGN_CALCIFICATION', 'MALIGNANT_MASS', 'MALIGNANT_CALCIFICATION']

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

df = pd.read_csv(info_file_path, sep=",")
df2 = pd.read_csv(info_file_path2, sep=",")
df3 = pd.read_csv(info_file_path3, sep=",")
df4 = pd.read_csv(info_file_path4, sep=",")
df5 = pd.read_csv(info_file_path5, sep=",")
dfn = pd.concat([df2, df3, df4, df5], axis=0)

all_patient_ids = ['P_00390']

labels_df = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

for pid in tqdm(all_patient_ids, desc='Patients'):
    sub_df = df[df['PatientID'].str.contains(pid)]
    print('1')
    print(sub_df['PatientID'])
    sub_df_full_images = sub_df[sub_df["SeriesDescription"]=="full mammogram images"]
    print('2')
    print(sub_df_full_images['PatientID'])
    for index, row in sub_df_full_images.iterrows():
        full_image_path = row["image_path"]
        full_image_id = row["PatientID"]
        print('3')
        print(full_image_id)
        print(get_image_paths(full_image_path))
        full_image = cv2.imread(get_image_paths(full_image_path))
        sub_df_rois = sub_df[sub_df["SeriesDescription"]=="cropped images"]
        sub_df_rois = sub_df_rois[sub_df_rois["PatientID"].str.contains(full_image_id)]
        print('4')
        print(sub_df_rois['PatientID'])
        for index2, row2 in sub_df_rois.iterrows():
            orig_roi_image_path = row2["image_path"]
            roi_name = row2["PatientID"]
            print('5')
            print(roi_name)
            print(orig_roi_image_path)
            pathology = list(dfn[dfn["cropped image file path"].str.contains(roi_name)]["pathology"].values)[0]
            if 'BENIGN' in pathology:
                pathology = 'BENIGN'
            else:
                pathology = 'MALIGNANT'
            roi_class = pathology + '_' + list(dfn[dfn["cropped image file path"].str.contains(roi_name)]["abnormality type"].values)[0].upper()
            dest_roi_image_path = roi_path + '/' + roi_class + '/' + roi_name + '.jpg'
            #shutil.copy2(get_image_paths(orig_roi_image_path), dest_roi_image_path)
        print('6')
        sub_df_masks = sub_df[sub_df["SeriesDescription"]=="ROI mask images"]
        print(sub_df_masks)
        sub_df_masks = sub_df_masks[sub_df_masks["PatientID"].str.contains(full_image_id)]
        print(sub_df_masks)
        for index3, row3 in sub_df_masks.iterrows():
            mask_image_path = row3["image_path"]
            print('7')
            print(get_image_paths(mask_image_path))
            #print(get_image_paths(mask_image_path))
            image_mask = cv2.imread(get_image_paths(mask_image_path))
            x,y,w,h = find_contours(image_mask)
            #rint(x,y,w,h)
            img2 = full_image.copy()
            height, width = img2.shape[:2]
            cv2.rectangle(img2, (x,y), (x+w,y+h), (0, 255, 0))
            new_row = {'filename': full_image_id + '.jpg', 'width': width, 'height': height, 'class': roi_class.split('_')[1], 'xmin': x, 'ymin': y, 'xmax':x+w, 'ymax': y+h}
            labels_df = labels_df.append(new_row, ignore_index=True)

        #cv2.imwrite(output_drawn_path + '/' + full_image_id + '.jpg', img2)
        #labels_df.to_csv(output_labels_csv, index=False)
