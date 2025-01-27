import os
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
from colorama import Fore
import glob
import pandas as pd
import cv2
import numpy as np
from patchify import patchify

#https://levelup.gitconnected.com/how-to-split-an-image-into-patches-with-python-e1cf42cf4f77
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.extract_patches_2d.html
#https://www.geeksforgeeks.org/extracting-patches-from-large-images-using-python/

birads_dict = {'normal': ['1'], 'benign': ['2','3'], 'malignant':['4a','4b','4c','5','6']}
class2int = {'normal': '0', 'benign': '1', 'malignant':'2'}
severity2class = {'': '0', 'B': '1', 'M':'2'}
pathology2class = {'BENIGN': '1', 'BENIGN_WITHOUT_CALLBACK': '1', 'MALIGNANT': '2'}
status2class = {'Normal': '0', 'Benign': '1', 'Cancer': '2'}
#background, malignant mass, benign mass, malignant calcification and benign calcification


def split_datasets(dataset, dataset_path, test_size):
    all_patient_ids = []
    case_dict = {}
    file_dict = {}
    if dataset == 'cbis_ddsm':
        info_file_path = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset/csv/dicom_info.csv'
        info_file_path2 = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset/csv/calc_case_description_train_set.csv'
        info_file_path3 = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset/csv/calc_case_description_test_set.csv'
        info_file_path4 = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset/csv/mass_case_description_train_set.csv'
        info_file_path5 = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset/csv/mass_case_description_test_set.csv'
        df = pd.read_csv(info_file_path, sep=",")
        df2 = pd.read_csv(info_file_path2, sep=",")
        df3 = pd.read_csv(info_file_path3, sep=",")
        df4 = pd.read_csv(info_file_path4, sep=",")
        df5 = pd.read_csv(info_file_path5, sep=",")
        dfn = pd.concat([df2, df3, df4, df5], axis=0)
        #df = df[df["SeriesDescription"]=="full mammogram images"]
        #df = df[df["SeriesDescription"]=="ROI mask images"]
        all_patient_ids = list(set(list(df['PatientID'].apply(get_patient_ids).values)))
        for pid in all_patient_ids:
            sub_df = df[df['PatientID'].str.contains(pid)]
            case_dict[pid] = list(sub_df['image_path'].values)
            for f in case_dict[pid]:
                pid = df[df["image_path"]==f]["PatientID"].values[0]
                pat = dfn[dfn['image file path'].str.contains(pid)]['pathology'].values[0]
                file_dict[f] = pathology2class[pat]

    input_train, input_test = train_test_split(all_patient_ids, test_size=test_size)
    input_valid, input_test = train_test_split(input_test, test_size=0.5)
    return input_train, input_valid, input_test, case_dict, file_dict

def birads2class(x):
    if x in birads_dict['normal']:
        return class2int['normal']
    elif x in birads_dict['benign']:
        return class2int['benign'] 
    elif x in birads_dict['malignant']:
        return class2int['malignant']
    
def get_patient_ids(x):
    return 'P_'+x.split('_')[2]

def split_patient_ids(x):
    return x.split('.')[0]

def get_image_paths(x):
    return x.replace('CBIS-DDSM/', 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset/')

def imgresize(filepath, resize_val, norm_pos=False, apply_denoise=False):
    apply_morph=True
    apply_medianf=False
    use_otsu=False
    apply_clahe=False
    img = cv2.imread(filepath)
    dim = (resize_val, resize_val)
    img_resize = cv2.resize(img, dim)
    if apply_denoise:
        img_resize = denoise(img_resize, apply_morph, apply_medianf, use_otsu, apply_clahe)
    if norm_pos:
        img_resize = flip_image(img_resize, 'left')
    cv2.imwrite(filepath, img_resize)

def flip_image(image, new_pos='left'):
    image_pos = check_image_orientation(image)
    if image_pos != new_pos:
        image = cv2.flip(image, 1)
    return image

def check_image_orientation(image):
    img_sum = np.sum(image, axis=0)
    img_mean = int(len(img_sum)/2)
    left_sum = np.sum(img_sum[0:img_mean])
    right_sum = np.sum(img_sum[img_mean::])
    if left_sum > right_sum:
        img_orientation = 'left'
    else:
        img_orientation = 'right'
    return img_orientation

def denoise(image, apply_morph, apply_medianf, use_otsu, apply_clahe):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_uint8 = cv2.convertScaleAbs(gray)
    if use_otsu:
        (thresh, blackAndWhiteImage) = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        (thresh, blackAndWhiteImage1) = cv2.threshold(gray_uint8, 25, 255, cv2.THRESH_BINARY)
        (thresh, blackAndWhiteImage2) = cv2.threshold(gray_uint8, 250, 255, cv2.THRESH_BINARY_INV)
        blackAndWhiteImage = cv2.bitwise_and(blackAndWhiteImage1, blackAndWhiteImage2)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhiteImage, None, None, None, 8, cv2.CV_32S)
    areas = stats[1:,cv2.CC_STAT_AREA]
    areas2 = list(set(areas))
    areas2.sort()
    areas_thr = areas2[-2] if len(areas2) > 1 else 8000
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] > areas_thr:
            result[labels == i + 1] = 1
    if apply_morph:
        kernel = np.ones((5,5),np.uint8)
        result = cv2.erode(result, kernel, iterations=1)
        result = cv2.dilate(result, kernel, iterations=1)
    gray = gray*result
    if apply_medianf:
        gray = cv2.medianBlur(gray,5)
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    return gray

def save_patches(image_file_path, output_path, step):
    img = cv2.imread(image_file_path)
    patches = patchify(img, (224, 224, 3), step=step)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j, 0]
            patch_path = output_path + '/' + os.path.basename(image_file_path).split('.')[0] + '_' + str(i) + '_' + str(j) + '.jpg'
            cv2.imwrite(patch_path, patch)

def create_folders(input_train, input_valid, input_test, case_dict, file_dict, outputpath, resize_val, 
                   norm_pos, apply_denoise, dataset, keep_folds, class2int):
    pass


if __name__ == '__main__':
    dataset = 'cbis_ddsm'
    dataset_path = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset'
    test_size = 0.3
    patch_shape = 224
    step=224
    norm_pos = False
    apply_denoise = True
    keep_folds = False

    input_train, input_valid, input_test, case_dict, file_dict = split_datasets(dataset, dataset_path, test_size)
