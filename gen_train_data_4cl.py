import os
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
from colorama import Fore
import glob
import pandas as pd
import cv2
import numpy as np
from gen_train_data import get_patient_ids, create_folders

pathology_dict = {'BENIGN': 'benign', 'BENIGN_WITHOUT_CALLBACK': 'benign', 'MALIGNANT': 'malignant'}
abnormality_dict = {'benign_mass':'0', 'benign_calcification':'1', 'malignant_mass':'2', 'malignant_calcification':'3'}


def split_datasets(dataset, dataset_path, test_size, only_cc_view):
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

        if only_cc_view:
            df = df[df["PatientOrientation"]=="CC"]
            dfn = dfn[dfn["image view"]=="CC"]

        df = df[df["SeriesDescription"]=="full mammogram images"]
        all_patient_ids = list(set(list(df['PatientID'].apply(get_patient_ids).values)))
        #benign_mass, benign_calcification, malignant_mass, malignant_calcification
        for pid in all_patient_ids:
            sub_df = df[df['PatientID'].str.contains(pid)]
            case_dict[pid] = list(sub_df['image_path'].values)
            for f in case_dict[pid]:
                pid = df[df["image_path"]==f]["PatientID"].values[0]
                pat = dfn[dfn['image file path'].str.contains(pid)]['pathology'].values[0]
                abnt = dfn[dfn['image file path'].str.contains(pid)]['abnormality type'].values[0]
                file_dict[f] = abnormality_dict[pathology_dict[pat] + '_' + abnt]

    input_train, input_test = train_test_split(all_patient_ids, test_size=test_size)
    input_valid, input_test = train_test_split(input_test, test_size=0.5)
    return input_train, input_valid, input_test, case_dict, file_dict


if __name__ == '__main__':
    dataset3 = 'cbis_ddsm'
    dataset_path3 = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset'
    outputpath = 'C:/Users/rodri/Doutorado_2023/detect_lesions/cbis_ddsm_folds_4cl_cc'
    test_size = 0.3
    resize_val = 224
    norm_pos = False
    apply_denoise = True
    keep_folds = False
    only_cc_view = True

    input_train, input_valid, input_test, case_dict, file_dict = split_datasets(dataset3, dataset_path3, test_size, only_cc_view)
    create_folders(input_train, input_valid, input_test, case_dict, file_dict, outputpath, 
                   resize_val, norm_pos, apply_denoise, dataset3, keep_folds, abnormality_dict)

