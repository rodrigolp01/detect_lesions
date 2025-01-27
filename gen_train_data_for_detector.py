import os
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
from colorama import Fore
import glob
import pandas as pd
import cv2
import numpy as np


def split_datasets(dataset, dataset_path, test_size):
    pass

def create_folders(dataset_path, input_train, input_valid, input_test, label_train, label_valid, label_test, label_csv, apply_denoise):
    pass

if __name__ == '__main__':
    dataset1 = 'inbreast'
    dataset2 = 'mias'
    dataset3 = 'cbis_ddsm'
    dataset4 = 'mini_ddsm'

    dataset_path1 = 'C:/Users/rodri/Doutorado_2023/detect_lesions/inbreast/drawn'
    dataset_path2 = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/mias_dataset'
    dataset_path3 = 'C:/Users/rodri/Doutorado_2023/detect_lesions/cbis_ddsm_rois/drawn'
    dataset_path4 = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mini_ddsm'

    dataset3_label_csv = 'C:/Users/rodri/Doutorado_2023/detect_lesions/cbis_ddsm_rois/labels.csv'

    test_size = 0.3
    keep_folds = False
    apply_denoise = False

    input_train, input_valid, input_test, label_train, label_valid, label_test = split_datasets(dataset3, dataset_path3, test_size)

    create_folders(dataset_path3, input_train, input_valid, input_test, label_train, label_valid, label_test, dataset3_label_csv, apply_denoise)