import os
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
from colorama import Fore
import glob
import pandas as pd
import cv2
import numpy as np

birads_dict = {'normal': ['1'], 'benign': ['2','3'], 'malignant':['4a','4b','4c','5','6']}
class2int = {'normal': '0', 'benign': '1', 'malignant':'2'}
severity2class = {'': '0', 'B': '1', 'M':'2'}
pathology2class = {'BENIGN': '1', 'BENIGN_WITHOUT_CALLBACK': '1', 'MALIGNANT': '2'}
status2class = {'Normal': '0', 'Benign': '1', 'Cancer': '2'}

def split_datasets(dataset, dataset_path, test_size):
    all_patient_ids = []
    #all_patient_cds = []
    case_dict = {}
    file_dict = {}
    if dataset == 'inbreast':
        info_file_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/INbreast Release 1.0/INbreast.csv'
        df = pd.read_csv(info_file_path, sep=";")
        for file in os.listdir(dataset_path):
            try:
                all_patient_ids.append(file.split('_')[1])
            except:
                continue
        all_patient_ids = list(set(all_patient_ids))
        all_filenames = list(df["File Name"].values)
        #all_patient_cds = list(df["Bi-Rads"].apply(birads2class).values)
        for pid in all_patient_ids:
            case_dict[pid] = glob.glob(dataset_path + '/*' + pid + '*.png')
        for fn in all_filenames:
            birads = df[df["File Name"]==fn]["Bi-Rads"].values[0]
            file_dict[str(fn)] = birads2class(birads)
    elif dataset == 'mias':
        info_file_path = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/mias_dataset/mias_Info.txt'
        df = pd.read_csv(info_file_path, sep=" ")
        df.fillna('', inplace=True)
        all_filenames = list(df["REFNUM"].values)
        all_filenames = list(set(all_filenames))
        for idx in range(0,len(all_filenames),2):
            patient_id = all_filenames[idx].split('.')[0] + '_' + all_filenames[idx+1].split('.')[0]
            all_patient_ids.append(patient_id)
            case_dict[patient_id] = [dataset_path + '/' + patient_id.split('_')[0]+'.png', 
                                     dataset_path + '/' + patient_id.split('_')[1]+'.png']
            severity = df[df["REFNUM"]==all_filenames[idx]]["SEVERITY"].values[0]
            file_dict[all_filenames[idx]] = severity2class[severity]
            file_dict[all_filenames[idx+1]] = severity2class[severity]
    elif dataset == 'cbis_ddsm':
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
        df = df[df["SeriesDescription"]=="full mammogram images"]
        all_patient_ids = list(set(list(df['PatientID'].apply(get_patient_ids).values)))
        for pid in all_patient_ids:
            sub_df = df[df['PatientID'].str.contains(pid)]
            #case_dict[pid] = list(sub_df['image_path'].apply(get_image_paths).values)
            case_dict[pid] = list(sub_df['image_path'].values)
            for f in case_dict[pid]:
                pid = df[df["image_path"]==f]["PatientID"].values[0]
                pat = dfn[dfn['image file path'].str.contains(pid)]['pathology'].values[0]
                #file_dict[get_image_paths(f)] = pathology2class[pat]
                file_dict[f] = pathology2class[pat]
    elif dataset == 'mini_ddsm':
        info_file_path = dataset_path + '/Data-MoreThanTwoMasks/Data-MoreThanTwoMasks.xlsx'
        df = pd.read_excel(info_file_path)
        all_patient_ids = list(set(list(df["fileName"].apply(split_patient_ids).values)))
        for pid in all_patient_ids:
            case_dict[pid] = list(df[df['fullPath'].str.contains(pid)]['fullPath'].values)
            for f in case_dict[pid]:
                status = df[df["fullPath"]==f]["Status"].values[0]
                file_dict[f] = status2class[status]

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

def create_folders(input_train, input_valid, input_test, case_dict, file_dict, outputpath, resize_val, 
                   norm_pos, apply_denoise, dataset, keep_folds, class2int):
    #cls = list(birads_dict.keys())
    cls = list(class2int.values())

    for cl in cls:
        train_fold_cl = os.path.sep.join([outputpath, 'train', cl])
        if os.path.isdir(train_fold_cl) and os.listdir(train_fold_cl):
            if not keep_folds:
                shutil.rmtree(train_fold_cl)
        os.makedirs(train_fold_cl, exist_ok=True)

        valid_fold_cl = os.path.sep.join([outputpath, 'valid', cl])
        if os.path.isdir(valid_fold_cl) and os.listdir(valid_fold_cl):
            if not keep_folds:
                shutil.rmtree(valid_fold_cl)
        os.makedirs(valid_fold_cl, exist_ok=True)

        test_fold_cl = os.path.sep.join([outputpath, 'test', cl])
        if os.path.isdir(test_fold_cl) and os.listdir(test_fold_cl):
            if not keep_folds:
                shutil.rmtree(test_fold_cl)
        os.makedirs(test_fold_cl, exist_ok=True)

    for case_id in tqdm(input_train, desc='Train Cases'):
        case_files = case_dict[case_id]
        for file_path in case_files:
            if dataset == 'mias':
                file_name = os.path.basename(file_path)
                label = file_dict[file_name.split('.')[0]]
            elif dataset == 'cbis_ddsm':
                label = file_dict[file_path]
                file_path = get_image_paths(file_path)
                file_name = os.path.basename(file_path).split('.')[0]+'-'+case_id+'.jpg'
            elif dataset == 'mini_ddsm':
                label = file_dict[file_path]
                #file_path = file_path.replace('\\', '/')
                file_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mini_ddsm/MINI-DDSM-Complete-PNG-16/' + file_path.replace('\\', '/')
                file_name = file_path.split('/')[-1]
            else:
                file_name = os.path.basename(file_path)
                label = file_dict[file_name.split('_')[0]]
            newfilepath = os.path.sep.join([outputpath, 'train', label, file_name])
            shutil.copy2(file_path, newfilepath)
            imgresize(newfilepath, resize_val, norm_pos, apply_denoise)

    for case_id in tqdm(input_valid, desc='Valid Cases'):
        case_files = case_dict[case_id]
        for file_path in case_files:
            if dataset == 'mias':
                file_name = os.path.basename(file_path)
                label = file_dict[file_name.split('.')[0]]
            elif dataset == 'cbis_ddsm':
                label = file_dict[file_path]
                file_path = get_image_paths(file_path)
                file_name = os.path.basename(file_path)
            elif dataset == 'mini_ddsm':
                label = file_dict[file_path]
                #file_path = file_path.replace('\\', '/')
                file_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mini_ddsm/MINI-DDSM-Complete-PNG-16/' + file_path.replace('\\', '/')
                file_name = file_path.split('/')[-1]
            else:
                file_name = os.path.basename(file_path)
                label = file_dict[file_name.split('_')[0]]
            newfilepath = os.path.sep.join([outputpath, 'valid', label, file_name])
            shutil.copy2(file_path, newfilepath)
            imgresize(newfilepath, resize_val, norm_pos, apply_denoise)

    for case_id in tqdm(input_test, desc='Test Cases'):
        case_files = case_dict[case_id]
        for file_path in case_files:
            if dataset == 'mias':
                file_name = os.path.basename(file_path)
                label = file_dict[file_name.split('.')[0]]
            elif dataset == 'cbis_ddsm':
                label = file_dict[file_path]
                file_path = get_image_paths(file_path)
                file_name = os.path.basename(file_path)
            elif dataset == 'mini_ddsm':
                label = file_dict[file_path]
                #file_path = file_path.replace('\\', '/')
                file_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mini_ddsm/MINI-DDSM-Complete-PNG-16/' + file_path.replace('\\', '/')
                file_name = file_path.split('/')[-1]
            else:
                file_name = os.path.basename(file_path)
                label = file_dict[file_name.split('_')[0]]
            newfilepath = os.path.sep.join([outputpath, 'test', label, file_name])
            shutil.copy2(file_path, newfilepath)
            imgresize(newfilepath, resize_val, norm_pos, apply_denoise)

if __name__ == '__main__':
    #inbreast = C:/Users/rodri/Doutorado_2023/breast_density_classification/data/inbreast_dataset
    #mias = C:/Users/rodri/Doutorado_2023/breast_density_classification/data/mias_dataset
    #cbis-ddsm = C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset
    #mini-ddsm = C:/Users/rodri/Doutorado_2023/detect_lesions/mini_ddsm
    dataset1 = 'inbreast'
    dataset2 = 'mias'
    dataset3 = 'cbis_ddsm'
    dataset4 = 'mini_ddsm'
    #dataset_path1 = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/inbreast_dataset'
    dataset_path1 = 'C:/Users/rodri/Doutorado_2023/detect_lesions/inbreast/drawn'
    dataset_path2 = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/mias_dataset'
    dataset_path3 = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/cbis_ddsm_dataset'
    dataset_path4 = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mini_ddsm'
    outputpath = 'C:/Users/rodri/Doutorado_2023/detect_lesions/inbreast_folds_drawn'
    test_size = 0.3
    resize_val = 224
    norm_pos = False
    apply_denoise = False
    keep_folds = False

    input_train, input_valid, input_test, case_dict, file_dict = split_datasets(dataset1, dataset_path1, test_size)
    create_folders(input_train, input_valid, input_test, case_dict, file_dict, outputpath, 
                   resize_val, norm_pos, apply_denoise, dataset1, keep_folds, class2int)

    # keep_folds = True

    # input_train, input_valid, input_test, case_dict, file_dict = split_datasets(dataset2, dataset_path2, test_size)
    # create_folders(input_train, input_valid, input_test, case_dict, file_dict, outputpath, 
    #                resize_val, norm_pos, apply_denoise, dataset2, keep_folds, class2int)
    
    # input_train, input_valid, input_test, case_dict, file_dict = split_datasets(dataset3, dataset_path3, test_size)
    # create_folders(input_train, input_valid, input_test, case_dict, file_dict, outputpath, 
    #                resize_val, norm_pos, apply_denoise, dataset3, keep_folds, class2int)
    
    # input_train, input_valid, input_test, case_dict, file_dict = split_datasets(dataset4, dataset_path4, test_size)
    # create_folders(input_train, input_valid, input_test, case_dict, file_dict, outputpath, 
    #                resize_val, norm_pos, apply_denoise, dataset4, keep_folds, class2int)


