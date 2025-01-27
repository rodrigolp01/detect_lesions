import pydicom
import os
from tqdm import tqdm
from colorama import Fore
import cv2


def dicomFolder2Png(inputpath, outputpath, isinbreast):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    for file in tqdm(os.listdir(inputpath), desc='dicom images', position=0, leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
        #print('converting file: {} to PNG'.format(file))
        if os.path.splitext(file)[1] == '.dcm':
            dicom2png( os.path.sep.join([inputpath, file]), os.path.sep.join([outputpath, file.replace('.dcm', '.png')]), isinbreast )

def dicom2png(inputfilepath, outputfilepath, isinbreast=False):
    ds = pydicom.read_file(inputfilepath)
    img = ds.pixel_array
    if isinbreast:
        img = img*22 
    cv2.imwrite(outputfilepath, img)

dataset_path = "C:/Users/vntrolp/breast_density_classification/data/INbreast Release 1.0/AllDICOMs"
output_path = "C:/Users/vntrolp/detect_lesions/inbreast/png"

dicomFolder2Png(dataset_path, output_path, True)