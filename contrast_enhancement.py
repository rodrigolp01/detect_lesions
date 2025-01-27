#Script for contrast enhancement before training

import cv2
import argparse
import os
import shutil
import numpy as np
from tqdm import tqdm
from colorama import Fore


def gammaCorrection(image, gamma): #try gamma < 1
    invGamma = 1.0 / gamma
    table = [((i / 255.0) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(image, table)

def do_gamma_correction(images_path, output_path, gamma):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    for filename in tqdm(os.listdir(images_path), desc='/'.join(images_path.split('/')[-1:]), position=0, leave=True, 
                          bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
        img = cv2.imread(os.path.sep.join([images_path, filename]))
        adjusted_img = gammaCorrection(img, gamma)
        cv2.imwrite(os.path.sep.join([output_path, filename]), adjusted_img)

def do_contrast_enhancement(images_path, output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    for filename in tqdm(os.listdir(images_path), desc='/'.join(images_path.split('/')[-1:]), position=0, leave=True, 
                          bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
        img = cv2.imread(os.path.sep.join([images_path, filename]))
        #-----Converting image to LAB Color model----------------------------------- 
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        #-----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)
        #-----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl,a,b))
        #-----Converting image from LAB Color model to RGB model--------------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        cv2.imwrite(os.path.sep.join([output_path, filename]),final)

def do_gamma_and_contrast_enhancement(images_path, output_path, gamma=0.7):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    for filename in tqdm(os.listdir(images_path), desc='/'.join(images_path.split('/')[-1:]), position=0, leave=True, 
                          bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
        img = cv2.imread(os.path.sep.join([images_path, filename]))
        adjusted_img = gammaCorrection(img, gamma)
        lab= cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        cl = clahe.apply(cl)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        cv2.imwrite(os.path.sep.join([output_path, filename]),final)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Script for contrast enhancement before training')
    argparser.add_argument('-i', '--images_path', help='path to images')
    argparser.add_argument('-o', '--output_path', help='output path')
    args = argparser.parse_args()
    images_path = args.images_path
    output_path = args.output_path
    do_contrast_enhancement(images_path, output_path)