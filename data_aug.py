from imgaug import augmenters as iaa
import argparse
import os
import cv2
import shutil
from tqdm import tqdm
from colorama import Fore


def do_augmentation(images_path, output_path, num_times):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    for filename in tqdm(os.listdir(images_path), desc='/'.join(images_path.split('/')[-2:]), position=0, leave=True, 
                         bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
        images = []
        im = cv2.imread(os.path.sep.join([images_path, filename]))
        images.append(im)
        for i in range(1, num_times):
            geo_fliplr = iaa.OneOf([iaa.Fliplr(1.0)])#iaa.Fliplr(1.0) #iaa.OneOf([iaa.Fliplr(1.0)])
            geo_flipud = iaa.OneOf([iaa.Fliplr(1.0)])#iaa.Flipud(1.0) #iaa.OneOf([iaa.Fliplr(1.0)])
            geo_rotn = iaa.OneOf([iaa.Affine(rotate=-30.0)])#iaa.Affine(rotate=-30.0) #iaa.OneOf([iaa.Affine(rotate=-30.0)])
            geo_rotp = iaa.OneOf([iaa.Affine(rotate=30.0)])#iaa.Affine(rotate=30.0) #iaa.OneOf([iaa.Affine(rotate=30.0)])
            #geo_fliplrup = iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)])
            geo_fliplr_rotp = iaa.Sequential([iaa.Fliplr(1.0), iaa.Affine(rotate=30.0)])

            geo_fliplr_images = geo_fliplr(images=images)
            geo_flipud_images = geo_flipud(images=images)
            #geo_fliplrup_images = geo_fliplrup(images=images)
            geo_rotn_images = geo_rotn(images=images)
            geo_rotp_images = geo_rotp(images=images)
            geo_fliplr_rotp_images = geo_fliplr_rotp(images=images)

            output_name = os.path.splitext(filename)[0] + '_flr' + str(i) + '.png'
            cv2.imwrite( os.path.sep.join([output_path, output_name]), geo_fliplr_images[0] )

            output_name = os.path.splitext(filename)[0] + '_fup' + str(i) + '.png'
            cv2.imwrite( os.path.sep.join([output_path, output_name]), geo_flipud_images[0] )

            #output_name = os.path.splitext(filename)[0] + '_lrup' + str(i) + '.png'
            #cv2.imwrite( os.path.sep.join([output_path, output_name]), geo_fliplrup_images[0] )

            output_name = os.path.splitext(filename)[0] + '_rn' + str(i) + '.png'
            cv2.imwrite( os.path.sep.join([output_path, output_name]), geo_rotn_images[0] )

            output_name = os.path.splitext(filename)[0] + '_rp' + str(i) + '.png'
            cv2.imwrite( os.path.sep.join([output_path, output_name]), geo_rotp_images[0] )

            output_name = os.path.splitext(filename)[0] + '_lrrp' + str(i) + '.png'
            cv2.imwrite( os.path.sep.join([output_path, output_name]), geo_fliplr_rotp_images[0] )

        cv2.imwrite( os.path.sep.join([output_path, filename]), im )