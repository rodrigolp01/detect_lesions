#from skimage.draw import polygon_perimeter
#import numpy as np
import plistlib
import cv2
import math
from load_inbreast_roi import load_point
from tqdm import tqdm
from colorama import Fore
import os


def calc_radius(x_list, y_list, x_center, y_center):
    dist_list = []
    for i in range(0, len(x_list)):
        x = x_list[i]
        y = y_list[i]
        dist_list.append(math.sqrt( (x_center - x)**2 + (y_center - y)**2 ))

    return dist_list.index(max(dist_list))
    

def calc_center_point_and_radius(points):
    x_list = []
    y_list = []
    for point in points:
        y, x = load_point(point)
        x_list.append(x)
        y_list.append(y)

    x_center, y_center = sum(x_list)/len(x_list), sum(y_list)/len(y_list)
    radius = calc_radius(x_list, y_list, x_center, y_center)

    return x_center, y_center, radius
    

def return_inbreast_rois(xml_images_path, images_path, output_roi_path, output_drawn_path):

    counter = 0

    if not os.path.isdir( output_drawn_path ):
        os.makedirs(output_drawn_path)

    for file in tqdm(os.listdir(xml_images_path), desc='xml image files', position=0, leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)): 
        file_path = os.path.sep.join([xml_images_path, file])
        filename, _  = os.path.splitext(file)

        with open(file_path, 'rb') as f:
            plist_dict = plistlib.load(f, fmt=plistlib.FMT_XML)['Images'][0]
            numRois = plist_dict['NumberOfROIs']
            rois = plist_dict['ROIs']
            assert len(rois) == numRois
            #img_path = os.path.sep.join([images_path, filename + '.png'])
            filename = [fn for fn in os.listdir(images_path) if fn.startswith(filename)]
            img_path = os.path.sep.join([images_path, filename[0]])
            #print(img_path)
            img=cv2.imread(img_path)
            img2 = img.copy()
            #print(img)

            for roi in tqdm(rois, desc=filename[0], position=0, leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
                numPoints = roi['NumberOfPoints']
                points = roi['Point_px']
                cl = roi['Name']
                assert numPoints == len(points)
                #print(points)
                x_center, y_center, radius = calc_center_point_and_radius(points)
                #print(x_center, y_center, radius)
                if radius == 0:
                    radius = 10
                if x_center-radius < 0 or y_center-radius < 0:
                    radius = min([x_center, y_center])

                bbox_x = int(x_center-radius)
                bbox_y = int(y_center-radius)
                radius = int(1.5*radius)
                counter += 1
                #print(x_center, y_center, radius)
                roi_cropped = img[bbox_y:bbox_y+2*radius, bbox_x:bbox_x+2*radius]
                cv2.rectangle(img2, (bbox_x,bbox_y), (bbox_x+2*radius,bbox_y+2*radius), (0, 255, 0))
                #print(roi_cropped)
                cl_path = os.path.sep.join([output_roi_path, cl])
                if not os.path.isdir( cl_path ):
                    os.makedirs(cl_path)

                #filename, _ = os.path.splitext(filename[0])
                roi_cropped_path = os.path.sep.join([cl_path, filename[0].split('.png')[0] + '_' + str(counter) + '.png'])
                try:
                    cv2.imwrite(roi_cropped_path,roi_cropped)
                except:
                    print(points)
                    print(x_center, y_center, radius)
                    print(roi_cropped)

        cv2.imwrite(output_drawn_path + '/' + filename[0], img2)

xml_images_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/INbreast Release 1.0/AllXML'
images_path = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/inbreast_dataset'
output_roi_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/inbreast/roi'
output_drawn_path = 'C:/Users/rodri/Doutorado_2023/detect_lesions/inbreast/drawn'
return_inbreast_rois(xml_images_path, images_path, output_roi_path, output_drawn_path)
