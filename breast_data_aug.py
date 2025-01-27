from data_aug import do_augmentation
import os
import shutil
from tqdm import tqdm
from colorama import Fore

num_times = 2
class_list = ['0','1','2','3']
foldersets = ['train', 'valid', 'test']
input_dataset = 'cbis_ddsm_folds_4cl_cc_enh2'
output_dataset = 'cbis_ddsm_folds_4cl_cc_enh2_aug'

for fs in foldersets:
    for cl in class_list:
        input_path = input_dataset + '/' + fs + '/' + cl
        output_path = output_dataset + '/' + fs + '/' + cl
        do_augmentation(input_path, output_path, num_times)

