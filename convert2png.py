import os
from PIL import Image
from tqdm import tqdm
from colorama import Fore

pgm_database_path = 'C:/Users/vntrolp/detect_lesions/MIAS_Database/all-mias-pgm'
png_database_path = 'C:/Users/vntrolp/detect_lesions/MIAS_Database/all-mias-png'

for file in tqdm(os.listdir(pgm_database_path), desc='pgm images', position=0, leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
    filename, extension  = os.path.splitext(file)
    if extension == ".pgm":
        new_file = "{}.png".format(filename)
        new_file = os.path.sep.join([png_database_path, new_file])
        file = os.path.sep.join([pgm_database_path, file])
        with Image.open(file) as im:
            im.save(new_file)