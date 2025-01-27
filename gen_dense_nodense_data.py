import os 
from tensorflow.keras.applications import Xception, VGG16, ResNet50, \
        InceptionV3, DenseNet169, NASNetLarge, EfficientNetB0, EfficientNetB3, \
        MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.layers import *
import pandas as pd
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import get_all_images
from train_nets import get_model
import eli5
from pathlib import Path
from tqdm import tqdm
from colorama import Fore

model_name = 'resnet50'
pre_processing = 'resnet50_norm'
breast_density_2cl_dict = {'0':'0', '1':'0', '2':'1', '3':'1'}

model_weights_path = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/code/training/resnet50/train_2023-05-08_23-41/training9/weights/cw1-aug0-base-inbreast_mini_cbis_ddsm_finetuning.h5'
base_model = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
test_directory = 'inbreast_cbis_mini_ddsm_folds'

model = get_model(model_name, base_model, 4, 'categorical_crossentropy')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model.load_weights(model_weights_path)
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

for cn in list(set(list(breast_density_2cl_dict.values()))):
    output_directory = test_directory + cn
    for ds in os.listdir(test_directory):
        ds_path = test_directory + '/' + ds
        for cl in os.listdir(ds_path):
            cl_path = output_directory + '/' + ds + '/' + cl
            if os.path.isdir(cl_path) and os.listdir(cl_path):
                shutil.rmtree(cl_path)
            os.makedirs(cl_path, exist_ok=True)

for ds in os.listdir(test_directory):
    ds_path = test_directory + '/' + ds
    for cl in os.listdir(ds_path):
        cl_path = ds_path + '/' + cl
        for im in tqdm(os.listdir(cl_path), desc=ds+'/'+cl):
            img_path = cl_path + '/' + im
            img = image.load_img(img_path, target_size = (224, 224))
            img_array = image.img_to_array(img)
            img_expanded = np.expand_dims(img_array, axis = 0)
            img_ready = resnet50_preprocess_input(img_expanded)
            pred = model.predict(img_ready)
            pred = str(np.argmax(pred,axis=1)[0])
            newfilepath = test_directory+breast_density_2cl_dict[pred] + '/' + ds + '/' + cl + '/' + im
            shutil.copy2(img_path, newfilepath)



#test_datagen = ImageDataGenerator(preprocessing_function=resnet50_preprocess_input)

# Import image and preprocess_input
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the image with the right target size for your model
#img = image.load_img(img_path, target_size = (224, 224))

# Turn it into an array
#img_array = image.img_to_array(img)

# Expand the dimensions of the image, this is so that it fits the expected model input format
#img_expanded = np.expand_dims(img_array, axis = 0)

# Pre-process the img in the same way original images were
#img_ready = preprocess_input(img_expanded)

