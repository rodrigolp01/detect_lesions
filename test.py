import os 
from tensorflow.keras.applications import VGG16, ResNet50
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
from sklearn.metrics import confusion_matrix


LABEL_DICT_3cl = {'normal': 0, 'benign': 1, 'malignant':2}
LABEL_DICT_4cl = {'benign_mass':0, 'benign_calcification':1, 'malignant_mass':2, 'malignant_calcification':3}

MODELLAYER = {
    "vgg16": 'block4_conv3', #block4_conv3, block5_conv3
    "resnet50": 'conv5_block3_out' #conv5_block3_3_conv
}

def save_gradcam(model_name, test_generator, origpath, dims, newpath):
    if model_name == 'vgg16':

        img, _ = test_generator.next()
        im = image.load_img(origpath, target_size=dims)

    elif model_name == 'resnet50':

        im = image.load_img(origpath, target_size=dims)
        doc = image.img_to_array(im)
        doc = np.expand_dims(doc, axis=0)
        img = resnet50_preprocess_input(doc)

    r=eli5.show_prediction(model, img, image=im, layer=MODELLAYER[model_name])
    r.save(newpath)


model_name = 'resnet50'
pre_processing = 'resnet50_norm'

if model_name == 'vgg16':
    data = '2023-05-15_22-38'
    model_weights_path = 'training/vgg16/train_2023-05-15_22-38/training9/weights/cw1-aug0-base-inbreast_mini_cbis_ddsm_finetuning.h5'
    base_model = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    test_directory = 'C:/Users/rodri/Doutorado_2023/breast_density_classification/data/inbreast_mini_cbis_ddsm_folds_denoise_gamma_enh/test'
elif model_name == 'resnet50':
    data = '2023-07-18_23-11'
    model_weights_path = 'training/resnet50/train_2023-07-18_23-11/training/weights/cw1-aug0-base-mias_inbreast_cbis_mini_ddsm_folds_enh_finetuning.h5'
    base_model = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    test_directory = 'C:/Users/rodri/Doutorado_2023/detect_lesions/mias_inbreast_cbis_mini_ddsm_folds_enh/test'

model = get_model(model_name, base_model, 3, 'categorical_crossentropy')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model.load_weights(model_weights_path)
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

if pre_processing == 'scale_norm':
    test_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rescale=1. / 255)
elif pre_processing == 'resnet50_norm':
    test_datagen = ImageDataGenerator(preprocessing_function=resnet50_preprocess_input)
elif pre_processing == 'vgg16_norm':
    test_datagen = ImageDataGenerator(preprocessing_function=vgg16_preprocess_input)

class_names = ['0','1','2']
test_datagen.fit(get_all_images(test_directory,class_names))
test_generator = test_datagen.flow_from_directory(test_directory,
                target_size=(224, 224),color_mode="rgb",batch_size=1,
                shuffle = False,class_mode='categorical') #, save_to_dir='C:/Users/rodri/Doutorado_2023/breast_density_classification/code/test/vgg16/test_2023-05-15_22-38/wrong_predictions/flow')

filenames = test_generator.filenames
nb_samples = len(filenames)

predict = model.predict_generator(test_generator,steps = nb_samples, verbose=1)
predicted_class_indices=np.argmax(predict,axis=1)

conf_matrix = confusion_matrix(y_true=test_generator.classes, 
                        y_pred=predicted_class_indices, labels=list(LABEL_DICT_3cl.values()))

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})

wrong_prediction_path = f'C:/Users/rodri/Doutorado_2023/detect_lesions/test/{model_name}/test_{data}/wrong_predictions/predictions'
if os.path.exists(wrong_prediction_path):
    shutil.rmtree(wrong_prediction_path)

wrong_prediction_gradcam_path = f'C:/Users/rodri/Doutorado_2023/detect_lesions/test/{model_name}/test_{data}/wrong_predictions/gradcam/{MODELLAYER[model_name]}'
if os.path.exists(wrong_prediction_gradcam_path):
    shutil.rmtree(wrong_prediction_gradcam_path)

right_prediction_gradcam_path = f'C:/Users/rodri/Doutorado_2023/detect_lesions/test/{model_name}/test_{data}/right_predictions/gradcam/{MODELLAYER[model_name]}'
if os.path.exists(right_prediction_gradcam_path):
    shutil.rmtree(right_prediction_gradcam_path)
os.makedirs(right_prediction_gradcam_path, exist_ok=True)

results.to_csv(f'C:/Users/rodri/Doutorado_2023/detect_lesions/test/{model_name}/test_{data}' + '/results.csv', index=False)

dims = model.input_shape[1:3]
count = 0
for f, p in tqdm(zip(filenames, predictions), desc='predictions', position=0, 
            leave=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET), total=nb_samples):
    ex = f.split('\\')[0]
    origpath = test_directory + '/' + f.replace('\\', '/')

    if ex != p:
        folder_name = ex + '-' + p
        os.makedirs(wrong_prediction_path + '/' + folder_name, exist_ok=True)
        os.makedirs(wrong_prediction_gradcam_path + '/' + folder_name, exist_ok=True)

        newpath = wrong_prediction_path + '/' + folder_name + '/' + f.split('\\')[1]
        shutil.copy2(origpath, newpath)

        newpath = wrong_prediction_gradcam_path + '/' + folder_name + '/' + Path(f.split('\\')[1]).stem + '.png'
        save_gradcam(model_name, test_generator, origpath, dims, newpath)

    else:

        if count % 10 == 0:

            os.makedirs(right_prediction_gradcam_path + '/' + ex, exist_ok=True)
            newpath = right_prediction_gradcam_path + '/' + ex + '/' + Path(f.split('\\')[1]).stem + '.png'
            save_gradcam(model_name, test_generator, origpath, dims, newpath)

    count += 1