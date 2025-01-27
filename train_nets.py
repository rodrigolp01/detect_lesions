from tensorflow.keras.layers import *
import os
import numpy as np
import argparse
import configparser
from tensorflow.keras.models import Model
import tensorflow as tf
#from src.generator import DataGenerator
#https://www.kaggle.com/code/bistuzmf/mobilenetv3large/notebook
from tensorflow.keras.applications import Xception, VGG16, ResNet50, \
        InceptionV3, DenseNet169, NASNetLarge, EfficientNetB0, EfficientNetB3, \
        MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess_input
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess_input
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, \
        CSVLogger, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as k
from datetime import datetime
from tensorflow.keras import optimizers
import pandas as pd
import csv
from utils import plotConfMatrix, confusion_matrix_metrics, get_all_images, resize_images
from sklearn.metrics import confusion_matrix, jaccard_score, f1_score, roc_auc_score, precision_score, accuracy_score
import math
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from colorama import Fore
from tensorflow.keras.optimizers import Adam
from create_models import get_model, get_model_v2


MODELS = {
	"vgg16": VGG16,
    "mobilenetv3large": MobileNetV3Large,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet50": ResNet50,
    "efficientnetb0": EfficientNetB0, #https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    "efficientnetb3": EfficientNetB3, #https://stackoverflow.com/questions/71662044/how-can-i-fine-tune-efficientnetb3-model-and-retain-some-of-its-exisiting-labels
    "nasnetlarge": NASNetLarge, #https://www.kaggle.com/ashishpatel26/beginner-tutorial-nasnet-pneumonia-detection
    'densenet169': DenseNet169 #https://www.pluralsight.com/guides/introduction-to-densenet-with-tensorflow
}

TRAGETSIZE = {
    "xception": 299,
    "vgg16": 224,
    "mobilenetv3": 224,
    "mobilenetv3large": 224,
    "inception": 299,
    "resnet50": 224,
    "densenet169": 224,
    "nasnetlarge": 331,
    "efficientnetb0": 224,
    "efficientnetb3": 300
}

LABEL_DICT_3cl = {'normal': 0, 'benign': 1, 'malignant':2}
LABEL_DICT_4cl = {'benign_mass':0, 'benign_calcification':1, 'malignant_mass':2, 'malignant_calcification':3}

def get_batch_size(model_name, fine_tuning_all):
    if model_name == 'vgg16':
        return 16
    elif model_name == 'mobilenetv3':
        return 8
    elif model_name == 'mobilenetv3large':
        return 8
    elif model_name == 'xception':
        if fine_tuning_all == 1:
            return 8
        else:
            return 16
    elif model_name == 'resnet50':
        if fine_tuning_all == 1:
            return 2
        else:
            return 16
    elif model_name == 'inception':
        if fine_tuning_all == 1:
            return 2
        else:
            return 16
    elif model_name == 'nasnetlarge':
        return 8
    elif model_name == 'efficientnetb0':
        return 16
    elif model_name == 'efficientnetb3':
        return 16
    
def _main(args):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ** get configuration
    config_file = args.conf
    config_client = configparser.ConfigParser()
    config_client.read(config_file)

    # ** set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config_client.get('gpu', 'gpu')

    # ** Model configuration
    input_size = config_client.getint('model', 'input_size')
    num_classes = config_client.getint('model', 'num_classes')
    model_name = config_client.get('model', 'model_name') if not args.model_name else args.model_name

    # ** training configuration
    fine_tuning_epochs = config_client.getint('train', 'fine_tuning_epochs')
    fine_tuning_epochs_all = config_client.getint('train', 'fine_tuning_epochs_all')
    transfer_learning_epochs = config_client.getint('train', 'transfer_learning_epochs')
    save_path = config_client.get('train', 'save_path')
    class_weights = config_client.getint('train', 'class_weights')
    loss_func = config_client.get('train', 'loss')
    aug_freq = config_client.getfloat('train', 'aug_freq')
    tl_optimizer = config_client.get('train', 'tl_optimizer')
    ft_optimizer = config_client.get('train', 'ft_optimizer')
    pre_processing = config_client.get('train', 'pre_processing')
    do_aug = config_client.getint('train', 'do_aug')
    tl_patience = config_client.getint('train', 'tl_patience')
    ft_patience = config_client.getint('train', 'ft_patience')
    fine_tuning = config_client.getint('train', 'fine_tuning')
    fine_tuning_all = config_client.getint('train', 'fine_tuning_all') if not args.finetunning_all else int(args.finetunning_all)
    kfolds = config_client.getint('train', 'kfolds')
    descr = config_client.get('train', 'descr')
    class_mode = config_client.get('train', 'class_mode')
    learning_rate = config_client.getfloat('train', 'learning_rate')
    weight_decay = config_client.getfloat('train', 'weight_decay')

    #if fine_tuning_all == 1:
    #    fine_tuning = 0

    batch_size = get_batch_size(model_name, fine_tuning_all)

    dataset = config_client.get('data', 'dataset') if not args.dataset_name else args.dataset_name
    class_names = config_client.get('data', 'class_names')
    train_directory = config_client.get('data', 'train')
    valid_directory = config_client.get('data', 'valid')
    test_directory = config_client.get('data', 'test')

    LABEL_DICT = LABEL_DICT_3cl

    if pre_processing == 'resnet50_norm':
        if do_aug == 1:
            train_datagen = ImageDataGenerator(vertical_flip=True, rotation_range=20, brightness_range=[0.2,1.0], preprocessing_function=resnet50_preprocess_input)
        else:
            train_datagen = ImageDataGenerator(preprocessing_function=resnet50_preprocess_input)
        validation_datagen = ImageDataGenerator(preprocessing_function=resnet50_preprocess_input)
        test_datagen = ImageDataGenerator(preprocessing_function=resnet50_preprocess_input)
    elif pre_processing == 'vgg16_norm':
        if do_aug == 1:
            train_datagen = ImageDataGenerator(vertical_flip=True, rotation_range=20, brightness_range=[0.2,1.0], preprocessing_function=vgg16_preprocess_input)
        else:
            train_datagen = ImageDataGenerator(preprocessing_function=vgg16_preprocess_input)
        validation_datagen = ImageDataGenerator(preprocessing_function=vgg16_preprocess_input)
        test_datagen = ImageDataGenerator(preprocessing_function=vgg16_preprocess_input)
    elif pre_processing == 'xception_norm':
        if do_aug == 1:
            train_datagen = ImageDataGenerator(vertical_flip=True, rotation_range=20, brightness_range=[0.2,1.0], preprocessing_function=xception_preprocess_input)
        else:
            train_datagen = ImageDataGenerator(preprocessing_function=xception_preprocess_input)
        validation_datagen = ImageDataGenerator(preprocessing_function=xception_preprocess_input)
        test_datagen = ImageDataGenerator(preprocessing_function=xception_preprocess_input)    
    else:
        validation_datagen = ImageDataGenerator()
        train_datagen = ImageDataGenerator()

    model_dim = TRAGETSIZE[model_name]
    model_instance = MODELS[model_name]
    base_model = None    
    base_model = model_instance(input_shape=(model_dim, model_dim, 3), weights='imagenet', include_top=False)
    model_filename = 'cw' + str(class_weights) + '-aug' + str(do_aug) + '-base-' + dataset + '.h5'
    model_filename_ft = 'cw' + str(class_weights) + '-aug' + str(do_aug) + '-base-' + dataset + '_finetuning.h5'
    model_filename_ft_all = 'cw' + str(class_weights) + '-aug' + str(do_aug) + '-base-' + dataset + '_finetuningall.h5'
    csv_log_filename = 'cw' + str(class_weights) + '-aug' + str(do_aug) + '-base-' + dataset + '.csv'
    csv_log_filename_ft = 'cw' + str(class_weights) + '-aug' + str(do_aug) + '-base-' + dataset + '_finetuning.csv'
    csv_log_filename_ft_all = 'cw' + str(class_weights) + '-aug' + str(do_aug) + '-base-' + dataset + '_finetuningall.csv'

    class_names = class_names.split(',')
    #weighing = {int(el):0 for el in class_names}

    train_test_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M')
    test_kf_folder = os.path.join(ROOT_DIR, 'test', model_name, 'test_'+train_test_datetime)
    if not os.path.exists(test_kf_folder):
        os.makedirs(test_kf_folder)

    csv_df_mean = pd.DataFrame()
    csv_df_mean['labels'] = list(LABEL_DICT.keys())
    csv_df_mean['acc'] = 0
    csv_df_mean['jaccard'] = 0
    csv_df_mean['dice'] = 0
    csv_df_mean['precision'] = 0
    conf_matrix_mean = np.zeros((len(LABEL_DICT), len(LABEL_DICT)))
    train_elapsed_time_tl = 0
    train_elapsed_time_ft = 0
    train_elapsed_time_ft_all = 0

    csv_confs = {}
    csv_confs['fine_tuning_epochs'] = fine_tuning_epochs
    csv_confs['fine_tuning_epochs_all'] = fine_tuning_epochs_all
    csv_confs['transfer_learning_epochs'] = transfer_learning_epochs
    csv_confs['transfer_learning_optimizer'] = tl_optimizer
    csv_confs['fine_tuning_optimizer'] = ft_optimizer
    csv_confs['transfer_learning_loss_func'] = loss_func
    csv_confs['fine_tuning_loss_func'] = loss_func
    csv_confs['pre_processing'] = pre_processing #options: imagenet_mean, norm
    csv_confs['data_aug'] = do_aug
    csv_confs['class_weights'] = class_weights
    csv_confs['fine_tuning'] = fine_tuning
    csv_confs['fine_tuning_all'] = fine_tuning_all
    csv_confs['model_name'] = model_name
    csv_confs['dataset'] = dataset
    csv_confs['train_directory'] = train_directory
    csv_confs['descr'] = descr
    csv_confs['class_mode'] = class_mode
    with open(os.path.join(test_kf_folder, 'train_configs.csv'), 'w') as f:
        for key in csv_confs.keys():
            f.write("%s,%s\n"%(key,csv_confs[key]))

    mean_dice_list = []

    #model = get_model(model_name, base_model, num_classes, loss_func)
    model = get_model_v2(model_name, base_model, num_classes, loss_func)
    if tl_optimizer == 'adam':
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_func, metrics=['accuracy'])
    else:
        model.compile(optimizer=tl_optimizer, loss=loss_func, metrics=['accuracy'])
    weighing = {int(el):0 for el in class_names}

    train_generator = train_datagen.flow_from_directory(train_directory, 
                target_size=(model_dim, model_dim),batch_size=batch_size,class_mode=class_mode)
    valid_generator = validation_datagen.flow_from_directory(valid_directory, 
                target_size=(model_dim, model_dim),batch_size=batch_size,class_mode=class_mode)
    test_generator = test_datagen.flow_from_directory(test_directory,
                target_size=(model_dim, model_dim),color_mode="rgb",batch_size=1,
                shuffle = False,class_mode=class_mode)
    
    _, classes_counts = np.unique(train_generator.labels, return_counts=True)
    total_counts = sum(classes_counts)
    for cls in range(num_classes):
        weighing[cls] = 1 - classes_counts[cls]/total_counts
    print('weights: ', weighing)

    train_folder = os.path.join(ROOT_DIR, 'training', model_name, 
                                    'train_'+train_test_datetime, 'training')
    test_folder = os.path.join(ROOT_DIR, 'test', model_name, 
                                   'test_'+train_test_datetime, 'test')
    
    weights_directory = os.path.join(train_folder, 'weights')
    if not os.path.exists(weights_directory):
        os.makedirs(weights_directory)

    csv_log_path = os.path.join(train_folder, 'logs')
    if not os.path.exists(csv_log_path):
        os.makedirs(csv_log_path)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    save_path = os.path.join(weights_directory, model_filename)
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    csv_log_path = os.path.sep.join([csv_log_path, csv_log_filename])
    csvlog_callback=CSVLogger(filename=csv_log_path)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=tl_patience, verbose=1)

    tl_start = time.time()

    if class_weights == 0:
            print('No weighing***********************')
            weighing = None

    history = model.fit_generator(generator   = train_generator,
                        validation_data = valid_generator,
                        epochs          = transfer_learning_epochs,
                        callbacks       = [checkpoint, early_stopping, csvlog_callback],
                        class_weight    = weighing
                        )
    
    tl_end = time.time()
    train_elapsed_time_tl = train_elapsed_time_tl + (tl_end - tl_start)

    training_loss=history.history['loss']
    valid_loss=history.history['val_loss']

    plt.figure()
    plt.plot(training_loss, label='training_loss')  
    plt.plot(valid_loss, label='valid_loss')  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(loc='upper left')
    plt.savefig(os.path.sep.join([train_folder, 'logs', 'tl_train_history.png']))
    plt.close()

    if fine_tuning == 1:
        print('Fine-Tuning!')
        model.load_weights(save_path)

        if model_name=='resnet50':
            based_model_last_block_layer_number = 165
        elif model_name=='vgg16':
            based_model_last_block_layer_number = 15
        elif model_name=='xception':
            based_model_last_block_layer_number = 126

        for layer in model.layers[:based_model_last_block_layer_number]:
            layer.trainable = False
        for layer in model.layers[based_model_last_block_layer_number:]:
            layer.trainable = True

        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

        if ft_optimizer == 'adam':
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_func, metrics=['accuracy'])
        else:
            model.compile(optimizer=ft_optimizer, loss=loss_func, metrics=['accuracy'])

        save_path = os.path.join(weights_directory, model_filename_ft)
        checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
        csv_log_path = os.path.join(train_folder, 'logs')
        csv_log_path = os.path.sep.join([csv_log_path, csv_log_filename_ft])
        csvlog_callback=CSVLogger(filename=csv_log_path)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=ft_patience, verbose=1)

        ft_start = time.time()

        if class_weights == 0:
            print('No weighing***********************')
            weighing = None

        history = model.fit_generator(generator   = train_generator,
                            validation_data = valid_generator,
                            epochs          = fine_tuning_epochs,
                            callbacks       = [checkpoint, early_stopping, csvlog_callback],
                            class_weight    = weighing
                            )
        
        ft_end = time.time()
        train_elapsed_time_ft = train_elapsed_time_tl + (ft_end - ft_start)
            
        training_loss=history.history['loss']
        valid_loss=history.history['val_loss']
            
        plt.figure()
        plt.plot(training_loss, label='training_loss')  
        plt.plot(valid_loss, label='valid_loss')
        plt.title('model loss')  
        plt.ylabel('loss')  
        plt.xlabel('epoch')  
        plt.legend(loc='upper left')
        plt.savefig(os.path.sep.join([train_folder, 'logs', 'fl_train_history.png']))
        plt.close()

    if fine_tuning_all == 1:
        print('Fine-Tuning-All!')
        model.load_weights(save_path)

        for layer in model.layers[0:]:
                layer.trainable = True
            
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

        if ft_optimizer == 'adam':
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_func, metrics=['accuracy'])
        else:
            model.compile(optimizer=ft_optimizer, loss=loss_func, metrics=['accuracy'])

        save_path = os.path.join(weights_directory, model_filename_ft_all)
        checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
        csv_log_path = os.path.join(train_folder, 'logs')
        csv_log_path = os.path.sep.join([csv_log_path, csv_log_filename_ft_all])
        csvlog_callback=CSVLogger(filename=csv_log_path)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=ft_patience, verbose=1)

        ft_all_start = time.time()

        if class_weights == 0:
            print('No weighing***********************')
            weighing = None

        history = model.fit_generator(generator   = train_generator,
                            validation_data = valid_generator,
                            epochs          = fine_tuning_epochs_all,
                            callbacks       = [checkpoint, early_stopping, csvlog_callback],
                            class_weight    = weighing
                            )
            
        ft_all_end = time.time()
        train_elapsed_time_ft_all = train_elapsed_time_ft + (ft_all_end - ft_all_start)

        training_loss=history.history['loss']
        valid_loss=history.history['val_loss']
            
        plt.figure()
        plt.plot(training_loss, label='training_loss')  
        plt.plot(valid_loss, label='valid_loss')  
        plt.title('model loss')  
        plt.ylabel('loss')  
        plt.xlabel('epoch')  
        plt.legend(loc='upper left')
        plt.savefig(os.path.sep.join([train_folder, 'logs', 'fl_all_train_history.png']))
        plt.close()

    filenames = test_generator.filenames
    nb_samples = len(filenames)
    model.load_weights(save_path)
    model.compile(optimizer=ft_optimizer, loss=loss_func, metrics=['accuracy'])
    predict = model.predict_generator(test_generator,steps = nb_samples)
    predicted_class_indices=np.argmax(predict,axis=1)
    conf_matrix = confusion_matrix(y_true=test_generator.classes, 
                        y_pred=predicted_class_indices, labels=list(LABEL_DICT.values()))
    plotConfMatrix(conf_matrix, list(LABEL_DICT.keys()), 
                       os.path.join(test_folder, 'confusion_matrix.png'))
    csv_df = pd.DataFrame()
    csv_df['labels'] = list(LABEL_DICT.keys())
    acc = conf_matrix.diagonal()/conf_matrix.sum(axis=1)
    acc = [0 if math.isnan(x) else x for x in acc]
    csv_df['acc'] = acc
    jaccard = jaccard_score(y_true=test_generator.classes, y_pred=predicted_class_indices, labels=list(LABEL_DICT.values()), average=None)
    csv_df['jaccard'] = jaccard
    dice = f1_score(y_true=test_generator.classes, y_pred=predicted_class_indices, labels=list(LABEL_DICT.values()), average=None)
    csv_df['dice'] = dice
    mean_dice = f1_score(y_true=test_generator.classes, y_pred=predicted_class_indices, labels=list(LABEL_DICT.values()), average='weighted')
    precision = precision_score(y_true=test_generator.classes, y_pred=predicted_class_indices, labels=list(LABEL_DICT.values()), average=None)
    csv_df['precision'] = precision
    csv_df.to_csv(os.path.join(test_folder, 'metrics.csv'), index=False)
    print(mean_dice)

    csv_confs = {}
    csv_confs['train_elapsed_time'] = train_elapsed_time_ft_all
    with open(os.path.join(test_kf_folder, 'train_configs.csv'), 'a') as f:
        for key in csv_confs.keys():
            f.write("%s,%s\n"%(key,csv_confs[key]))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('-c', '--conf', help='path to configuration file', default='config/train.ini')   
    argparser.add_argument('-m', '--model_name', help='Model Name', default=None)
    argparser.add_argument('-d', '--dataset_path', help='Dataset Path', default=None)
    argparser.add_argument('-b', '--dataset_name', help='Dataset Name', default=None)
    argparser.add_argument('-f', '--finetunning_all', help='Fine-Tunning All Layers', default=None)
    args = argparser.parse_args()
    _main(args)

    k.clear_session()