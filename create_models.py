from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf


def get_model_v2(model_name, base_model, num_classes, loss_func):

    for layer in base_model.layers:
        layer.trainable = False

    if model_name == 'xception':
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)

    elif model_name == 'vgg16':
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)

    elif model_name == 'resnet50':
        x = base_model.output
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.4)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)

    elif model_name == 'inception':
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation="relu")(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)

    print('Classifier Layers before fine-tuning......')
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
    
    return model


def get_model(model_name, base_model, num_classes, loss_func):

    if model_name == 'xception':
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        if loss_func == 'binary_crossentropy':
            predictions = Dense(1, activation='sigmoid')(x)
        else:
            predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)
        print('Classifier Layers before fine-tuning......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

    elif model_name == 'vgg16':
        for layer in base_model.layers: #ou [:15] o que é melhor?
            layer.trainable = False
        print('Base Models Layers before fine-tuning......')
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name, layer.trainable)
        x = base_model.output
        x = Flatten()(x) # Flatten dimensions to for use in FC layers
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
        if loss_func == 'binary_crossentropy':
            x = Dense(1, activation='sigmoid')(x)
        else:
            x = Dense(num_classes, activation='softmax')(x) # Softmax for multiclass
        model = Model(inputs=base_model.input, outputs=x)
        print('Classifier Layers before fine-tuning......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

    elif model_name == 'resnet50':
        headModel = base_model.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(256, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        if loss_func == 'binary_crossentropy':
            headModel = Dense(1, activation="sigmoid")(headModel)
        else:
            headModel = Dense(num_classes, activation="softmax")(headModel)
        model = Model(inputs=base_model.input, outputs=headModel)
        for layer in base_model.layers:
            layer.trainable = False

        print('Classifier Layers before fine-tuning......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

    elif model_name == 'inception':
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        if loss_func == 'binary_crossentropy':
            predictions = Dense(1, activation='sigmoid')(x)
        else:
            predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        print('Classifier Layers before fine-tuning......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

    elif model_name == 'mobilenetv3large':

        for layer in base_model.layers:
            layer.trainable = False

        if loss_func == 'binary_crossentropy':
            model = tf.keras.Sequential([base_model, GlobalAveragePooling2D(), 
                                Dropout(0.3), Dense(1, activation='sigmoid')])
        else:
            model = tf.keras.Sequential([base_model, GlobalAveragePooling2D(), 
                                Dropout(0.3), Dense(num_classes, activation='softmax')])

        print('Classifier Layers before fine-tuning......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

    elif model_name == 'nasnetlarge':

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x= BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)
        x= Dropout(0.5)(x)
        x= Dense(128,activation='elu')(x)
        x= Dropout(0.5)(x)
        if loss_func == 'binary_crossentropy':
             preds=Dense(1,activation='sigmoid')(x)
        else:
            preds=Dense(num_classes,activation='softmax')(x)
        model=Model(inputs=base_model.input,outputs=preds)

        print('Classifier Layers before fine-tuning......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable) 

    elif model_name == 'efficientnetb0':

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x= BatchNormalization()(x)
        x= Dropout(0.2)(x)
        preds=Dense(num_classes,activation='softmax')(x)
        model=Model(inputs=base_model.input,outputs=preds)

        print('Classifier Layers before fine-tuning......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

    elif model_name == 'efficientnetb3':

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x= BatchNormalization()(x)
        x= Dropout(0.2)(x)
        preds=Dense(num_classes,activation='softmax')(x)
        model=Model(inputs=base_model.input,outputs=preds)

        print('Classifier Layers before fine-tuning......')
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)

    return model