import numpy as np
seed = np.random.seed(17)
import os
import cv2
import time
import sys, getopt, ast

from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, GlobalAveragePooling2D, \
    warnings
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

BATCH_SIZE = 16
nb_classes = 2

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

WEIGHTS_PATH = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"


# Modular function for Fire Node
def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


# Original SqueezeNet from paper.
def SqueezeNet(input_tensor=None, input_shape=None,
               weights='imagenet',
               classes=1000):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=input_shape[0],
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=True)
                                      #include_top=True)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)

    x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, out, name='squeezenet')

    # load weights
    if weights == 'imagenet':

        weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models')
        model.load_weights(weights_path)
    return model


def get_model(dropout, shape):
    model = SqueezeNet(input_shape=shape)
    x = model.get_layer('fire9/concat')

    x = Dropout(dropout, name='drop9_tl')(x.output)
    x = Convolution2D(nb_classes, (1, 1), padding='valid', name='conv10_tl')(x)
    x = Activation('relu', name='relu_conv10_tl')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('sigmoid', name='loss')(x)

    model_tl = Model(inputs=model.input, outputs=out, name='squeezenet-tl')
    print(model_tl.summary())

    return model_tl


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def load_images_from_folder(folder, shape, process):
    images = []
    for filename in os.listdir(folder):
        if filename != '.DS_Store' and filename != 'simu':
            img = cv2.imread(os.path.join(folder,filename))
            if img.shape[0] >= 600 and img.shape[1] >= 800:
                img = cv2.resize(img, (shape[0], shape[1]), interpolation = cv2.INTER_AREA)
                if img is not None:
                    images.append(np.array(img).reshape(shape[0], shape[1], shape[2]))

                if process:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
                    img_clahe_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                    img_clahe_gamma_output = adjust_gamma(img_clahe_output, gamma=0.75)
                    img = cv2.cvtColor(img_clahe_gamma_output, cv2.COLOR_BGR2RGB)

                    if img is not None:
                        images.append(np.array(img).reshape(shape[0], shape[1], shape[2]))
    return images


def get_data(inputdir, shape, dataset, validratio, process):
    no_dir = inputdir + '/0/'
    red_dir = inputdir + '/1/'
    orange_dir = inputdir + '/2/'
    green_dir = inputdir + '/3/'
    if dataset == 'simulator':
        no_dir = no_dir + 'simu/'
        red_dir = red_dir + 'simu/'
        orange_dir = orange_dir + 'simu/'
        green_dir = green_dir + 'simu/'

    orange_images = load_images_from_folder(orange_dir, shape, process)
    no_tl_images = load_images_from_folder(no_dir, shape, process)
    green_images = load_images_from_folder(green_dir, shape, process)
    red_images = load_images_from_folder(red_dir, shape, process)

    orange_img_labels = np.zeros(len(orange_images)) * 2
    no_tl_img_labels = np.zeros(len(no_tl_images))
    green_img_labels = np.zeros(len(green_images)) * 3
    red_img_labels = np.ones(len(red_images))

    images = orange_images + red_images + green_images + no_tl_images
    labels = np.concatenate((orange_img_labels, red_img_labels, green_img_labels, no_tl_img_labels))
    X_train, X_valid, y_train, y_valid = train_test_split(images, labels, test_size=validratio, random_state=seed)

    return X_train, X_valid, y_train, y_valid


def print_time(t0, s):
    """Print how much time has been spent
    @param t0: previous timestamp
    @param s: description of this step
    """

    print("%.5f seconds to %s" % ((time.time() - t0), s))
    return time.time()


def train(inputdir, output, shape, epochs, dropout, lr, dataset, validratio, process):
    t0 = time.time()

    model = get_model(dropout, shape)
    print(model)

    adam = Adam(lr=lr)
    model.compile(
        optimizer=adam, loss='binary_crossentropy',
        metrics=['accuracy'])
    t0 = print_time(t0, 'compile the model')

    X_train, X_valid, y_train, y_valid = get_data(inputdir, shape, dataset, validratio, process)
    t0 = print_time(t0, 'load data')

    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    print('X_train shape', X_train.shape)

    # used for simu images
    # datagen = ImageDataGenerator(
    #     featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.05,
    #     zoom_range=.1,
    #     horizontal_flip=True)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.05,
        zoom_range=.1,
        horizontal_flip=True,
        channel_shift_range=0.2)

    datagen.fit(X_train)

    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(X_train,
                                     to_categorical(y_train, num_classes=nb_classes),
                                     batch_size = BATCH_SIZE,
                                     shuffle=True),
                        steps_per_epoch=len(X_train) / BATCH_SIZE,
                        epochs=epochs,
                        validation_data=(X_valid, to_categorical(y_valid, num_classes=nb_classes)))
    t0 = print_time(t0, 'train model')

    model.save(output)
    t0 = print_time(t0, 'save model')

if __name__ == '__main__':
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "i:o:s:e:d:l:t:v:p:")
    except getopt.GetoptError:
        print("SqueezenetTrafficLightModel.py -i <inputfolder> -o <output> -s <shape> " +
              "-t <dataset type(simulator, site)> -e <epochs> -d <dropout> -l <learning rate> " +
              "-v <validation ratio> -p <pre-process>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in "-s":
            shape = ast.literal_eval(arg)
        elif opt in "-e":
            epochs = int(arg)
        elif opt in "-i":
            inputdir = arg
        elif opt in "-o":
            output = arg
        elif opt in "-d":
            dropout = float(arg)
        elif opt in "-l":
            lr = float(arg)
        elif opt in "-t":
            dataset = arg
        elif opt in "-v":
            validratio = float(arg)
        elif opt in "-p":
            process = arg

    output = output + '_' + dataset + '_' + str(shape[0]) + '_' + str(epochs) + '_' + str(dropout) + '_' + str(lr) + '.h5'
    print('Input file is ', inputdir)
    print('Output file is ', output)
    print('Image to be resized ', shape)
    print('Epochs is ', epochs)
    print('Dropout is ', dropout)
    print('Learning rate is ', lr)
    print('Dataset is ', dataset)
    print('Validation ratio is ', validratio)
    print('Pre-process images ', process)

    train(inputdir, output, shape, epochs, dropout, lr, dataset, validratio, process)

#python SqueezenetTrafficLightModel.py -i images -o squeezenet -s "(320, 320, 3)" -t site -e 75 -d .40 -l .0001 -v 0.20 -p true