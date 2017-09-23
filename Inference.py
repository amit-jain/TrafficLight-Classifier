import cv2
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from keras import backend as K
from time import time
from keras.models import load_model
import os
import numpy as np
import sys, getopt, ast

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def infer_from_keras_model(input, output, shape, model):
    print('Inferring from keras model')
    print('')
    model = load_model(model)
    model._make_predict_function()
    graph = tf.get_default_graph()

    # Get the prediction for the image
    with graph.as_default():
        for filename in os.listdir(input):
            if filename != '.DS_Store' and filename != 'simu':
                print(shape)
                img = cv2.imread(os.path.join(input, filename))
                img = cv2.resize(img, (shape[0], shape[1]), interpolation=cv2.INTER_AREA)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
                img_clahe_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                img_clahe_gamma_output = adjust_gamma(img_clahe_output, gamma=0.75)
                img = cv2.cvtColor(img_clahe_gamma_output, cv2.COLOR_BGR2RGB)

                start = time()
                preds = model.predict(img[None,:,:,:])
                print(preds)
                state = np.argmax(preds[0])
                print(filename + ' ' + str(state))
                print('Finished prediction in ms ', (time() - start) * 1000)

                cv2.imwrite(os.path.join(output + '/' + str(state), filename), img)


def infer_from_tf_model(input, output, shape, model):
    print('Inferring from frozen model', model)
    K.clear_session()
    sess = tf.Session()

    if not os.path.isdir(output):
        os.mkdir(output)
    if not os.path.isdir(output + '/0'):
        os.mkdir(output + '/0')
    if not os.path.isdir(output + '/1'):
        os.mkdir(output + '/1')

    with open(model, "rb") as f:
        output_graph_def = graph_pb2.GraphDef()
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")
        image_input = sess.graph.get_tensor_by_name('input_1:0')
        softmax = sess.graph.get_tensor_by_name('output_node0:0')

        for filename in os.listdir(input):
            print(filename)
            if filename != '.DS_Store' and filename != 'simu':
                img = cv2.imread(os.path.join(input, filename))
                img = cv2.resize(img, (shape[0], shape[1]), interpolation=cv2.INTER_AREA)

                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                # img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
                # img_clahe_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                # img_clahe_gamma_output = adjust_gamma(img, gamma=0.75)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                start = time()
                preds = sess.run(softmax, {image_input: img[None, :, :, :]})

                print(preds)
                state = np.argmax(preds)
                print(filename + ' ' + str(state))
                print('Finished prediction in ms ', (time() - start) * 1000)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output + '/' + str(state), filename), img)

if __name__ == '__main__':
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "i:o:s:m:")
    except getopt.GetoptError:
        print
        'Inference.py -i <inputfolder> -o <outputfolder> -s <shape> -m <model>'
        sys.exit(2)
    for opt, arg in opts:
        if opt in "-s":
            shape = arg
        elif opt in "-m":
            model = arg
        elif opt in "-i":
            inputdir = arg
        elif opt in "-o":
            output = arg
    print('Input file is ', inputdir)
    print('Output file is ', output)
    print('Image to be resized ', shape)
    print('Model is ', model)

    if model.endswith('.h5'):
        infer_from_keras_model(inputdir, output, ast.literal_eval(shape), model)
    else:
        infer_from_tf_model(inputdir, output, ast.literal_eval(shape), model)