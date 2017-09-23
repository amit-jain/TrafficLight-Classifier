from keras.models import load_model
import tensorflow as tf
import os.path as osp
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import sys, getopt


def convert(input, output):
    num_output = 2
    prefix_output_node_names_of_final_network = 'output_node'

    weight_file_path = osp.join('./', input)

    K.set_learning_phase(0)
    net_model = load_model(weight_file_path)

    pred = [None]*num_output
    pred_node_names = [None]*num_output
    for i in range(num_output):
        pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
        pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
    print('output nodes names are: ', pred_node_names)

    sess = K.get_session()

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, './', output, as_text=False)
    print('saved the constant graph (ready for inference) at: ', osp.join('./', output))

if __name__ == '__main__':
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "i:o:")
    except getopt.GetoptError:
        print
        'CheckGraph.py -i <input> -o <output>'
        sys.exit(2)
    for opt, arg in opts:
        if opt in "-i":
            input = arg
        elif opt in "-o":
            output = arg
    convert(input, output)
