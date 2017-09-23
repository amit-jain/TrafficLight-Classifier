import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from tensorflow.python.framework import graph_io
import sys, getopt

def check_ops_optimized_tf_graph(graph_file, use_xla=False):
    jit_level = 0
    config = tf.ConfigProto()
    if use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        ops = sess.graph.get_operations()
        return sess.graph, ops


def check_ops_original_model(model_file, use_xla=False):
    model = load_model(model_file)
    sess = K.get_session()
    graph = tf.get_default_graph()
    output_graph_name = 'original_' + model_file + '.pb'

    graph_io.write_graph(graph, "./", output_graph_name, as_text=False)
    graph_file = './' + output_graph_name

    jit_level = 0
    config = tf.ConfigProto()
    if use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        ops = sess.graph.get_operations()
        return sess.graph, ops

if __name__ == '__main__':
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "m:")
    except getopt.GetoptError:
        print
        'CheckGraph.py -m <model>'
        sys.exit(2)
    for opt, arg in opts:
        if opt in "-m":
            model = arg

    if model.endswith('.pb'):
        sess, base_ops = check_ops_optimized_tf_graph(model)
        print(len(base_ops))
    else:
        sess, frozen_ops = check_ops_original_model(model)
        print(len(frozen_ops))
