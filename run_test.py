from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

import test.task as task
import framework as fw

if __name__ == '__main__':
    tf.flags.DEFINE_integer("print_freq", 100, "Frequency of printing")
    FLAGS = tf.flags.FLAGS

    hparams = fw.default_hparams
    hparams.add_hparam('n_inputs', 2)
    hparams.add_hparam('with_path_id', False)
    hparams.add_hparam('query_mode', 'single_query')
    hparams.add_hparam('learning_mode', 'supervised')
    hparams.add_hparam('flow_length', 4)
    maze = task.IntegerMaze()
    model = fw.Framework(maze)
    model.train(FLAGS)