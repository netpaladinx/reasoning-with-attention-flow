from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import tensorflow as tf

import knowledge_graph as klgraph


def run_minerva_on_fb15k237():
    from baselines import minerva

    tf.flags.DEFINE_string('pretrained_relation_embeddings', None, '')
    tf.flags.DEFINE_string('pretrained_entity_embeddings', None, '')
    tf.flags.DEFINE_integer('print_freq_on_itrs', 20, '')
    tf.flags.DEFINE_integer('eval_freq_on_epochs', 1, '')
    tf.flags.DEFINE_string('checkpoint', None, '')
    FLAGS = tf.flags.FLAGS
    hparams = tf.contrib.training.HParams(**minerva.default_hparams.values())
    kg = klgraph.KG('FB15K-237', './data/FB15K-237', build_graph=True, max_actions=hparams.max_actions,
                    pad_entity='PAD', pad_relation='PAD', dummy_start_relation='DUMMY_START', no_op_relation='NO_OP')
    model = minerva.Model(kg, hparams)
    model.train(FLAGS)


def run_neural_lp_on_fb15k237(dataset):
    from baselines import neural_lp

    tf.flags.DEFINE_integer('print_freq_on_itrs', 10, '')
    tf.flags.DEFINE_integer('eval_freq_on_epochs', 1, '')
    tf.flags.DEFINE_string('checkpoint', None, '')
    FLAGS = tf.flags.FLAGS
    hparams = tf.contrib.training.HParams(**neural_lp.default_hparams.values())

    kg = klgraph.KG(dataset, os.path.join('kbcompletion/data', dataset), 'neural_lp', build_graph=True, end_relation='END')
    model = neural_lp.Model(kg, hparams)
    model.train(FLAGS)

if __name__ == '__main__':
    # MINERVA
    #run_minerva_on_fb15k237()

    # Neural LP
    run_neural_lp_on_fb15k237('FB15K-237c-inv-fullKG')
