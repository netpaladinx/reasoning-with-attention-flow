from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from collections import defaultdict

import numpy as np
import tensorflow as tf

default_hparams = tf.contrib.training.HParams(
    n_steps=3,
    n_layers=1,
    n_rnn_units=128,
    n_emb_dims=128,
    batch_size=64,
    max_epochs=10,
    learning_rate=0.001,
    seed=1234,
    memory_nor=True,
    dropout=0.,
    top_k=10,
    clipped_grad=5.
)

EPSILON = 1e-20


class Model(object):
    def __init__(self, kg, hparams):
        self.kg = kg
        self.hparams = hparams

        np.random.seed(hparams.seed)
        tf.set_random_seed(hparams.seed)

        self.tf_graph = tf.Graph()
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        with self.tf_graph.as_default():
            self._build_model()

    def _build_model(self):
        kg = self.kg
        hp = self.hparams

        self._create_input(kg, hp)
        self._create_rnn(kg, hp)
        self._create_attention_and_memory(kg, hp)

        self.prediction_loss = - tf.reduce_sum(self.target * tf.log(tf.maximum(self.prediction, EPSILON)), axis=1)
        self.total_loss = tf.reduce_mean(self.prediction_loss)

        self.n_in_top = tf.reduce_sum(tf.cast(tf.nn.in_top_k(self.prediction, self.tail_input, hp.top_k), tf.float32))
        _, self.idx_pred_topk = tf.nn.top_k(self.prediction, k=kg.n_entities)

        self.optimizer = tf.train.AdamOptimizer(hp.learning_rate)
        grads_and_vars = self.optimizer.compute_gradients(tf.reduce_mean(self.total_loss))
        clippled_grads_and_vars = map(
            lambda (g,v): (tf.clip_by_value(g, -hp.clipped_grad, hp.clipped_grad), v) if g is not None else (g, v),
            grads_and_vars)
        self.train_op = self.optimizer.apply_gradients(clippled_grads_and_vars)

        self.saver = tf.train.Saver(max_to_keep=0)
        self.init_op = tf.global_variables_initializer()

    def _create_input(self, kg, hp):
        self.head_input = tf.placeholder(tf.int32, [None], name='head_input')
        self.tail_input = tf.placeholder(tf.int32, [None], name='tail_input')
        self.query_sequence = tf.placeholder(tf.int32, [None, hp.n_steps], name='query_sequence')

        self.bs = tf.shape(self.head_input)[0]
        self.target = tf.one_hot(self.tail_input, kg.n_entities)  # bs x n_entities
        self.source = tf.one_hot(self.head_input, kg.n_entities)  # bs x n_entities

        self.relation_embeddings = tf.get_variable('relation_embeddings', shape=[kg.n_relations, hp.n_emb_dims],
                                                   initializer=tf.variance_scaling_initializer())
        rnn_inputs = tf.nn.embedding_lookup(self.relation_embeddings, self.query_sequence)  # bs x n_steps x n_emb_dims
        self.rnn_inputs = [tf.reshape(t, [-1, hp.n_emb_dims]) for t in tf.split(rnn_inputs, hp.n_steps, axis=1)]

        self.kg_inputs = {r: tf.sparse_placeholder(tf.float32, name='kg_input_%d' % r) for r in range(kg.n_relations)}

    def _create_rnn(self, kg, hp):
        cells = []
        for i in range(hp.n_layers):
            cells.append(tf.nn.rnn_cell.LSTMCell(hp.n_rnn_units, state_is_tuple=True))
        self.rnn = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        self.rnn_init_state = self.rnn.zero_state(self.bs, tf.float32)

        self.rnn_outputs, self.runn_final_state = tf.nn.static_rnn(self.rnn, self.rnn_inputs, initial_state=self.rnn_init_state)

    def _create_attention_and_memory(self, kg, hp):
        self.attention_relations = []
        self.W = tf.get_variable('W', shape=[hp.n_rnn_units, kg.n_relations], initializer=tf.variance_scaling_initializer())
        self.b = tf.get_variable('b', shape=[kg.n_relations], initializer=tf.variance_scaling_initializer())
        for rnn_out in self.rnn_outputs:  # rnn_out: bs x n_units
            att_rels = tf.nn.softmax(tf.matmul(rnn_out, self.W) + self.b)  # bs x n_relations
            att_rels = tf.split(att_rels, kg.n_relations, axis=1)  # [ bs x 1, bs x 1, ..., bs x 1 ]
            self.attention_relations.append([tf.squeeze(a, axis=1) for a in att_rels])  # append [ bs, bs, ..., bs ]

        self.attention_memories = []
        self.memories = tf.expand_dims(self.source, axis=1)  # bs x 1 x n_entities

        for t in range(hp.n_steps):
            cur_rnn_out = tf.expand_dims(self.rnn_outputs[t], axis=1)  # bs x 1 x n_units
            past_rnn_out = tf.stack(self.rnn_outputs[:t+1], axis=2)  # bs x n_units x (t+1)
            att_mem = tf.nn.softmax(tf.squeeze(tf.matmul(cur_rnn_out, past_rnn_out), axis=1))  # bs x (t+1)
            self.attention_memories.append(att_mem)

            att_mem = tf.expand_dims(self.attention_memories[t], axis=1)  # bs x 1 x (t+1)
            memory_read = tf.squeeze(tf.matmul(att_mem, self.memories), axis=1)  # bs x n_entities

            if t < hp.n_steps - 1:
                results = []
                for r in range(kg.n_relations):
                    relation_mat = self.kg_inputs[r]  # (sparse) n_entity x n_entity
                    product = tf.sparse_tensor_dense_matmul(relation_mat, tf.transpose(memory_read))  # n_entity x bs
                    results.append(tf.transpose(product * self.attention_relations[t][r]))  # bs x n_entity
                summed_result = tf.add_n(results)  # bs x n_entity

                if hp.memory_nor:
                    summed_result /= tf.maximum(EPSILON, tf.reduce_sum(summed_result, axis=1, keepdims=True))
                if hp.dropout > 0.:
                    summed_result = tf.nn.dropout(summed_result, keep_prob=1.-hp.dropout)
                self.memories = tf.concat([self.memories, tf.expand_dims(summed_result, axis=1)], axis=1)
            else:
                self.prediction = memory_read  # bs x n_entities

    def train(self, FLAGS):
        kg = self.kg
        hp = self.hparams

        with tf.Session(graph=self.tf_graph, config=self.tf_config) as sess:
            if FLAGS.checkpoint and os.path.exists(FLAGS.checkpoint):
                self.saver.restore(sess, FLAGS.checkpoint)
            else:
                sess.run(self.init_op)

            n_itrs = 0
            n_queries = 0
            n_episodes = 0
            total_loss = -1.
            n_epochs = 0
            while n_epochs < hp.max_epochs:
                epoch_loss = 0.
                epoch_n_in_top = 0.
                n_batches = 0.
                n_examples = 0.
                for batch in kg.get_train_batch(hp.batch_size, neg_sampling=False):
                    head, query, tail = kg.get_triples_from_batch(batch)
                    kg_inputs = kg.get_filtered_kg(batch)

                    feed = {
                        self.head_input: head,
                        self.tail_input: tail,
                        self.query_sequence: [[q] * (hp.n_steps-1) + [kg.END_RELATION] for q in query]
                    }
                    for r in range(kg.n_relations):
                        feed[self.kg_inputs[r]] = tf.SparseTensorValue(*kg_inputs[r])

                    _, loss, n_in_top = sess.run([self.train_op, self.total_loss, self.n_in_top], feed_dict=feed)

                    epoch_loss += loss
                    epoch_n_in_top += n_in_top
                    n_batches += 1
                    n_examples += np.array(head).shape[0]
                    n_itrs += 1

                    if n_itrs % FLAGS.print_freq_on_itrs == 0 or n_itrs == 1:
                        print('[TRAIN] epoch: %d | batch: %d | example: %d | loss: %.8f | n_in_top: %.4f' %
                            (n_epochs, n_batches, n_examples, loss, n_in_top))

                epoch_loss /= n_batches
                epoch_rate_in_top = epoch_n_in_top / n_examples

                print('[TRAIN] epoch: %d | batch: %d | example: %d | epoch_loss: %.8f | epoch_rate_in_top: %.4f' %
                      (n_epochs, n_batches, n_examples, epoch_loss, epoch_rate_in_top))

                if n_epochs % FLAGS.eval_freq_on_epochs == 0:
                    eval_res, _ = self._evaluate(sess, source='valid')
                    print('[VALID] %s' % eval_res)
                    eval_res, _ = self._evaluate(sess, source='test')
                    print('[TEST] %s' % eval_res)

    def _evaluate(self, sess, source='test'):
        kg = self.kg
        hp = self.hparams

        triples_eval = []
        idx_tail_pred = []

        for batch in kg.get_eval_batch(hp.batch_size, source=source):
            head, query, tail = kg.get_triples_from_batch(batch)
            kg_inputs = kg.get_kg(batch)

            feed = {
                self.head_input: head,
                self.tail_input: tail,
                self.query_sequence: [[q] * (hp.n_steps - 1) + kg.END_RELATION for q in query]
            }
            for r in range(kg.n_relations):
                feed[self.kg_inputs[r]] = tf.SparseTensorValue(*kg_inputs[r])

            loss, predication = sess.run([self.total_loss, self.prediction], feed_dict=feed)

            idx_pred = np.argsort(-predication)
            idx_tail_pred.append(idx_pred)
            triples_eval.append(np.stack([head, query, tail], axis=-1))

        idx_tail_pred = np.concatenate(idx_tail_pred, 0).tolist()
        triples_eval = np.concatenate(triples_eval, 0).tolist()
        _, eval_tail, _ = kg.calc_metrics(triples_eval, idx_tail_pred)

        return eval_tail
