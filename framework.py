"""
framework.py
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf

import implementation as impl

default_hparams = tf.contrib.training.HParams(
    n_dims=64,
    learning_rate=0.0001,
    max_epochs=100,
    batch_size=10,
    epsilon=1e-8,
    using_input_gate=False,
    n_gpus=2
)

class Framework(object):
    def __init__(self, graph, hparams=default_hparams):
        self.hparams = hparams
        self.graph = graph

        self.tf_graph = tf.Graph()
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True

        with self.tf_graph.as_default():
            self._build_model()

    def _build_model(self):
        hp = self.hparams
        g = self.graph

        self.reuse_compute_query_context = [False] * hp.n_gpus
        self.reuse_compute_node_hidden = [False] * hp.n_gpus
        self.reuse_compute_edge_attention = [False] * hp.n_gpus
        self.reuse_compute_message_sent = [False] * hp.n_gpus
        self.reuse_compute_cell_state = [False] * hp.n_gpus
        self.node_emb = None

        # preprocess the inputs

        self.input_pl = tf.placeholder(tf.int32, [None, hp.n_inputs], name='input')  # bs x 2 for (src, dst)
                                                                                     # bs x 3 for (query_type, src, dst)
        self.inputs = tf.split(self.input_pl, hp.n_gpus)

        self._create_emb()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.learning_rate)

        grads, vars, models = [], [], []
        for gpu_id in range(hp.n_gpus):
            with tf.variable_scope('gpu_%d' % gpu_id), tf.device('/gpu:%d' % gpu_id):
                model = self._build_singlegpu_model(gpu_id, self.inputs[gpu_id])
                models.append(model)

                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
                grads_and_vars = self.optimizer.compute_gradients(model['loss'], var_list=var_list)

                grad, var = [], []
                for (g, v) in grads_and_vars:
                    grad.append(self._compress_indexedslices_grad(g))
                    var.append(v)
                grads.append(grad)
                vars.append(var)

        with tf.device('/gpu:0'):
            merged_grads = self._merge_grads(grads)

        train_ops = []
        global_step = tf.train.create_global_step()
        for gpu_id in range(hp.n_gpus):
            with tf.device('/gpu:%d' % gpu_id):
                if gpu_id == 0:
                    train_op = self.optimizer.apply_gradients(zip(merged_grads, vars[gpu_id]), global_step=global_step)
                else:
                    train_op = self.optimizer.apply_gradients(zip(merged_grads, vars[gpu_id]))
                train_ops.append(train_op)
        self.train_op = tf.group(*train_ops)

        self.loss = tf.reduce_mean(tf.stack([model['loss'] for model in models]))
        self.hit_top1 = tf.reduce_mean(tf.stack([model['hit_top1'] for model in models]))
        self.hit_top5 = tf.reduce_mean(tf.stack([model['hit_top5'] for model in models]))
        self.hit_top10 = tf.reduce_mean(tf.stack([model['hit_top10'] for model in models]))
        self.n_hit_top1 = tf.reduce_sum(tf.stack([model['n_hit_top1'] for model in models]))
        self.n_hit_top5 = tf.reduce_sum(tf.stack([model['n_hit_top5'] for model in models]))
        self.n_hit_top10 = tf.reduce_sum(tf.stack([model['n_hit_top10'] for model in models]))

        self.init_op = tf.global_variables_initializer()

        self.saver = tf.train.Saver(var_list=vars[0], max_to_keep=0)

        assign_ops = []
        for var in vars[1:]:
            for v_id, v in enumerate(var):
                assign_ops.append(tf.assign(v, vars[0][v_id]))
        self.sync_op = tf.group(*assign_ops)

    def _build_singlegpu_model(self, gpu_id, inp):
        hp = self.hparams
        g = self.graph

        src, dst, query_type = None, None, None
        if hp.n_inputs == 2:
            src, dst = tf.split(inp, 2, axis=1)  # bs x 1, bs x 1
        elif hp.n_inputs == 3:
            query_type, src, dst = tf.split(inp, 3, axis=1)  # bs x 1, bs x 1, bs x 1

        src = tf.squeeze(src, axis=1)  # bs
        dst = tf.squeeze(dst, axis=1)  # bs
        if query_type is not None:
            query_type = tf.squeeze(query_type, axis=1)  # bs

        qcontext = self._compute_query_context(gpu_id, src, dst, query_type)  # bs x n_dims

        node_attentions = []
        cell_states = []

        node_attentions.append(tf.one_hot(src, g.n_nodes))  # bs x n_nodes
        cell_states.append(tf.expand_dims(node_attentions[-1], axis=-1) * tf.expand_dims(qcontext, axis=1))  # bs x n_nodes x n_dims

        for i in range(hp.flow_length):
            cell_state, node_attention = self._flow(gpu_id, cell_states[-1], node_attentions[-1])
            node_attentions.append(node_attention)
            cell_states.append(cell_state)

        dst_idx_flattened = tf.range(0, hp.batch_size) * g.n_nodes + dst  # bs
        prediction_prob = tf.gather(tf.reshape(node_attentions[-1], [-1]), dst_idx_flattened)  # bs

        loss = tf.reduce_mean(-tf.log(prediction_prob + 1. / g.n_nodes * hp.epsilon))

        hit_top1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(node_attentions[-1], axis=1, output_type=tf.int32), dst), tf.float32))
        hit_top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(node_attentions[-1], dst, 5), tf.float32))
        hit_top10 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(node_attentions[-1], dst, 10), tf.float32))

        n_hit_top1 = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(node_attentions[-1], axis=1, output_type=tf.int32), dst), tf.float32))
        n_hit_top5 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(node_attentions[-1], dst, 5), tf.float32))
        n_hit_top10 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(node_attentions[-1], dst, 10), tf.float32))

        return {'loss': loss, 'hit_top1': hit_top1, 'hit_top5': hit_top5, 'hit_top10': hit_top10,
                'n_hit_top1': n_hit_top1, 'n_hit_top5': n_hit_top5, 'n_hit_top10': n_hit_top10}

    def _compress_indexedslices_grad(self, grad):
        if isinstance(grad, tf.IndexedSlices):
            unique_indices, new_index_position = tf.unique(grad.indices)
            summed_values = tf.unsorted_segment_sum(grad.values, new_index_position, tf.shape(unique_indices)[0])
            grad = tf.IndexedSlices(summed_values, unique_indices)
        return grad

    def _merge_grads(self, grads):
        grads = zip(*grads)
        n_grads = len(grads)
        merged_grads = []
        for grad_id in range(n_grads):
            if isinstance(grads[0][grad_id], tf.IndexedSlices):
                grad = tf.IndexedSlices(tf.concat([g.values for g in grads[grad_id]], axis=0), tf.concat([g.indices for g in grads[grad_id]], axis=0))
                grad = self._compress_indexedslices_grad(grad)
            else:
                grad = tf.add_n(grads[grad_id])
            merged_grads.append(grad)
        return merged_grads

    def _flow(self, gpu_id, cell_state, node_attention):
        ''' cell_state: bs x n_nodes x n_dims
            node_attention: bs x n_nodes
        '''
        hp = self.hparams

        hidden = self._compute_node_hidden(gpu_id, cell_state)  # bs x n_nodes x n_dims
        edge_attention = self._compute_edge_attention(gpu_id, hidden)  # bs x n_edges
        flowing_attention = self._compute_flowing_attention(node_attention, edge_attention)  # bs x n_edges

        message_sent = self._compute_message_sent(gpu_id, hidden)  # bs x n_edges x n_dims
        message_recv = self._compute_message_recv(flowing_attention, message_sent)  # bs x n_edges x n_dims
        message_aggr = self._compute_message_aggr(message_recv)  # bs x n_nodes x n_dims

        cell_state_new = self._compute_cell_state(gpu_id, cell_state, message_aggr, hidden)  # bs x n_nodes x n_dims
        node_attention_new = self._compute_node_attention(flowing_attention)  # bs x n_nodes
        return cell_state_new, node_attention_new

    def _create_emb(self):
        ''' indices: bs
        '''
        hp = self.hparams
        g = self.graph

        with tf.device("/cpu:0"):
            self.node_emb = tf.get_variable('node_emb', shape=[g.n_nodes, hp.n_dims], initializer=tf.truncated_normal_initializer(stddev=0.01))  # n_nodes x n_dims

    def _get_emb(self, indices):
        return tf.gather(self.node_emb, indices)  # bs x n_dims

    def _compute_query_context(self, gpu_id, src, dst, query_type):
        hp = self.hparams

        src_emb = self._get_emb(src)
        dst_emb = self._get_emb(dst)

        qcontext = None
        if hp.query_mode == 'single_query':
            if hp.learning_mode == 'supervised':
                qcontext = impl.compute_query_context_given_src_by_one_layer(src_emb, hp.n_dims, self.reuse_compute_query_context[gpu_id])
            elif hp.learning_mode == 'unsupervised':
                qcontext = impl.compute_query_context_given_src_dst_by_one_layer(src_emb, dst_emb, hp.n_dims, self.reuse_compute_query_context[gpu_id])
        elif hp.query_mode == 'multiple_queries':
            if hp.learning_mode == 'supervised':
                qcontext = impl.compute_query_context_given_src_qtype_by_one_layer(src_emb, query_type, hp.n_dims, hp.n_qtypes, self.reuse_compute_query_context[gpu_id])
            elif hp.learning_mode == 'unsupervised':
                qcontext = impl.compute_query_context_given_src_qtype_dst_by_one_layer(src_emb, query_type, dst_emb, hp.n_dims, hp.n_qtypes, self.reuse_compute_query_context[gpu_id])
        else:
            raise ValueError('_compute_query_context')

        self.reuse_compute_query_context[gpu_id] = True
        return qcontext  # bs x n_dims

    def _compute_node_hidden(self, gpu_id, cell_state):
        hp = self.hparams
        hidden = impl.compute_node_hidden_by_one_layer(cell_state, self.node_emb, hp.n_dims, self.reuse_compute_node_hidden[gpu_id])  # bs x n_nodes x n_dims
        self.reuse_compute_node_hidden[gpu_id] = True
        return hidden

    def _compute_edge_attention(self, gpu_id, hidden):
        hp = self.hparams
        edge_attention = impl.compute_edge_attention_by_one_layer(hidden, self.graph, hp.n_dims, self.reuse_compute_edge_attention[gpu_id])  # bs x n_edges
        self.reuse_compute_edge_attention[gpu_id] = True
        return edge_attention

    def _compute_flowing_attention(self, node_attention, edge_attention):
        ''' node_attention: bs x n_nodes
            edge_attention: bs x n_edges
        '''
        node_attention = tf.transpose(tf.gather(tf.transpose(node_attention), self.graph.folded_edges('v1_list')))  # bs x n_edges
        flowing_attention = node_attention * edge_attention  # bs x n_edges
        return flowing_attention

    def _compute_message_sent(self, gpu_id, hidden):
        hp = self.hparams
        message_sent = impl.compute_message_sent_by_one_layer(hidden, self.graph, hp.n_dims, self.reuse_compute_message_sent[gpu_id])  # bs x n_edges x n_dims
        self.reuse_compute_message_sent[gpu_id] = True
        return message_sent

    def _compute_message_recv(self, flowing_attention, message_sent):
        message_recv = tf.expand_dims(flowing_attention, axis=2) * message_sent  # bs x n_edges x n_dims
        return message_recv

    def _compute_message_aggr(self, message_recv):
        ''' message_recv: bs x n_edges x n_dims
        '''
        message_recv = tf.transpose(message_recv, perm=[1, 0, 2])  # n_edges x bs x n_dims
        message_aggr = tf.unsorted_segment_sum(message_recv, self.graph.folded_edges('v2_list'), self.graph.n_nodes)  # n_nodes x bs x n_dims
        message_aggr = tf.transpose(message_aggr, perm=[1, 0, 2])  # bs x n_nodes x n_dims
        return message_aggr

    def _compute_node_attention(self, flowing_attention):
        node_attention = tf.transpose(tf.unsorted_segment_sum(tf.transpose(flowing_attention), self.graph.folded_edges('v2_list'), self.graph.n_nodes))  # bs x n_nodes
        return node_attention

    def _compute_cell_state(self, gpu_id, cell_state, message_aggr, hidden):
        hp = self.hparams
        if hp.using_input_gate:
            cell_state = impl.compute_cell_state_with_input_gate(cell_state, message_aggr, hidden, hp.n_dims, self.reuse_compute_cell_state[gpu_id])  # bs x n_nodes x n_dims
            self.reuse_compute_cell_state[gpu_id] = True
        else:
            cell_state = impl.compute_cell_state_by_adding(cell_state, message_aggr)  # bs x n_nodes x n_dims
        return cell_state

    def train(self, FLAGS):
        hp = self.hparams
        g = self.graph
        batch_size = hp.batch_size * hp.n_gpus

        with tf.Session(graph=self.tf_graph, config=self.tf_config) as sess:
            if FLAGS.checkpoint:
                self.saver.restore(sess, FLAGS.checkpoint)
                sess.run(self.sync_op)
            else:
                sess.run(self.init_op)

            n_itrs = 0
            avg_hit_top1 = -1.
            avg_hit_top5 = -1.
            avg_hit_top10 = -1.
            for n_epochs in range(hp.max_epochs):
                n_batches = g.n_batches(batch_size)
                for i in range(n_batches):
                    batch = g.get_batch(i, batch_size, with_path_id=hp.with_path_id)
                    _, loss, hit_top1, hit_top5, hit_top10 = \
                        sess.run([self.train_op, self.loss, self.hit_top1, self.hit_top5, self.hit_top10],
                                 feed_dict={self.input_pl: batch})

                    avg_hit_top1 = hit_top1 if avg_hit_top1 == -1 else 0.99*avg_hit_top1 + 0.01*hit_top1
                    avg_hit_top5 = hit_top5 if avg_hit_top5 == -1 else 0.99 * avg_hit_top5 + 0.01 * hit_top5
                    avg_hit_top10 = hit_top10 if avg_hit_top10 == -1 else 0.99 * avg_hit_top10 + 0.01 * hit_top10

                    n_itrs += 1
                    if n_itrs % FLAGS.print_freq == 0:
                        print('[TRAIN] n_epochs: %d | n_itrs: %d | loss: %.8f | hit_top1: %.8f (%.8f) | hit_top5: %.8f (%.8f) | hit_top10: %.8f (%.8f)' %
                              (n_epochs, n_itrs, loss, hit_top1, avg_hit_top1, hit_top5, avg_hit_top5, hit_top10, avg_hit_top10))

                n_batches = g.n_batches(batch_size, target='test')
                sum_hit_top1 = 0.
                sum_hit_top5 = 0.
                sum_hit_top10 = 0.
                sum_loss = 0.
                for i in range(n_batches):
                    batch = g.get_batch(i, batch_size, target='test', with_path_id=hp.with_path_id)
                    loss, n_hit_top1, n_hit_top5, n_hit_top10 = \
                        sess.run([self.loss, self.n_hit_top1, self.n_hit_top5, self.n_hit_top10],
                                 feed_dict={self.input_pl: batch})

                    sum_hit_top1 += n_hit_top1
                    sum_hit_top5 += n_hit_top5
                    sum_hit_top10 += n_hit_top10
                    sum_loss += loss

                test_hit_top1 = sum_hit_top1 / g.n_test
                test_hit_top5 = sum_hit_top5 / g.n_test
                test_hit_top10 = sum_hit_top10 / g.n_test
                test_loss = sum_loss / n_batches
                print('[TEST] n_epochs: %d | n_itrs: %d | loss: %.8f | hit_top1: %.8f | hit_top5: %.8f | hit_top10: %.8f' %
                      (n_epochs, n_itrs, test_loss, test_hit_top1, test_hit_top5, test_hit_top10))


