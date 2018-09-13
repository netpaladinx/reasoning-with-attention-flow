"""
implementation.py
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf


def compute_query_context_given_src_by_one_layer(src_emb, n_dims, reuse=None):
    ''' src_emb: bs x n_dims
    '''
    with tf.variable_scope('query_context', reuse=reuse):
        weight_src = tf.get_variable('weight_src', shape=[n_dims, n_dims], initializer=tf.initializers.identity())  # n_dims x n_dims
        bias = tf.get_variable('bias', shape=[n_dims], initializer=tf.zeros_initializer())  # n_dims
        context = tf.tanh(tf.matmul(src_emb, weight_src) + bias)  # bs x n_dims
        return context

def compute_query_context_given_src_dst_by_one_layer(src_emb, dst_emb, n_dims, reuse=None):
    ''' src_emb: bs x n_dims
        dst_emb: bs x n_dims
    '''
    with tf.variable_scope('query_context', reuse=reuse):
        weight_src = tf.get_variable('weight_src', shape=[n_dims, n_dims], initializer=tf.initializers.identity())  # n_dims x n_dims
        weight_dst = tf.get_variable('weight_dst', shape=[n_dims, n_dims], initializer=tf.initializers.identity())  # n_dims x n_dims
        bias = tf.get_variable('bias', shape=[n_dims], initializer=tf.zeros_initializer())  # n_dims
        context = tf.tanh(tf.matmul(src_emb, weight_src) + tf.matmul(dst_emb, weight_dst) + bias)  # bs x n_dims
        return context

def compute_query_context_given_src_qtype_by_one_layer(src_emb, query_type, n_dims, n_qtypes, reuse=None):
    ''' src_emb: bs x n_dims
        query_type: bs
    '''
    with tf.variable_scope('query_context', reuse=reuse):
        weight_src_set = tf.get_variable('weight_src_set', shape=[n_qtypes, n_dims, n_dims],
                                         initializer=tf.constant_initializer(np.tile(np.identity(n_dims), (n_qtypes, 1, 1))))  # n_qtypes x n_dims x n_dims
        bias_set = tf.get_variable('bias_set', shape=[n_qtypes, n_dims], initializer=tf.zeros_initializer())  # n_qtypes x n_dims

        weight_src_i = tf.gather(weight_src_set, query_type)  # bs x n_dims x n_dims
        bias_i = tf.gather(bias_set, query_type)  # bs x n_dims

        weight_src = tf.get_variable('weight_src', shape=[n_dims, n_dims], initializer=tf.initializers.identity())  # n_dims x n_dims
        bias = tf.get_variable('bias', shape=[n_dims], initializer=tf.zeros_initializer())  # n_dims

        src_emb = tf.expand_dims(src_emb, axis=1)  # bs x 1 x n_dims
        w = 0.5 * weight_src + 0.5 * weight_src_i  # bs x n_dims x n_dims
        b = 0.5 * bias + 0.5 * bias_i  # bs x n_dims
        context = tf.tanh(tf.squeeze(tf.matmul(src_emb, w), axis=1) + b)  # bs x n_dims
        return context

def compute_query_context_given_src_qtype_dst_by_one_layer(src_emb, query_type, dst_emb, n_dims, n_qtypes, reuse=None):
    ''' src_emb: bs x n_dims
        dst_emb: bs x n_dims
        query_type: bs
    '''
    with tf.variable_scope('query_context', reuse=reuse):
        weight_src_set = tf.get_variable('weight_src_set', shape=[n_qtypes, n_dims, n_dims],
                                         initializer=tf.constant_initializer(np.tile(np.identity(n_dims), (n_qtypes, 1, 1))))  # n_qtypes x n_dims x n_dims
        weight_dst_set = tf.get_variable('weight_dst_set', shape=[n_qtypes, n_dims, n_dims],
                                         initializer=tf.constant_initializer(np.tile(np.identity(n_dims), (n_qtypes, 1, 1))))  # n_qtypes x n_dims x n_dims
        bias_set = tf.get_variable('bias_set', shape=[n_qtypes, n_dims], initializer=tf.zeros_initializer())  # n_qtypes x n_dims

        weight_src_i = tf.gather(weight_src_set, query_type)  # bs x n_dims x n_dims
        weight_dst_i = tf.gather(weight_dst_set, query_type)  # bs x n_dims x n_dims
        bias_i = tf.gather(bias_set, query_type)  # bs x n_dims

        weight_src = tf.get_variable('weight_src', shape=[n_dims, n_dims], initializer=tf.initializers.identity())  # n_dims x n_dims
        weight_dst = tf.get_variable('weight_dst', shape=[n_dims, n_dims], initializer=tf.initializers.identity())  # n_dims x n_dims
        bias = tf.get_variable('bias', shape=[n_dims], initializer=tf.zeros_initializer())  # n_dims

        src_emb = tf.expand_dims(src_emb, axis=1)  # bs x 1 x n_dims
        dst_emb = tf.expand_dims(dst_emb, axis=1)  # bs x 1 x n_dims
        w1 = 0.5 * weight_src + 0.5 * weight_src_i  # bs x n_dims x n_dims
        w2 = 0.5 * weight_dst + 0.5 * weight_dst_i  # bs x n_dims x n_dims
        b = 0.5 * bias + 0.5 * bias_i  # bs x n_dims
        context = tf.tanh(tf.squeeze(tf.matmul(src_emb, w1), axis=1) + tf.squeeze(tf.matmul(dst_emb, w2), axis=1) + b)  # bs x n_dims
        return context

def compute_node_hidden_by_one_layer(cell_state, node_emb, n_dims, reuse=None):
    ''' cell_state: bs x n_nodes x n_dims
        node_emb: n_nodes x n_dims
    '''
    with tf.variable_scope('node_hidden', reuse=reuse):
        weight = tf.get_variable('weight', shape=[n_dims, n_dims], initializer=tf.initializers.identity())  # n_dims x n_dims
        hidden = tf.tanh(tf.tensordot(cell_state, weight, [[2], [0]]) + node_emb)  # bs x n_nodes x n_dims
        return hidden

def compute_edge_attention_by_one_layer(hidden, graph, n_dims, reuse=None):
    ''' hidden: bs x n_nodes x n_dims
    '''
    with tf.variable_scope('edge_attention', reuse=reuse):
        weight_rel_set = tf.get_variable('weight_rel_set', shape=[graph.n_relations, n_dims, n_dims],
                                         initializer=tf.constant_initializer(np.tile(np.identity(n_dims), (graph.n_relations, 1, 1))))  # n_relations x n_dims x n_dims
        weight_comb = tf.sparse_segment_sum(weight_rel_set, graph.unfolded_edges('rel_id_list'), graph.unfolded_edges('edge_id_list'))  # n_edges x n_dims x n_dims

        hidden_emb = tf.transpose(hidden, perm=[1, 0, 2])  # n_nodes x bs x n_dims
        hidden_v1 = tf.gather(hidden_emb, graph.folded_edges('v1_list'))  # n_edges x bs x n_dims
        hidden_v2 = tf.gather(hidden_emb, graph.folded_edges('v2_list'))  # n_edges x bs x n_dims

        edge_attention = tf.nn.softplus(tf.reduce_sum(tf.matmul(hidden_v1, weight_comb) * hidden_v2, axis=2))  # n_edges x bs
        edge_attention_sum = tf.segment_sum(edge_attention, graph.folded_edges('v1_list'))  # n_nodes x bs
        edge_attention_sum = tf.gather(edge_attention_sum, graph.folded_edges('v1_list'))  # n_edges x bs
        edge_attention_nor = tf.transpose(edge_attention / edge_attention_sum)  # bs x n_edges

        return edge_attention_nor

def compute_message_sent_by_one_layer(hidden, graph, n_dims, reuse=None):
    ''' hidden: bs x n_nodes x n_dims
    '''
    with tf.variable_scope('message', reuse=reuse):
        weight_rel_set = tf.get_variable('weight_rel_set', shape=[graph.n_relations, n_dims, n_dims],
                                         initializer=tf.constant_initializer(np.tile(np.identity(n_dims), (graph.n_relations, 1, 1))))  # n_relations x n_dims x n_dims
        bias_rel_set = tf.get_variable('bias_rel_set', shape=[graph.n_relations, n_dims], initializer=tf.zeros_initializer())  # n_relations x n_dims

        weight_comb = tf.sparse_segment_sum(weight_rel_set, graph.unfolded_edges('rel_id_list'), graph.unfolded_edges('edge_id_list'))  # n_edges x n_dims x n_dims
        bias_comb = tf.sparse_segment_sum(bias_rel_set, graph.unfolded_edges('rel_id_list'), graph.unfolded_edges('edge_id_list'))  # n_edges x n_dims

        hidden_emb = tf.transpose(hidden, perm=[1, 0, 2])  # n_nodes x bs x n_dims
        hidden_v1 = tf.gather(hidden_emb, graph.folded_edges('v1_list'))  # n_edges x bs x n_dims

        message = tf.tanh(tf.matmul(hidden_v1, weight_comb) + tf.expand_dims(bias_comb, axis=1))  # n_edges x bs x n_dims
        message = tf.transpose(message, perm=[1, 0, 2])  # bs x n_edges x n_dims
        return message

def compute_cell_state_by_adding(cell_state_prev, message_aggr):
    ''' cell_state_prev: bs x n_nodes x n_dims
        message_aggr: bs x n_nodes x n_dims
    '''
    cell_state = cell_state_prev + message_aggr  # bs x n_nodes x n_dims
    return cell_state

def compute_cell_state_with_input_gate(cell_state_prev, message_aggr, hidden, n_dims, reuse=None):
    with tf.variable_scope('cell_state', reuse=reuse):
        weight = tf.get_variable('weight_inp_gate', shape=[2*n_dims, n_dims], initializer=tf.variance_scaling_initializer())  # (2*n_dims) x n_dims
        bias = tf.get_variable('bias_inp_gate', shape=[n_dims], initializer=tf.zeros_initializer())  # n_dims

        hidden_message = tf.concat([hidden, message_aggr], 2)  # bs x n_nodes x (2*n_dims)
        input_gate = tf.sigmoid(tf.matmul(hidden_message, weight) + bias)  # bs x n_nodes x n_dims
        cell_state = cell_state_prev + message_aggr * input_gate  # bs x n_nodes x n_dims
        return cell_state