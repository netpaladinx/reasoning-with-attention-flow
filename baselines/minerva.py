from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from collections import defaultdict

import numpy as np
import tensorflow as tf

default_hparams = tf.contrib.training.HParams(
    n_hidden_dims=50,
    n_emb_dims=50,
    batch_size=256,
    n_rollouts=20,  # real batch_size = batch_size * n_rollouts
    n_eval_rollouts=100,
    max_epochs=1000,
    learning_rate=0.001,
    use_entity_embs=False,
    train_entity_embs=False,
    train_relation_embs=True,
    n_lstm_layers=1,
    max_actions=200,
    n_steps=3,
    beta=0.02,  # for decaying the entroy regularization loss
    decay_steps=200,
    decay_rate=0.9,
    lmbda=0.02,  # for baseline update
    grad_clip_norm=5.,
    pos_reward=1.,
    neg_reward=0.,
    gamma=1, # for cummulative discounted reward
    eval_pool='max'
)

class Model(object):
    def __init__(self, kg, hparams):
        self.kg = kg
        self.hparams = hparams

        self.tf_graph = tf.Graph()
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        with self.tf_graph.as_default():
            self._build_model()

    def _build_model(self):
        kg = self.kg
        hp = self.hparams

        self.query_relation = tf.placeholder(tf.int32, [None], name='query_relation')  # bs
        self.cum_discounted_rewards = tf.placeholder(tf.float32, [None, hp.n_steps], name='cum_discounted_rewards')

        self.bs = tf.shape(self.query_relation)[0]

        self.candidate_relations_sequence = []
        self.candidate_entities_sequence = []
        self.choosen_entity_sequence = []
        for t in range(hp.n_steps):
            candidate_relations = tf.placeholder(tf.int32, [None, hp.max_actions], name='candidate_relations_%d' % t)  # bs x max_actions
            candidate_entities = tf.placeholder(tf.int32, [None, hp.max_actions], name='candidate_entities_%d' % t)
            choosen_entity = tf.placeholder(tf.int32, [None], name='choosen_entity_%d' % t)  # bs
            self.candidate_relations_sequence.append(candidate_relations)
            self.candidate_entities_sequence.append(candidate_entities)
            self.choosen_entity_sequence.append(choosen_entity)

        self._create_agent(kg, hp)
        self._call_agent(kg, hp)

        self._calc_reinfoce_loss(kg, hp)
        self._create_backprop(kg, hp)

        self._create_eval_graph(kg, hp)

        self.saver = tf.train.Saver(max_to_keep=2)

        self.init_op = tf.global_variables_initializer()

    def _create_agent(self, kg, hp):
        with tf.variable_scope('embeddings'):
            self.entity_embeddings = tf.get_variable('entity_embeddings', shape=[kg.n_entities, 2 * hp.n_emb_dims],
                                                     initializer=tf.variance_scaling_initializer() if hp.use_entity_embs else tf.zeros_initializer(),
                                                     trainable=hp.train_entity_embs)  # n_entities x (2*n_emb_dims)
            self.relation_embeddings = tf.get_variable('relation_embeddings', shape=[kg.n_entities, 2 * hp.n_emb_dims],
                                                       initializer=tf.variance_scaling_initializer(),
                                                       trainable=hp.train_relation_embs)  # n_entities x (2*n_emb_dims)

        with tf.variable_scope('history_rnn'):
            self.cells = []
            self.n_in = 4 if hp.use_entity_embs else 2
            for _ in range(hp.n_lstm_layers):
                self.cells.append(tf.nn.rnn_cell.LSTMCell(self.n_in * hp.n_hidden_dims, use_peepholes=True, state_is_tuple=True))  # n_units = n_in*n_hidden_dims
            self.history = tf.nn.rnn_cell.MultiRNNCell(self.cells, state_is_tuple=True)

    def _call_agent(self, kg, hp):
        with tf.variable_scope('rollout') as scope:
            query_relation_emb = tf.nn.embedding_lookup(self.relation_embeddings, self.query_relation)  # bs x (2*n_emb_dims)
            prev_lstm_state = self.history.zero_state(self.bs, tf.float32)  # bs x n_units
            prev_relation = tf.ones([self.bs], dtype=tf.int32) * kg.DUMMY_START_RELATION  # bs

            self.per_step_loss = []
            self.per_step_logits = []
            self.per_step_action = []

            for t in range(hp.n_steps):
                if t > 0:
                    scope.reuse_variables()

                loss, new_lstm_state, logits, next_action, next_relation = \
                    self._step(self.candidate_relations_sequence[t], self.candidate_entities_sequence[t], prev_lstm_state, prev_relation,
                               self.choosen_entity_sequence[t], query_relation_emb, kg, hp)

                self.per_step_loss.append(loss)
                self.per_step_logits.append(logits)
                self.per_step_action.append(next_action)

                prev_lstm_state = new_lstm_state
                prev_relation = next_relation

    def _step(self, candidate_relations, candidate_entities, prev_lstm_state, prev_relation, current_entity,
              query_relation_emb, kg, hp):
        prev_action_emb = self._action_encoder(prev_relation, current_entity, kg, hp)  # bs x (n_in*n_emb_dims)
        history_hidden, new_lstm_state = self.history(prev_action_emb, prev_lstm_state)  # bs x n_units, bs x n_units

        entity_emb = tf.nn.embedding_lookup(self.entity_embeddings, current_entity)  # bs x (2*n_emb_dims)
        if hp.use_entity_embs:
            policy_state = tf.concat([history_hidden, entity_emb], axis=-1)  # bs x (n_units+2*n_emb_dims)
        else:
            policy_state = history_hidden  # bs x n_units
        policy_state = tf.concat([policy_state, query_relation_emb], axis=-1)

        candidate_actions_emb = self._action_encoder(candidate_relations, candidate_entities, kg,
                                                     hp)  # max_actions x (n_in*n_emb_dims)

        policy_out = self._policy(policy_state, kg, hp)  # bs x (n_in*n_emb_dims)
        scores = tf.reduce_sum(tf.multiply(tf.expand_dims(policy_out, axis=1), candidate_actions_emb),
                               axis=2)  # bs x max_actions
        scores = self._mask_out_pad_relations(scores, candidate_relations, kg, hp)

        next_action = tf.squeeze(tf.to_int32(tf.multinomial(scores, 1)), axis=1)  # bs
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=next_action)  # bs
        logits = tf.nn.log_softmax(scores)  # bs x max_actions

        next_relation = tf.gather_nd(candidate_relations, tf.stack([tf.range(self.bs), next_action], axis=1))
        return loss, new_lstm_state, logits, next_action, next_relation

    def _action_encoder(self, relation, entity, kg, hp):
        relation_emb = tf.nn.embedding_lookup(self.relation_embeddings, relation)
        entity_emb = tf.nn.embedding_lookup(self.entity_embeddings, entity)
        if hp.use_entity_embs:
            action_emb = tf.concat([relation_emb, entity_emb], axis=-1)
        else:
            action_emb = relation_emb
        return action_emb

    def _policy(self, state, kg, hp):
        with tf.variable_scope('policy'):
            hidden = tf.layers.dense(state, 4 * hp.n_hidden_dims, activation=tf.nn.relu)
            out = tf.layers.dense(hidden, self.n_in * hp.n_emb_dims, activation=tf.nn.relu)
        return out

    def _mask_out_pad_relations(self, scores, candidate_relations, kg, hp):
        mask = tf.equal(candidate_relations, tf.ones_like(candidate_relations, dtype=tf.int32) * kg.PAD_RELATION)
        scores = tf.where(mask, tf.ones_like(scores) * -99999., scores)
        return scores

    def _calc_reinfoce_loss(self, kg, hp):
        self._create_baseline()
        final_reward = self.cum_discounted_rewards - self.baseline  # bs x n_steps
        reward_mean, reward_var = tf.nn.moments(final_reward, axes=[0, 1])
        reward_std = tf.sqrt(reward_var) + 1e-6
        final_reward = tf.div(final_reward - reward_mean, reward_std)  # bs x n_steps

        loss = tf.stack(self.per_step_loss, axis=1)  # bs x n_steps
        loss = tf.multiply(loss, final_reward)  # bs x n_steps
        self.loss_before_reg = loss  # bs x n_steps

        self.global_step = tf.train.create_global_step()
        self.decaying_beta = tf.train.exponential_decay(hp.beta, self.global_step, hp.decay_steps, hp.decay_rate, staircase=False)
        self.entropy_reg_loss = - self._calc_entroy_reg_loss(self.per_step_logits)

        self.total_loss = tf.reduce_mean(loss) + self.decaying_beta * self.entropy_reg_loss  # 1

    def _create_baseline(self):
        with tf.variable_scope('baseline'):
            self.baseline = tf.get_variable('baseline', shape=[], initializer=tf.constant_initializer(0.), trainable=False)

    def _calc_entroy_reg_loss(self, per_step_logits):
        all_logits = tf.stack(per_step_logits, axis=2)  # bs x max_actions x n_steps
        entropy = - tf.reduce_mean(tf.reduce_sum(tf.exp(all_logits) * all_logits, axis=1))  # 1
        return entropy

    def _create_backprop(self, kg, hp):
        update_op = self._update_baseline(self.cum_discounted_rewards, kg, hp)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self.total_loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, hp.grad_clip_norm)

        self.optimizer = tf.train.AdamOptimizer(hp.learning_rate)
        train_op = self.optimizer.apply_gradients(zip(grads, tvars))

        with tf.control_dependencies([train_op, update_op]):
            self.train_op = tf.constant(0)

    def _update_baseline(self, cum_discounted_rewards, kg, hp):
        target = tf.reduce_mean(cum_discounted_rewards)
        update_op = tf.assign(self.baseline, self.baseline * (1-hp.lmbda) + target * hp.lmbda)
        return update_op

    def _create_eval_graph(self, kg, hp):
        self.prev_lstm_state = tf.placeholder(tf.float32, [hp.n_lstm_layers, 2, None, self.n_in * hp.n_hidden_dims], name='prev_lstm_state')
        layer_lstm_state = tf.unstack(self.prev_lstm_state, hp.n_lstm_layers)
        formated_lstm_state = [tf.unstack(s, 2) for s in layer_lstm_state]

        self.prev_relation = tf.placeholder(tf.int32, [None], name='prev_relation')
        self.current_entity = tf.placeholder(tf.int32, [None], name='current_entity')
        self.next_relations = tf.placeholder(tf.int32, [None, hp.max_actions], name='next_relations')
        self.next_entities = tf.placeholder(tf.int32, [None, hp.max_actions], name='next_entities')

        self.query_relation_emb = tf.nn.embedding_lookup(self.relation_embeddings, self.query_relation)

        with tf.variable_scope('rollout') as scope:
            scope.reuse_variables()
            self.eval_loss, eval_new_lstm_state, self.eval_logits, self.eval_next_action, self.eval_next_relation = \
                self._step(self.next_relations, self.next_entities, formated_lstm_state, self.prev_relation, self.current_entity, self.query_relation_emb, kg, hp)
            self.eval_lstm_state = tf.stack(eval_new_lstm_state)

    def train(self, FLAGS):
        kg = self.kg
        hp = self.hparams

        fetches, feeds, feed_dict = self._episode_partial_run_setup(kg, hp)

        with tf.Session(graph=self.tf_graph, config=self.tf_config) as sess:
            if FLAGS.checkpoint and os.path.exists(FLAGS.checkpoint):
                self.saver.restore(sess, FLAGS.checkpoint)
            else:
                sess.run(self.init_op)

            self._load_pretrained_embeddings(sess, FLAGS)

            n_itrs = 0
            n_queries = 0
            n_episodes = 0
            total_loss = -1.
            n_epochs = 0
            while n_epochs < hp.max_epochs:
                n_epochs += 1

                for episodes in kg.get_train_episodes(hp.batch_size, hp.n_rollouts, hp.n_steps):
                    rollout, step, state = episodes
                    batch_size = rollout['batch_size']
                    n_rollouts = rollout['n_rollouts']

                    n_itrs += 1
                    n_queries += batch_size
                    n_episodes += n_rollouts

                    h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
                    feed_dict[0][self.query_relation] = rollout['query_relation']

                    for i in range(hp.n_steps):
                        feed_dict[i][self.candidate_relations_sequence[i]] = state['next_relations']
                        feed_dict[i][self.candidate_entities_sequence[i]] = state['next_entities']
                        feed_dict[i][self.choosen_entity_sequence[i]] = state['current_entity']

                        cur_step_loss, cur_step_logits, next_action = \
                            sess.partial_run(h, [self.per_step_loss[i], self.per_step_logits[i], self.per_step_action[i]], feed_dict=feed_dict[i])

                        step, state = kg.get_next_state(next_action, step, state, rollout)

                    final_reward = kg.get_reward(state, rollout, hp.pos_reward, hp.neg_reward)
                    cum_discounted_rewards = self._calc_cum_discounted_rewards(final_reward, kg, hp)

                    batch_total_loss, _ = sess.partial_run(h, [self.total_loss, self.train_op], feed_dict={self.cum_discounted_rewards: cum_discounted_rewards})
                    total_loss = batch_total_loss if total_loss == -1. else 0.99 * total_loss + 0.01 * batch_total_loss
                    sum_rewards = np.sum(final_reward)
                    avg_correct_per_query = np.sum(np.sum(np.reshape(final_reward, (batch_size, n_rollouts)), axis=1) > 0) / batch_size

                    if n_itrs % FLAGS.print_freq_on_itrs == 0:
                        print('[TRAIN] itr: %d | query: %d | episode: %d | loss: %.8f | sum_rewards (num_hits): %.4f | avg_correct_per_query: %.4f' %
                              (n_itrs, n_queries, n_episodes, total_loss, sum_rewards, avg_correct_per_query))

                if n_epochs % FLAGS.eval_freq_on_epochs == 0:
                    eval_res, _ = self._evaluate(sess, source='valid')
                    print('[VALID] %s' % eval_res)
                    eval_res, _ = self._evaluate(sess, source='test')
                    print('[TEST] %s' % eval_res)

    def _episode_partial_run_setup(self, kg, hp):
        fetches = self.per_step_loss + self.per_step_action + self.per_step_logits + [self.total_loss, self.train_op]
        feeds = self.candidate_relations_sequence + self.candidate_entities_sequence + self.choosen_entity_sequence + [self.query_relation, self.cum_discounted_rewards]

        feed_dict = [dict() for _ in range(hp.n_steps)]

        feed_dict[0][self.query_relation] = None
        for i in range(hp.n_steps):
            feed_dict[i][self.candidate_relations_sequence[i]] = None
            feed_dict[i][self.candidate_entities_sequence[i]] = None
            feed_dict[i][self.choosen_entity_sequence[i]] = None

        return fetches, feeds, feed_dict

    def _load_pretrained_embeddings(self, sess, FLAGS):
        if FLAGS.pretrained_relation_embeddings and os.path.exists(FLAGS.pretrained_relation_embeddings):
            embeddings = np.loadtxt(FLAGS.pretrained_relation_embeddings)
            self.relation_embeddings.load(embeddings, session=sess)
        if FLAGS.pretrained_entity_embeddings and os.path.exists(FLAGS.pretrained_entity_embeddings):
            embeddings = np.loadtxt(FLAGS.pretrained_entity_embeddings)
            self.entity_embeddings.load(embeddings, session=sess)

    def _calc_cum_discounted_rewards(self, final_reward, kg, hp):
        running_add = np.zeros([final_reward.shape[0]])
        cum_discounted_rewards = np.zeros([final_reward.shape[0], hp.n_steps])
        cum_discounted_rewards[:, hp.n_steps-1] = final_reward
        for t in reversed(range(hp.n_steps)):
            running_add = hp.gamma * running_add + cum_discounted_rewards[:, t]
            cum_discounted_rewards[:, t] = running_add
        return cum_discounted_rewards

    def _evaluate(self, sess, beam=False, print_paths=False, source='test'):
        hp = self.hparams
        kg = self.kg

        triples_eval = []
        idx_tail_pred = []
        paths = defaultdict(list)

        n_itrs = 0
        feed_dict= {}
        for episodes in kg.get_eval_episodes(hp.batch_size, hp.n_eval_rollouts, hp.n_steps, source=source):
            rollout, step, state = episodes
            batch_size = rollout['batch_size']
            n_rollouts = rollout['n_rollouts']
            bs = rollout['bs']

            prev_relation = np.ones(bs, dtype='int64') * kg.DUMMY_START_RELATION
            prev_lstm_state = np.zeros((hp.n_lstm_layers, 2, bs, self.n_in * hp.n_hidden_dims)).astype('float32')

            n_itrs += 1

            beam_scores = np.zeros((bs, 1))
            log_probs = np.zeros((bs,))

            if print_paths:
                entity_trajectory = []
                relation_trajectory = []

            feed_dict[self.query_relation] = rollout['query_relation']
            for i in range(hp.n_steps):
                feed_dict[self.next_relations] = state['next_relations']  # bs x max_actions
                feed_dict[self.next_entities] = state['next_entities']  # bs x max_actions
                feed_dict[self.current_entity] = state['current_entity']  # bs
                feed_dict[self.prev_relation] = prev_relation
                feed_dict[self.prev_lstm_state] = prev_lstm_state

                loss, logits, lstm_state, next_action, next_relation = \
                    sess.run([self.eval_loss, self.eval_logits, self.eval_lstm_state, self.eval_next_action, self.eval_next_relation],
                             feed_dict=feed_dict)

                if beam:
                    scores = logits + beam_scores  # bs x max_actions
                    if i == 0:
                        idx = np.argsort(scores)[:, -n_rollouts:]  # bs x n_rollouts
                        idx = idx[np.arange(bs), np.tile(range(n_rollouts), batch_size)]  # bs
                    else:
                        idx = np.argsort(np.reshape(scores, (-1, n_rollouts * hp.max_actions)))[:, -n_rollouts:]  # batch_size x n_rollouts
                        idx = np.reshape(idx, (-1))  # bs

                    x, y = idx % hp.max_actions, idx // hp.max_actions
                    y += np.repeat([b * n_rollouts for b in range(batch_size)], n_rollouts)

                    state['current_entity'] = state['current_entity'][y]
                    state['next_relations'] = state['next_relations'][y, :]
                    state['next_entities'] = state['next_entities'][y, :]
                    prev_lstm_state = prev_lstm_state[:, :, y, :]
                    next_action = x
                    next_relation = state['next_relations'][np.arange(bs), x]

                    beam_scores = np.reshape(scores[y, x], (-1, 1))

                    if print_paths:
                        for j in range(i):
                            entity_trajectory[j] = entity_trajectory[j][y]
                            relation_trajectory[j] = relation_trajectory[j][y]

                prev_relation = next_relation

                if print_paths:
                    entity_trajectory.append(state['current_entity'])
                    relation_trajectory.append(next_relation)

                step, state = kg.get_next_state(next_action, step, state, rollout)

                log_probs += logits[np.arange(bs), next_action]

            if beam:
                log_probs = beam_scores

            log_probs = np.reshape(log_probs, (batch_size, n_rollouts))
            sorted_idx = np.argsort(-log_probs)
            final_entity = np.reshape(state['current_entity'], (batch_size, n_rollouts))
            final_probs = np.zeros((batch_size, kg.n_entities))
            for i in range(n_rollouts):
                e = final_entity[:, i]
                if hp.eval_pool == 'max':
                    mask = final_probs[np.arange(batch_size), e] < log_probs[:, i]
                    final_probs[np.arange(batch_size)[mask], e[mask]] = np.exp(log_probs[:, i])[mask]
                elif hp.eval_pool == 'sum':
                    final_probs[np.arange(batch_size), e] += np.exp(log_probs[:, i])
                else:
                    raise ValueError('Invalid `eval_pool` in hparams')
            idx_pred = np.argsort(-final_probs)
            idx_tail_pred.append(idx_pred)

            start_entity = np.reshape(rollout['start_entity'], (batch_size, n_rollouts))[:, 0]
            query_relation = np.reshape(rollout['query_relation'], (batch_size, n_rollouts))[:, 0]
            end_entity = np.reshape(rollout['end_entity'], (batch_size, n_rollouts))[:, 0]
            triples_eval.append(np.stack([start_entity, query_relation, end_entity], axis=-1))

            if print_paths:
                entity_trajectory.append(state['current_entity'])
                final_reward = kg.get_reward(state, rollout, hp.pos_reward, hp.neg_reward)
                paths = kg.get_paths(paths, rollout, entity_trajectory, relation_trajectory, sorted_idx, final_reward, log_probs)


        idx_tail_pred = np.concatenate(idx_tail_pred, 0).tolist()
        triples_eval = np.concatenate(triples_eval, 0).tolist()
        _, eval_tail, _ = kg.calc_metrics(triples_eval, idx_tail_pred)

        if print_paths:
            for q in paths:
                for p in paths[q]:
                    print('=== Print Paths === %s' % p)

        return eval_tail, paths






