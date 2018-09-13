from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from collections import defaultdict
import random
import multiprocessing as mp

import numpy as np


def preprocess(data_dir):
    entities, relations = set(), set()
    for filename in ['train.txt', 'test.txt', 'valid.txt', 'graph.txt']:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                for line in f.readlines():
                    line = line.strip()
                    sp = line.split('\t') if '\t' in line else line.split(' ')
                    entities.add(sp[0])
                    entities.add(sp[2])
                    relations.add(sp[1])
    with open(os.path.join(data_dir, 'entity2id.txt'), 'w') as f:
        for id, entity in enumerate(sorted(entities)):
            f.write('%s\t%d\n' % (entity, id))
    with open(os.path.join(data_dir, 'relation2id.txt'), 'w') as f:
        for id, relation in enumerate(sorted(relations)):
            f.write('%s\t%d\n' % (relation, id))

class KG(object):

    # for basic setting

    def __init__(self, name, data_dir, model_name, neg_sampling_mode='unit', build_graph=False, max_actions=None,
                 pad_entity=None, pad_relation=None, dummy_start_relation=None, no_op_relation=None,
                 end_relation=None, seed=1234):
        self.name = name
        self.data_dir = data_dir
        self.neg_sampling_mode = neg_sampling_mode  # 'unit', 'bern', 'both'
        self.model_name = model_name

        self.build_graph = build_graph
        self.max_actions = max_actions
        self.pad_entity = pad_entity
        self.pad_relation = pad_relation
        self.dummy_start_relation = dummy_start_relation
        self.no_op_relation = no_op_relation
        self.end_relation = end_relation

        if seed:
            random.seed(seed)
            np.random.seed(seed)

        if not os.path.exists(os.path.join(data_dir, 'entity2id.txt')):
            preprocess(data_dir)

        self.entity2id, self.id2entity = self._load_dict('entity2id.txt')
        self.relation2id, self.id2relation = self._load_dict('relation2id.txt')

        self._update_entity_dict()
        self._update_relation_dict()

        self.entities = list(self.entity2id.values())

        self.triples_train = self._load_triples('train.txt')
        self.triples_test = self._load_triples('test.txt')
        self.triples_valid = self._load_triples('valid.txt')

        self.triples_train_pool = set(self.triples_train)
        self.triples_pool = set(self.triples_train) | set(self.triples_test) | set(self.triples_valid)

        self.relation_property = self._prepare_for_negative_sampling()

        if self.build_graph:
            filename = 'graph.txt'
            assert os.path.exists(os.path.join(data_dir, filename)), "'%s' does not exist" % filename
            self.triples_graph = self._load_triples(filename)
            self.triples_pool |= set(self.triples_graph)
            self._build_graph(filename)

    def _build_graph(self, filename):
        if self.model_name == 'minerva':
            self._build_minarva_graph(filename)
        elif self.model_name == 'neural_lp':
            self._build_neurallp_graph(filename)

    def _load_dict(self, filename):
        val2id = dict()
        id2val = dict()
        with open(os.path.join(self.data_dir, filename)) as f:
            for line in f.readlines():
                line = line.strip()
                sp = line.split('\t') if '\t' in line else line.split(' ')
                val2id[sp[0]] = int(sp[1])
                id2val[int(sp[1])] = sp[0]
        return val2id, id2val

    def _load_triples(self, filename):
        triples = []
        with open(os.path.join(self.data_dir, filename)) as f:
            for line in f.readlines():
                line = line.strip()
                sp = line.split('\t') if '\t' in line else line.split(' ')
                head_id = self.entity2id[sp[0]]
                relation_id = self.relation2id[sp[1]]
                tail_id = self.entity2id[sp[2]]
                triples.append((head_id, relation_id, tail_id))
        return triples

    def _update_entity_dict(self):
        n = len(self.entity2id)
        if self.pad_entity == 'PAD' and self.pad_entity not in self.entity2id:
            self.entity2id[self.pad_entity] = n
            self.id2entity[n] = self.pad_entity
            n += 1

    def _update_relation_dict(self):
        n = len(self.relation2id)
        if self.pad_relation == 'PAD' and self.pad_relation not in self.relation2id:
            self.relation2id[self.pad_relation] = n
            self.id2relation[n] = self.pad_relation
            n += 1
        if self.dummy_start_relation == 'DUMMY_START' and self.dummy_start_relation not in self.relation2id:
            self.relation2id[self.dummy_start_relation] = n
            self.id2relation[n] = self.dummy_start_relation
            n += 1
        if self.no_op_relation == 'NO_OP' and self.no_op_relation not in self.relation2id:
            self.relation2id[self.no_op_relation] = n
            self.id2relation[n] = self.no_op_relation
            n += 1
        if self.end_relation == 'END' and self.end_relation not in self.relation2id:
            self.relation2id[self.end_relation] = n
            self.id2relation[n] = self.end_relation
            n += 1

    def _prepare_for_negative_sampling(self):
        # use 'bern'
        relation_headset = defaultdict(set)
        relation_tailset = defaultdict(set)
        relation_property = dict()
        for triple in self.triples_train:
            head, relation, tail = triple
            relation_headset[relation].add(head)
            relation_tailset[relation].add(tail)
        for relation in relation_headset.iterkeys():
            relation_property[relation] = len(relation_tailset[relation]) * 1. / (len(relation_headset[relation]) + len(relation_tailset[relation])) \
                                          if self.neg_sampling_mode == 'bern' else 0.5
        return relation_property

    @property
    def n_entities(self):
        return len(self.entity2id)

    @property
    def n_relations(self):
        return len(self.relation2id)

    @property
    def n_triples_train(self):
        return len(self.triples_train)

    @property
    def n_triples_test(self):
        return len(self.triples_test)

    @property
    def n_triples_valid(self):
        return len(self.triples_valid)

    def get_train_batch(self, batch_size, neg_sampling=True):
        rand_idx = np.random.permutation(self.n_triples_train)
        start = 0
        while start < self.n_triples_train:
            end = min(start + batch_size, self.n_triples_train)
            batch_pos = [self.triples_train[i] for i in rand_idx[start:end]]

            if neg_sampling:
                batch_neg = []
                for head, relation, tail in batch_pos:
                    corrupt_head_prob = np.random.binomial(1, self.relation_property[relation])  # 'bern' (from TransH (Wang, 2014)) or 'unit'

                    if corrupt_head_prob or self.neg_sampling_mode == 'both':
                        while True:
                            head_neg = random.choice(self.entities)
                            tail_neg = tail
                            if (head_neg, relation, tail_neg) not in self.triples_train_pool:
                                break
                        batch_neg.append((head_neg, relation, tail_neg))

                    if not corrupt_head_prob or self.neg_sampling_mode == 'both':
                        while True:
                            head_neg = head
                            tail_neg = random.choice(self.entities)
                            if (head_neg, relation, tail_neg) not in self.triples_train_pool:
                                break
                        batch_neg.append((head_neg, relation, tail_neg))

                yield np.array(batch_pos), np.array(batch_neg)
            else:
                yield np.array(batch_pos)

            start = end

    def get_eval_batch(self, batch_size, source='test'):
        start = 0
        n_triples, triples = (self.n_triples_test, self.triples_test) if source == 'test' else (self.n_triples_valid, self.triples_valid)
        while start < n_triples:
            end = min(start + batch_size, n_triples)
            batch = triples[start:end]
            yield np.array(batch)
            start = end

    def calc_metrics(self, triples, idx_tail_pred, idx_head_pred=None):  # `idx_tail_pred` before `idx_head_pred`
        metrics = ['hits@1_raw', 'hits@5_raw', 'hits@10_raw', 'mr_raw', 'mrr_raw',
                   'hits@1_flt', 'hits@5_flt', 'hits@10_flt', 'mr_flt', 'mrr_flt']
        head = dict((name, 0.) for name in metrics) if idx_head_pred is not None else None
        tail = dict((name, 0.) for name in metrics)

        in_queue = mp.JoinableQueue()
        out_queue = mp.Queue()

        n_processes = 8
        for _ in range(n_processes):
            mp.Process(target=self._calc_rank, kwargs={'in_queue': in_queue, 'out_queue': out_queue}).start()

        for i, triple in enumerate(triples):
            idx_head = idx_head_pred[i] if idx_head_pred is not None else None
            idx_tail = idx_tail_pred[i]
            in_queue.put((triple, idx_head, idx_tail))

        for _ in range(n_processes):
            in_queue.put(None)
        in_queue.join()

        for _ in enumerate(triples):
            head_rank_raw, head_rank_flt, tail_rank_raw, tail_rank_flt = out_queue.get()

            if idx_head_pred is not None:
                if head_rank_raw == 0:
                    head['hits@1_raw'] += 1
                if head_rank_raw < 5:
                    head['hits@5_raw'] += 1
                if head_rank_raw < 10:
                    head['hits@10_raw'] += 1
                head['mr_raw'] += (head_rank_raw + 1)
                head['mrr_raw'] += 1. / (head_rank_raw + 1)

                if head_rank_flt == 0:
                    head['hits@1_flt'] += 1
                if head_rank_flt < 5:
                    head['hits@5_flt'] += 1
                if head_rank_flt < 10:
                    head['hits@10_flt'] += 1
                head['mr_flt'] += (head_rank_flt + 1)
                head['mrr_flt'] += 1. / (head_rank_flt + 1)

            if tail_rank_raw == 0:
                tail['hits@1_raw'] += 1
            if tail_rank_raw < 5:
                tail['hits@5_raw'] += 1
            if tail_rank_raw < 10:
                tail['hits@10_raw'] += 1
            tail['mr_raw'] += (tail_rank_raw + 1)
            tail['mrr_raw'] += 1. / (tail_rank_raw + 1)

            if tail_rank_flt == 0:
                tail['hits@1_flt'] += 1
            if tail_rank_flt < 5:
                tail['hits@5_flt'] += 1
            if tail_rank_flt < 10:
                tail['hits@10_flt'] += 1
            tail['mr_flt'] += (tail_rank_flt + 1)
            tail['mrr_flt'] += 1. / (tail_rank_flt + 1)

        for name in metrics:
            if idx_head_pred is not None:
                head[name] /= len(triples)
            tail[name] /= len(triples)

        eval_head = '|'.join(map(lambda name: ' %s: %.8f ' % (name, head[name]), metrics)) if idx_head_pred is not None else None
        eval_tail = '|'.join(map(lambda name: ' %s: %.8f ' % (name, tail[name]), metrics))
        eval_both = '|'.join(map(lambda name: ' %s: %.8f ' % (name, (head[name] + tail[name]) / 2), metrics)) if idx_head_pred is not None else None
        return eval_head, eval_tail, eval_both

    def _calc_rank(self, in_queue, out_queue):
        while True:
            prediction = in_queue.get()
            if prediction is None:
                in_queue.task_done()
                return
            else:
                triple, idx_head, idx_tail = prediction
                head, relation, tail = triple

                head_rank_raw, head_rank_flt = 0, 0
                if idx_head is not None:
                    for candidate in idx_head:
                        if candidate == head:
                            break
                        else:
                            head_rank_raw += 1
                            if (candidate, relation, tail) in self.triples_pool:
                                continue
                            else:
                                head_rank_flt += 1

                tail_rank_raw, tail_rank_flt = 0, 0
                for candidate in idx_tail:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, relation, candidate) in self.triples_pool:
                            continue
                        else:
                            tail_rank_flt += 1

                out_queue.put((head_rank_raw, head_rank_flt, tail_rank_raw, tail_rank_flt))
                in_queue.task_done()

    # for MINERVA's setting

    @property
    def PAD_ENTITY(self):
        return self.entity2id[self.pad_entity]

    @property
    def PAD_RELATION(self):
        return self.relation2id[self.pad_relation]

    @property
    def DUMMY_START_RELATION(self):
        return self.relation2id[self.dummy_start_relation]

    @property
    def NO_OP_RELATION(self):
        return self.relation2id[self.no_op_relation]

    def _build_minarva_graph(self, filename):
        assert self.pad_entity is not None, "`pad_entity` should not be None"
        assert self.pad_relation is not None, "`pad_relation` should not be None"
        assert self.max_actions is not None, "`max_actions` should not be None"
        assert self.dummy_start_relation is not None, "`dummy_start_relation` should not be None"
        assert self.no_op_relation is not None, "`no_op_relation` should not be None"

        self.graph_e2re = np.ones((self.n_entities, self.max_actions, 2), dtype=np.dtype('int32'))
        self.graph_e2n = np.zeros((self.n_entities,), dtype=np.dtype('int32'))
        for e in range(self.n_entities):
            self.graph_e2re[e, :, 0] *= self.PAD_RELATION
            self.graph_e2re[e, :, 1] *= self.PAD_ENTITY
            self.graph_e2re[e, 0, 0] = self.NO_OP_RELATION
            self.graph_e2re[e, 0, 1] = e
            self.graph_e2n[e] += 1

        with open(os.path.join(self.data_dir, filename)) as f:
            for line in f.readlines():
                line = line.strip()
                sp = line.split('\t') if '\t' in line else line.split(' ')
                e1 = self.entity2id[sp[0]]
                r = self.relation2id[sp[1]]
                e2 = self.entity2id[sp[2]]

                n = self.graph_e2n[e1]
                if n < self.max_actions:
                    self.graph_e2re[e1, n, 0] = r
                    self.graph_e2re[e1, n, 1] = e2
                    n += 1
                    self.graph_e2n[e1] = n

    def _outgoing_edges(self, current_entity, start_entity, query_relation, answer, is_last_step, n_rollouts):  # current_entity: bs
        ret = self.graph_e2re[current_entity, :, :].copy()
        for i in range(current_entity.shape[0]):
            e1 = current_entity[i]
            if e1 == start_entity[i]:
                mask = np.logical_and(ret[i, :, 0] == query_relation[i], ret[i, :, 1] == answer[i])
                ret[i, :, 0][mask] = self.PAD_RELATION
                ret[i, :, 1][mask] = self.PAD_ENTITY
            if is_last_step:
                correct_e = answer[i]
                for j in range(ret[i, :, 1].shape[0]):
                    e2 = ret[i, j, 1]
                    r = ret[i, j, 0]
                    if (e1, r, e2) in self.triples_train_pool and e2 != correct_e:  # Might be wrong in the original code which uses all triples from train/valid/test as the pool
                        ret[i, j, 0] = self.PAD_RELATION
                        ret[i, j, 1] = self.PAD_ENTITY
        return ret

    def get_train_episodes(self, batch_size, n_rollouts, n_steps):
        for batch in self.get_train_batch(batch_size, neg_sampling=False):
            yield self._get_episodes(batch, n_rollouts, n_steps)

    def get_eval_episodes(self, batch_size, n_rollouts, n_steps, source='test'):
        for batch in self.get_eval_batch(batch_size, source=source):
            yield self._get_episodes(batch, n_rollouts, n_steps)

    def _get_episodes(self, batch, n_rollouts, n_steps):
        step = 0
        start_entity, query_relation, end_entity = np.split(batch, 3, axis=1)
        rollout = {
            'start_entity': np.repeat(start_entity, n_rollouts),  # (batch_size*n_rollouts)
            'query_relation': np.repeat(query_relation, n_rollouts),  # (batch_size*n_rollouts)
            'end_entity': np.repeat(end_entity, n_rollouts),
            'bs': batch.shape[0] * n_rollouts,
            'batch_size': batch.shape[0],
            'n_rollouts': n_rollouts,
            'n_steps': n_steps
        }
        current_entity = np.array(rollout['start_entity'])
        out_edges = self._outgoing_edges(current_entity, rollout['start_entity'], rollout['query_relation'],
                                         rollout['end_entity'], step == n_steps - 1, n_rollouts)
        state = {
            'next_relations': out_edges[:, :, 0],
            'next_entities': out_edges[:, :, 1],
            'current_entity': current_entity
        }
        return rollout, step, state

    def get_next_state(self, action, step, state, rollout):
        step += 1
        current_entity = state['next_entities'][np.arange(rollout['bs']), action]
        out_edges = self._outgoing_edges(current_entity, rollout['start_entity'], rollout['query_relation'],
                                         rollout['end_entity'], step == rollout['n_steps'] -  1, rollout['n_rollouts'])
        new_state = {
            'next_relations': out_edges[:, :, 0],
            'next_entities': out_edges[:, :, 1],
            'current_entity': current_entity
        }
        return step, new_state

    def get_reward(self, state, rollout, pos_reward, neg_reward):
        reward = (state['current_entity'] == rollout['end_entity'])
        condlist = [reward == True, reward == False]
        choicelist = [pos_reward, neg_reward]
        reward = np.select(condlist, choicelist)
        return reward

    def get_paths(self, paths, rollout, entity_trajectory, relation_trajectory, sorted_indx, final_reward, log_probs):
        batch_size = rollout['batch_size']
        n_rollouts = rollout['n_rollouts']
        start_entity = np.reshape(rollout['start_entity'], (batch_size, n_rollouts))[:, 0]
        query_relation = np.reshape(rollout['query_relation'], (batch_size, n_rollouts))[:, 0]
        end_entity = np.reshape(rollout['end_entity'], (batch_size, n_rollouts))[:, 0]

        for b in range(batch_size):
            se = self.id2entity[start_entity[b]]
            ee = self.id2entity[end_entity[b]]
            qr = self.id2relation[query_relation[b]]
            paths[qr].append('query: %s - %s' % (se, ee))
            for i, r in enumerate(sorted_indx[b]):
                indx = b * n_rollouts + r
                paths[qr].append('entity_trajectory[%d]: %s' % (i, ' - '.join(self.id2entity[e[indx]] for e in entity_trajectory)))
                paths[qr].append('relation_trajectory[%d]: %s' % (i, ' - '.join(self.id2relation[e[indx]] for e in relation_trajectory)))
                paths[qr].append('reward[%d]: %d' % (i, final_reward[indx]))
                paths[qr].append('log_prob[%d]: %f' % (i, log_probs[b, r]))

        return paths

    # for Neural LP's setting

    @property
    def END_RELATION(self):
        return self.relation2id[self.end_relation]

    def _build_neurallp_graph(self, filename):
        self.graph_r2ee = {r: set() for r in range(self.n_relations)}
        with open(os.path.join(self.data_dir, filename)) as f:
            for line in f.readlines():
                line = line.strip()
                sp = line.split('\t') if '\t' in line else line.split(' ')
                e1 = self.entity2id[sp[0]]
                r = self.relation2id[sp[1]]
                e2 = self.entity2id[sp[2]]
                self.graph_r2ee[r].add((e1, e2))

    def get_triples_from_batch(self, batch):
        head, query, tail = zip(*batch)
        return head, query, tail

    def get_filtered_kg(self, batch):
        batch_r2ee = {r: set() for r in range(self.n_relations)}
        for (head, query, tail) in batch:
            batch_r2ee[query].add((head, tail))

        kg_inputs_r2ee = {r: self.graph_r2ee[r] - batch_r2ee[r] for r in range(self.n_relations)}
        kg_inputs = {}
        for r in range(self.n_relations):
            indices = [[0, 0]]
            values = [0.]
            for ee in kg_inputs_r2ee[r]:
                indices.append([ee[0], ee[1]])
                values.append(1.)
            kg_inputs[r] = (indices, values, [self.n_entities, self.n_entities])
        return kg_inputs

    def get_kg(self, batch):
        kg_inputs = {}
        for r in range(self.n_relations):
            indices = []
            values = []
            for ee in self.graph_r2ee[r]:
                indices.append([ee[0], ee[1]])
                values.append(1.)
            kg_inputs[r] = (indices, values, [self.n_entities, self.n_entities])
        return kg_inputs
