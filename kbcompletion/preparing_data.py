from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from collections import defaultdict
import random

class Obj(object):
    pass

def load_triples(data_dir, filename):
    triples = []
    with open(os.path.join(data_dir, filename)) as f:
        for line in f.readlines():
            line = line.strip()
            sp = line.split('\t' if '\t' in line else ' ')
            triples.append(sp)
    return triples

def investigate_basic_stats(obj, data_dir):
    train_triples = load_triples(data_dir, 'train.txt')
    zip_triples = zip(*train_triples)
    train_entities = set(zip_triples[0]) | set(zip_triples[2])
    train_relations = set(zip_triples[1])

    valid_triples = load_triples(data_dir, 'valid.txt')
    zip_triples = zip(*valid_triples)
    valid_entities = set(zip_triples[0]) | set(zip_triples[2])
    valid_relations = set(zip_triples[1])

    test_triples = load_triples(data_dir, 'test.txt')
    zip_triples = zip(*test_triples)
    test_entities = set(zip_triples[0]) | set(zip_triples[2])
    test_relations = set(zip_triples[1])

    graph_tripes = None
    if os.path.exists(os.path.join(data_dir, 'graph.txt')):
        graph_tripes = load_triples(data_dir, 'graph.txt')
        zip_triples = zip(*graph_tripes)
        graph_entities = set(zip_triples[0]) | set(zip_triples[2])
        graph_relations = set(zip_triples[1])
        print('graph n_nodes: %d, n_edges: %d, n_edge_types: %d' % (len(graph_entities), len(graph_tripes), len(graph_relations)))

    print('n_train_triples: %d, n_valid_triples: %d, n_test_triples: %d\n' %
          (len(train_triples), len(valid_triples), len(test_triples)))

    entities = train_entities | valid_entities | test_entities
    relations = train_relations | valid_relations | test_relations
    if graph_tripes:
        entities |= graph_entities
        relations |= graph_relations

    n_train_entities = len(train_entities)
    n_valid_entities = len(valid_entities)
    n_test_entities = len(test_entities)
    n_entities = len(entities)
    pct_train_entities = n_train_entities * 1. / n_entities
    pct_valid_entities = n_valid_entities * 1. / n_entities
    pct_test_entities = n_test_entities * 1. / n_entities

    unseen_entities_valid = valid_entities - train_entities
    unseen_entities_test = test_entities - train_entities
    n_new_entities_valid2train = len(unseen_entities_valid)
    n_new_entities_test2train = len(unseen_entities_test)

    if graph_tripes:
        unseen_entities_valid_fromkg = valid_entities - graph_entities
        unseen_entities_test_fromkg = test_entities - graph_entities
        n_new_entities_valid2kg = len(unseen_entities_valid_fromkg)
        n_new_entities_test2kg = len(unseen_entities_test_fromkg)
        print('n_new_entities_valid2kg: %d, n_new_entities_test2kg: %d' % (n_new_entities_valid2kg, n_new_entities_test2kg))

    n_train_relations = len(train_relations)
    n_valid_relations = len(valid_relations)
    n_test_relations = len(test_relations)
    n_relations = len(relations)
    pct_train_relations = n_train_relations * 1. / n_relations
    pct_valid_relations = n_valid_relations * 1. / n_relations
    pct_test_relations = n_test_relations * 1. / n_relations

    n_new_relations_valid2train = len(valid_relations - train_relations)
    n_new_relations_test2train = len(test_relations - train_relations)

    if graph_tripes:
        n_new_relations_valid2kg = len(valid_relations - graph_relations)
        n_new_relations_test2kg = len(test_relations - graph_relations)
        print('n_new_relations_valid2kg: %d, n_new_relations_test2kg: %d\n' % (n_new_relations_valid2kg, n_new_relations_test2kg))

    print('n_train_entities: %d, n_valid_entities: %d, n_test_entities: %d, n_entities: %d' %
          (n_train_entities, n_valid_entities, n_test_entities, n_entities))

    print('pct_train_entities: %f, pct_valid_entities: %f, pct_test_entities: %f' %
          (pct_train_entities, pct_valid_entities, pct_test_entities))

    print('n_new_entities_valid2train: %d, n_new_entities_test2train: %d\n' %
          (n_new_entities_valid2train, n_new_entities_test2train))

    print('n_train_relations: %d, n_valid_relations: %d, n_test_relations: %d, n_relations: %d' %
          (n_train_relations, n_valid_relations, n_test_relations, n_relations))

    print('pct_train_relations: %f, pct_valid_relations: %f, pct_test_relations: %f' %
          (pct_train_relations, pct_valid_relations, pct_test_relations))

    print('n_new_relations_valid2train: %d, n_new_relations_test2train: %d\n' %
          (n_new_relations_valid2train, n_new_relations_test2train))

    entity2id = {e: i for i, e in enumerate(entities)}
    id2entity = {i: e for e, i in entity2id.iteritems()}

    relation2id = {r: i for i, r in enumerate(relations)}
    id2relation = {i: r for r, i in relation2id.iteritems()}

    obj.train_t = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in train_triples]
    obj.valid_t = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in valid_triples]
    obj.test_t = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in test_triples]
    obj.triples = obj.train_t + obj.valid_t + obj.test_t
    obj.graph_t = []
    if graph_tripes:
        obj.graph_t = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in graph_tripes]
        obj.triples += obj.graph_t
    obj.entity2id = entity2id
    obj.id2entity = id2entity
    obj.relation2id = relation2id
    obj.id2relation = id2relation
    obj.unseen_e_valid = set([entity2id[e] for e in unseen_entities_valid])
    obj.unseen_e_test = set([entity2id[e] for e in unseen_entities_test])

def investigate_four_types_of_relations(obj, count_graph=True):
    hr2t = defaultdict(set)
    tr2h = defaultdict(set)

    for h, r, t in obj.triples:
        hr2t[(h, r)].add(t)
        tr2h[(t, r)].add(h)

    def _count_relation_types(tri):
        rt_1_1, rt_1_n, rt_n_1, rt_n_n = 0, 0, 0, 0
        for h, r, t in tri:
            if len(hr2t[(h, r)]) == 1:
                if len(tr2h[(t, r)]) == 1:
                    rt_1_1 += 1
                elif len(tr2h[(t, r)]) > 1:
                    rt_n_1 += 1
                else:
                    raise ValueError
            elif len(hr2t[(h, r)]) > 1:
                if len(tr2h[(t, r)]) == 1:
                    rt_1_n += 1
                elif len(tr2h[(t, r)]) > 1:
                    rt_n_n += 1
                else:
                    raise ValueError
            else:
                raise ValueError
        return rt_1_1, rt_1_n, rt_n_1, rt_n_n

    train_1_1, train_1_n, train_n_1, train_n_n = _count_relation_types(obj.train_t)
    valid_1_1, valid_1_n, valid_n_1, valid_n_n = _count_relation_types(obj.valid_t)
    test_1_1, test_1_n, test_n_1, test_n_n = _count_relation_types(obj.test_t)
    graph_1_1, graph_1_n, graph_n_1, graph_n_n = _count_relation_types(obj.graph_t)
    all_1_1 = train_1_1 + valid_1_1 + test_1_1
    all_1_n = train_1_n + valid_1_n + test_1_n
    all_n_1 = train_n_1 + valid_n_1 + test_n_1
    all_n_n = train_n_n + valid_n_n + test_n_n
    if count_graph:
        all_1_1 += graph_1_1
        all_1_n += graph_1_n
        all_n_1 += graph_n_1
        all_n_n += graph_n_n

    print('#train_1_1: %d, #train_1_n: %d, #train_n_1: %d, #train_n_n: %d' %
          (train_1_1, train_1_n, train_n_1, train_n_n))
    print('#valid_1_1: %d, #valid_1_n: %d, #valid_n_1: %d, #valid_n_n: %d' %
          (valid_1_1, valid_1_n, valid_n_1, valid_n_n))
    print('#test_1_1: %d, #test_1_n: %d, #test_n_1: %d, #test_n_n: %d' %
          (test_1_1, test_1_n, test_n_1, test_n_n))
    if obj.graph_t:
        print('#graph_1_1: %d, #graph_1_n: %d, #graph_n_1: %d, #graph_n_n: %d' %
              (graph_1_1, graph_1_n, graph_n_1, graph_n_n))
    print('#all_1_1: %d, #all_1_n: %d, #all_n_1: %d, #all_n_n: %d\n' %
          (all_1_1, all_1_n, all_n_1, all_n_n))

def investigate_interchangeable_triples(obj, count_graph=True):
    tri = set(obj.triples)
    ich_tri = set()

    n_ich = 0
    n_ich_graph = 0
    for h, r, t in obj.graph_t:
        if (t, r, h) in tri:
            n_ich_graph += 1
            if count_graph:
                n_ich += 1
                ich_tri.add((h, r, t))

    n_ich_train = 0
    for h, r, t in obj.train_t:
        if (t, r, h) in tri:
            n_ich_train += 1
            n_ich += 1
            ich_tri.add((h, r, t))

    n_ich_valid = 0
    for h, r, t in obj.valid_t:
        if (t, r, h) in tri:
            n_ich_valid += 1
            n_ich += 1
            ich_tri.add((h, r, t))

    n_ich_test = 0
    for h, r, t in obj.test_t:
        if (t, r, h) in tri:
            n_ich_test += 1
            n_ich += 1
            ich_tri.add((h, r, t))

    obj.ich_triples = ich_tri
    print('n_ich: %d, n_ich_graph: %d, n_ich_train: %d, n_ich_valid: %d, n_ich_test: %d' % (n_ich, n_ich_graph, n_ich_train, n_ich_valid, n_ich_test))

def investigate_selfloop_triples(obj, count_graph=True):
    sl_tri = set()

    n_slp = 0
    n_slp_graph = 0
    for h, r, t in obj.graph_t:
        if h == t:
            n_slp_graph += 1
            if count_graph:
                n_slp += 1
                sl_tri.add((h, r, t))

    n_slp_train = 0
    for h, r, t in obj.train_t:
        if h == t:
            n_slp_train += 1
            n_slp += 1
            sl_tri.add((h, r, t))

    n_slp_valid = 0
    for h, r, t in obj.valid_t:
        if h == t:
            n_slp_valid += 1
            n_slp += 1
            sl_tri.add((h, r, t))

    n_slp_test = 0
    for h, r, t in obj.test_t:
        if h == t:
            n_slp_test += 1
            n_slp += 1
            sl_tri.add((h, r, t))

    obj.sl_triples = sl_tri
    print('n_slp: %d, n_slp_graph: %d, n_slp_train: %d, n_slp_valid: %d, n_slp_test: %d' % (n_slp, n_slp_graph, n_slp_train, n_slp_valid, n_slp_test))

def clean_triples(obj, removed_entities, removed_triples):
    train_t = []
    for h, r, t in obj.train_t:
        if (h not in removed_entities) and (t not in removed_entities) and ((h, r, t) not in removed_triples):
            train_t.append((h, r, t))

    valid_t = []
    for h, r, t in obj.valid_t:
        if (h not in removed_entities) and (t not in removed_entities) and ((h, r, t) not in removed_triples):
            valid_t.append((h, r, t))

    test_t = []
    for h, r, t in obj.test_t:
        if (h not in removed_entities) and (t not in removed_entities) and ((h, r, t) not in removed_triples):
            test_t.append((h, r, t))

    obj.cleaned_train_t = train_t
    obj.cleaned_valid_t = valid_t
    obj.cleaned_test_t = test_t

def make_dataset(train_t, valid_t, test_t, id2entity, id2relation, kg_t=None):
    train_triples = [(id2entity[h], id2relation[r], id2entity[t]) for h, r, t in train_t]
    valid_triples = [(id2entity[h], id2relation[r], id2entity[t]) for h, r, t in valid_t]
    test_triples = [(id2entity[h], id2relation[r], id2entity[t]) for h, r, t in test_t]
    kg_triples = None
    if kg_t:
        kg_triples = [(id2entity[h], id2relation[r], id2entity[t]) for h, r, t in kg_t]
    return train_triples, valid_triples, test_triples, kg_triples

def add_reversed(obj, ori_triples, ich_triples):
    new_triples = []
    triples_set = set()
    for h, r, t in ori_triples:
        if (h, r, t) not in triples_set:
            new_triples.append((h, r, t))
            triples_set.add((h, r, t))

        if (h, r, t) in ich_triples:
            if (t, r, h) not in triples_set:
                new_triples.append((t, r, h))
                triples_set.add((t, r, h))
        else:
            r_str = obj.id2relation[r]
            inv_r_str = 'inv-%s' % r_str
            inv_r = len(obj.id2relation)

            obj.relation2id[inv_r_str] = inv_r
            obj.id2relation[inv_r] = inv_r_str

            if (t, inv_r, h) not in triples_set:
                new_triples.append((t, inv_r, h))
                triples_set.add((t, inv_r, h))
    return new_triples

def write_to_file(triples, data_dir, filename):
    with open(os.path.join(data_dir, filename), 'w') as f:
        for h, r, t in triples:
            f.write('%s\t%s\t%s\n' % (h, r, t))

def extract_kg_from_train(train_t, mode='copy', split_ratio=3.):
    new_train_t = []
    kg = []

    if mode == 'copy':
        new_train_t = train_t
        kg = train_t
    else:
        n_edges = split_ratio / (split_ratio + 1) * len(train_t)
        indices = set(random.sample(range(len(train_t)), int(n_edges)))
        for i in range(len(train_t)):
            if i in indices:
                kg.append(train_t[i])
            else:
                new_train_t.append(train_t[i])
    return kg, new_train_t

def combine_train_valid_test(obj, data_dir):
    train_triples = load_triples(data_dir, 'train.txt')
    zip_triples = zip(*train_triples)
    train_entities = set(zip_triples[0]) | set(zip_triples[2])
    train_relations = set(zip_triples[1])

    valid_triples = load_triples(data_dir, 'valid.txt')
    zip_triples = zip(*valid_triples)
    valid_entities = set(zip_triples[0]) | set(zip_triples[2])
    valid_relations = set(zip_triples[1])

    test_triples = load_triples(data_dir, 'test.txt')
    zip_triples = zip(*test_triples)
    test_entities = set(zip_triples[0]) | set(zip_triples[2])
    test_relations = set(zip_triples[1])

    entities = train_entities | valid_entities | test_entities
    relations = train_relations | valid_relations | test_relations

    entity2id = {e: i for i, e in enumerate(entities)}
    id2entity = {i: e for e, i in entity2id.iteritems()}

    relation2id = {r: i for i, r in enumerate(relations)}
    id2relation = {i: r for r, i in relation2id.iteritems()}

    obj.train_t = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in train_triples]
    obj.valid_t = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in valid_triples]
    obj.test_t = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in test_triples]
    obj.triples = obj.train_t + obj.valid_t + obj.test_t
    obj.entity2id = entity2id
    obj.id2entity = id2entity
    obj.relation2id = relation2id
    obj.id2relation = id2relation

def split_graph_train_valid_test(obj):
    kg, triples = extract_kg_from_train(obj.triples, mode='split', split_ratio=3.1)

    zip_triples = zip(*triples)
    entities = set(zip_triples[0]) | set(zip_triples[2])
    entities = list(entities)

    n_train_e = int(len(entities) * 0.8)
    train_ind = set(random.sample(range(len(entities)), n_train_e))
    train_e = []
    eval_e = []
    for i in range(len(entities)):
        if i in train_ind:
            train_e.append(entities[i])
        else:
            eval_e.append(entities[i])
    random.shuffle(eval_e)
    mid = int(len(eval_e) * 0.5)
    valid_e = set(eval_e[:mid])
    test_e = set(eval_e[mid:])
    train_e = set(train_e)

    train_t, valid_t, test_t = [], [], []
    for h, r, t in triples:
        if (h in train_e) and (t in train_e):
            train_t.append((h, r, t))
        if (h in valid_e) and (t in valid_e):
            valid_t.append((h, r, t))
        if (h in test_e) and (t in test_e):
            test_t.append((h, r, t))
    return kg, train_t, valid_t, test_t

def split_graph_train_valid_test2(obj):
    zip_triples = zip(*obj.triples)
    entities = set(zip_triples[0]) | set(zip_triples[2])
    entities = list(entities)

    n_train_graph_e = int(len(entities) * 0.8)
    train_graph_ind = set(random.sample(range(len(entities)), n_train_graph_e))
    train_graph_e = []
    eval_e = []
    for i in range(len(entities)):
        if i in train_graph_ind:
            train_graph_e.append(entities[i])
        else:
            eval_e.append(entities[i])
    random.shuffle(eval_e)
    mid = int(len(eval_e) * 0.5)
    valid_e = set(eval_e[:mid])
    test_e = set(eval_e[mid:])
    train_graph_e = set(train_graph_e)

    train_graph_t, valid_t, test_t = [], [], []
    for h, r, t in obj.triples:
        if (h in train_graph_e) and (t in train_graph_e):
            train_graph_t.append((h, r, t))
        if (h in valid_e) and (t in valid_e):
            valid_t.append((h, r, t))
        if (h in test_e) and (t in test_e):
            test_t.append((h, r, t))

    kg, train_t = extract_kg_from_train(train_graph_t, mode='split', split_ratio=3.1)

    return kg, train_t, valid_t, test_t

def check_fb15k237():
    data_dir = os.path.join('data', 'FB15K-237')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj, count_graph=False)
    investigate_interchangeable_triples(obj, count_graph=False)
    investigate_selfloop_triples(obj, count_graph=False)

def make_fb15k237c():
    data_dir_src = os.path.join('data', 'FB15K-237')
    data_dir_tar = os.path.join('data', 'FB15K-237c')
    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    investigate_basic_stats(obj, data_dir_src)
    investigate_selfloop_triples(obj)
    removed_entities = obj.unseen_e_valid | obj.unseen_e_test
    removed_triples = obj.sl_triples
    clean_triples(obj, removed_entities, removed_triples)
    train_triples, valid_triples, test_triples, _ = \
        make_dataset(obj.cleaned_train_t, obj.cleaned_valid_t, obj.cleaned_test_t, obj.id2entity, obj.id2relation)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')

def check_fb15k237c():
    data_dir = os.path.join('data', 'FB15K-237c')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj, count_graph=False)
    investigate_interchangeable_triples(obj, count_graph=False)
    investigate_selfloop_triples(obj, count_graph=False)

def make_fb15k237cinv():
    data_dir_src = os.path.join('data', 'FB15K-237c')
    data_dir_tar = os.path.join('data', 'FB15K-237c-inv')
    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    investigate_basic_stats(obj, data_dir_src)
    investigate_interchangeable_triples(obj, count_graph=False)

    train_t = add_reversed(obj, obj.train_t, obj.ich_triples)
    valid_t = add_reversed(obj, obj.valid_t, obj.ich_triples)
    test_t = add_reversed(obj, obj.test_t, obj.ich_triples)
    train_triples, valid_triples, test_triples, _ = \
        make_dataset(train_t, valid_t, test_t, obj.id2entity, obj.id2relation)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')

def check_fb15k237cinv():
    data_dir = os.path.join('data', 'FB15K-237c-inv')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj, count_graph=False)
    investigate_interchangeable_triples(obj, count_graph=False)
    investigate_selfloop_triples(obj, count_graph=False)

def make_fb15k237c_fullkg():
    data_dir_src = os.path.join('data', 'FB15K-237c')
    data_dir_tar = os.path.join('data', 'FB15K-237c-fullKG')

    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    investigate_basic_stats(obj, data_dir_src)
    kg, new_train_t = extract_kg_from_train(obj.train_t, mode='copy')
    train_triples, valid_triples, test_triples, kg_triples = \
        make_dataset(new_train_t, obj.valid_t, obj.test_t, obj.id2entity, obj.id2relation, kg_t=kg)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')
    write_to_file(kg_triples, data_dir_tar, 'graph.txt')

def check_fb15k237c_fullkg():
    data_dir = os.path.join('data', 'FB15K-237c-fullKG')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj, count_graph=False)
    investigate_interchangeable_triples(obj, count_graph=False)
    investigate_selfloop_triples(obj, count_graph=False)

def make_fb15k237c_splitkg():
    data_dir_src = os.path.join('data', 'FB15K-237c')
    data_dir_tar = os.path.join('data', 'FB15K-237c-splitKG')

    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    investigate_basic_stats(obj, data_dir_src)
    kg, new_train_t = extract_kg_from_train(obj.train_t, mode='split', split_ratio=3.)
    train_triples, valid_triples, test_triples, kg_triples = \
        make_dataset(new_train_t, obj.valid_t, obj.test_t, obj.id2entity, obj.id2relation, kg_t=kg)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')
    write_to_file(kg_triples, data_dir_tar, 'graph.txt')

def check_fb15k237c_splitkg():
    data_dir = os.path.join('data', 'FB15K-237c-splitKG')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj)
    investigate_interchangeable_triples(obj)
    investigate_selfloop_triples(obj)

def make_fb15k237cinv_fullkg():
    data_dir_src = os.path.join('data', 'FB15K-237c-inv')
    data_dir_tar = os.path.join('data', 'FB15K-237c-inv-fullKG')

    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    investigate_basic_stats(obj, data_dir_src)
    kg, new_train_t = extract_kg_from_train(obj.train_t, mode='copy')
    train_triples, valid_triples, test_triples, kg_triples = \
        make_dataset(new_train_t, obj.valid_t, obj.test_t, obj.id2entity, obj.id2relation, kg_t=kg)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')
    write_to_file(kg_triples, data_dir_tar, 'graph.txt')

def check_fb15k237cinv_fullkg():
    data_dir = os.path.join('data', 'FB15K-237c-inv-fullKG')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj, count_graph=False)
    investigate_interchangeable_triples(obj, count_graph=False)
    investigate_selfloop_triples(obj, count_graph=False)

def make_fb15k237cinv_splitkg():
    data_dir_src = os.path.join('data', 'FB15K-237c-inv')
    data_dir_tar = os.path.join('data', 'FB15K-237c-inv-splitKG')

    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    investigate_basic_stats(obj, data_dir_src)
    kg, new_train_t = extract_kg_from_train(obj.train_t, mode='split', split_ratio=3.)
    train_triples, valid_triples, test_triples, kg_triples = \
        make_dataset(new_train_t, obj.valid_t, obj.test_t, obj.id2entity, obj.id2relation, kg_t=kg)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')
    write_to_file(kg_triples, data_dir_tar, 'graph.txt')

def check_fb15k237cinv_splitkg():
    data_dir = os.path.join('data', 'FB15K-237c-inv-splitKG')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj)
    investigate_interchangeable_triples(obj)
    investigate_selfloop_triples(obj)

def make_fb15k237cinv_disjoint1():
    data_dir_src = os.path.join('data', 'FB15K-237c-inv')
    data_dir_tar = os.path.join('data', 'FB15K-237c-inv-Disjoint-I')

    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    combine_train_valid_test(obj, data_dir_src)
    kg, train_t, valid_t, test_t = split_graph_train_valid_test(obj)
    train_triples, valid_triples, test_triples, kg_triples = \
        make_dataset(train_t, valid_t, test_t, obj.id2entity, obj.id2relation, kg_t=kg)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')
    write_to_file(kg_triples, data_dir_tar, 'graph.txt')

def check_fb15k237cinv_disjoint1():
    data_dir = os.path.join('data', 'FB15K-237c-inv-Disjoint-I')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj)
    investigate_interchangeable_triples(obj)
    investigate_selfloop_triples(obj)

def make_fb15k237cinv_disjoint2():
    data_dir_src = os.path.join('data', 'FB15K-237c-inv')
    data_dir_tar = os.path.join('data', 'FB15K-237c-inv-Disjoint-II')

    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    combine_train_valid_test(obj, data_dir_src)
    kg, train_t, valid_t, test_t = split_graph_train_valid_test2(obj)
    train_triples, valid_triples, test_triples, kg_triples = \
        make_dataset(train_t, valid_t, test_t, obj.id2entity, obj.id2relation, kg_t=kg)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')
    write_to_file(kg_triples, data_dir_tar, 'graph.txt')

def check_fb15k237cinv_disjoint2():
    data_dir = os.path.join('data', 'FB15K-237c-inv-Disjoint-II')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj)
    investigate_interchangeable_triples(obj)
    investigate_selfloop_triples(obj)

def check_wn18rr():
    data_dir = os.path.join('data', 'WN18RR')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj, count_graph=False)
    investigate_interchangeable_triples(obj, count_graph=False)
    investigate_selfloop_triples(obj, count_graph=False)

def make_wn18rrc():
    data_dir_src = os.path.join('data', 'WN18RR')
    data_dir_tar = os.path.join('data', 'WN18RRc')
    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    investigate_basic_stats(obj, data_dir_src)
    investigate_selfloop_triples(obj, count_graph=False)
    removed_entities = obj.unseen_e_valid | obj.unseen_e_test
    removed_triples = obj.sl_triples
    clean_triples(obj, removed_entities, removed_triples)
    train_triples, valid_triples, test_triples, _ = \
        make_dataset(obj.cleaned_train_t, obj.cleaned_valid_t, obj.cleaned_test_t, obj.id2entity, obj.id2relation)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')

def check_wn18rrc():
    data_dir = os.path.join('data', 'WN18RRc')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj, count_graph=False)
    investigate_interchangeable_triples(obj, count_graph=False)
    investigate_selfloop_triples(obj, count_graph=False)

def make_wn18rrcinv():
    data_dir_src = os.path.join('data', 'WN18RRc')
    data_dir_tar = os.path.join('data', 'WN18RRc-inv')
    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    investigate_basic_stats(obj, data_dir_src)
    investigate_interchangeable_triples(obj, count_graph=False)

    train_t = add_reversed(obj, obj.train_t, obj.ich_triples)
    valid_t = add_reversed(obj, obj.valid_t, obj.ich_triples)
    test_t = add_reversed(obj, obj.test_t, obj.ich_triples)
    train_triples, valid_triples, test_triples, _ = \
        make_dataset(train_t, valid_t, test_t, obj.id2entity, obj.id2relation)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')

def check_wn18rrcinv():
    data_dir = os.path.join('data', 'WN18RRc-inv')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj, count_graph=False)
    investigate_interchangeable_triples(obj, count_graph=False)
    investigate_selfloop_triples(obj, count_graph=False)

def make_wn18rrc_fullkg():
    data_dir_src = os.path.join('data', 'WN18RRc')
    data_dir_tar = os.path.join('data', 'WN18RRc-fullKG')

    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    investigate_basic_stats(obj, data_dir_src)
    kg, new_train_t = extract_kg_from_train(obj.train_t, mode='copy')
    train_triples, valid_triples, test_triples, kg_triples = \
        make_dataset(new_train_t, obj.valid_t, obj.test_t, obj.id2entity, obj.id2relation, kg_t=kg)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')
    write_to_file(kg_triples, data_dir_tar, 'graph.txt')

def check_wn18rrc_fullkg():
    data_dir = os.path.join('data', 'WN18RRc-fullKG')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj, count_graph=False)
    investigate_interchangeable_triples(obj, count_graph=False)
    investigate_selfloop_triples(obj, count_graph=False)

def make_wn18rrc_splitkg():
    data_dir_src = os.path.join('data', 'WN18RRc')
    data_dir_tar = os.path.join('data', 'WN18RRc-splitKG')

    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    investigate_basic_stats(obj, data_dir_src)
    kg, new_train_t = extract_kg_from_train(obj.train_t, mode='split', split_ratio=3.)
    train_triples, valid_triples, test_triples, kg_triples = \
        make_dataset(new_train_t, obj.valid_t, obj.test_t, obj.id2entity, obj.id2relation, kg_t=kg)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')
    write_to_file(kg_triples, data_dir_tar, 'graph.txt')

def check_wn18rrc_splitkg():
    data_dir = os.path.join('data', 'WN18RRc-splitKG')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj)
    investigate_interchangeable_triples(obj)
    investigate_selfloop_triples(obj)

def make_wn18rrcinv_fullkg():
    data_dir_src = os.path.join('data', 'WN18RRc-inv')
    data_dir_tar = os.path.join('data', 'WN18RRc-inv-fullKG')

    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    investigate_basic_stats(obj, data_dir_src)
    kg, new_train_t = extract_kg_from_train(obj.train_t, mode='copy')
    train_triples, valid_triples, test_triples, kg_triples = \
        make_dataset(new_train_t, obj.valid_t, obj.test_t, obj.id2entity, obj.id2relation, kg_t=kg)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')
    write_to_file(kg_triples, data_dir_tar, 'graph.txt')

def check_wn18rrcinv_fullkg():
    data_dir = os.path.join('data', 'WN18RRc-inv-fullKG')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj, count_graph=False)
    investigate_interchangeable_triples(obj, count_graph=False)
    investigate_selfloop_triples(obj, count_graph=False)

def make_wn18rrcinv_splitkg():
    data_dir_src = os.path.join('data', 'WN18RRc-inv')
    data_dir_tar = os.path.join('data', 'WN18RRc-inv-splitKG')

    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    investigate_basic_stats(obj, data_dir_src)
    kg, new_train_t = extract_kg_from_train(obj.train_t, mode='split', split_ratio=3.)
    train_triples, valid_triples, test_triples, kg_triples = \
        make_dataset(new_train_t, obj.valid_t, obj.test_t, obj.id2entity, obj.id2relation, kg_t=kg)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')
    write_to_file(kg_triples, data_dir_tar, 'graph.txt')

def check_wn18rrcinv_splitkg():
    data_dir = os.path.join('data', 'WN18RRc-inv-splitKG')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj)
    investigate_interchangeable_triples(obj)
    investigate_selfloop_triples(obj)

def make_wn18rrcinv_disjoint1():
    data_dir_src = os.path.join('data', 'WN18RRc-inv')
    data_dir_tar = os.path.join('data', 'WN18RRc-inv-Disjoint-I')

    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    combine_train_valid_test(obj, data_dir_src)
    kg, train_t, valid_t, test_t = split_graph_train_valid_test(obj)
    train_triples, valid_triples, test_triples, kg_triples = \
        make_dataset(train_t, valid_t, test_t, obj.id2entity, obj.id2relation, kg_t=kg)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')
    write_to_file(kg_triples, data_dir_tar, 'graph.txt')

def check_wn18rrcinv_disjoint1():
    data_dir = os.path.join('data', 'WN18RRc-inv-Disjoint-I')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj)
    investigate_interchangeable_triples(obj)
    investigate_selfloop_triples(obj)

def make_wn18rrcinv_disjoint2():
    data_dir_src = os.path.join('data', 'WN18RRc-inv')
    data_dir_tar = os.path.join('data', 'WN18RRc-inv-Disjoint-II')

    if not os.path.exists(data_dir_tar):
        os.mkdir(data_dir_tar)

    obj = Obj()
    combine_train_valid_test(obj, data_dir_src)
    kg, train_t, valid_t, test_t = split_graph_train_valid_test2(obj)
    train_triples, valid_triples, test_triples, kg_triples = \
        make_dataset(train_t, valid_t, test_t, obj.id2entity, obj.id2relation, kg_t=kg)

    write_to_file(train_triples, data_dir_tar, 'train.txt')
    write_to_file(valid_triples, data_dir_tar, 'valid.txt')
    write_to_file(test_triples, data_dir_tar, 'test.txt')
    write_to_file(kg_triples, data_dir_tar, 'graph.txt')

def check_wn18rrcinv_disjoint2():
    data_dir = os.path.join('data', 'WN18RRc-inv-Disjoint-II')
    obj = Obj()
    investigate_basic_stats(obj, data_dir)
    investigate_four_types_of_relations(obj)
    investigate_interchangeable_triples(obj)
    investigate_selfloop_triples(obj)

if __name__ == '__main__':
    ###########################################################################
    #       Preparing the dataset family of FB15K-237

    # Step 1: check the downloaded public dataset FB15K-237
    #check_fb15k237()

    # Step 2: (1) make the new dataset FB15K-237c; (2) check FB15K-237c
    #make_fb15k237c()
    #check_fb15k237c()

    # Step 3: (1) make the new dataset FB15K-237c-inv; (2) check FB15K-237c-inv
    #make_fb15k237cinv()
    #check_fb15k237cinv()

    # Step 4: (1) make the new dataset FB15K-237c-fullKG; (2) check FB15K-237c-fullKG
    #make_fb15k237c_fullkg()
    #check_fb15k237c_fullkg()

    # Step 5: (1) make the new dataset FB15K-237c-splitKG; (2) check FB15K-237c-splitKG
    #make_fb15k237c_splitkg()
    #check_fb15k237c_splitkg()

    # Step 6: (1) make the new dataset FB15K-237c-inv-fullKG; (2) check FB15K-237c-inv-fullKG
    #make_fb15k237cinv_fullkg()
    #check_fb15k237cinv_fullkg()

    # Step 7: (1) make the new dataset FB15K-237c-inv-splitKG; (2) check FB15K-237c-inv-splitKG
    #make_fb15k237cinv_splitkg()
    #check_fb15k237cinv_splitkg()

    # Step 8: (1) make the new dataset FB15K-237c-inv-Disjont-I (2) check FB15K-237c-inv-Disjoint-I
    #make_fb15k237cinv_disjoint1()
    #check_fb15k237cinv_disjoint1()

    # Step 9: (1) make the new dataset FB15K-237c-inv-Disjont-II (2) check FB15K-237c-inv-Disjoint-II
    #make_fb15k237cinv_disjoint2()
    #check_fb15k237cinv_disjoint2()

    ###########################################################################
    #       Preparing the dataset family of WN18RR

    # Step 1: check the downloaded public dataset WN18RR
    #check_wn18rr()

    # Step 2: (1) make the new dataset WN18RRc; (2) check WN18RRc
    #make_wn18rrc()
    #check_wn18rrc()

    # Step 3: (1) make the new dataset WN18RRc-inv; (2) check WN18RRc-inv
    #make_wn18rrcinv()
    #check_wn18rrcinv()

    # Step 4: (1) make the new dataset WN18RRc-fullKG; (2) check WN18RRc-fullKG
    #make_wn18rrc_fullkg()
    #check_wn18rrc_fullkg()

    # Step 5: (1) make the new dataset WN18RRc-splitKG; (2) check WN18RRc-splitKG
    #make_wn18rrc_splitkg()
    #check_wn18rrc_splitkg()

    # Step 6: (1) make the new dataset WN18RRc-inv-fullKG; (2) check WN18RRc-inv-fullKG
    #make_wn18rrcinv_fullkg()
    #check_wn18rrcinv_fullkg()

    # Step 7: (1) make the new dataset WN18RRc-inv-splitKG; (2) check WN18RRc-inv-splitKG
    #make_wn18rrcinv_splitkg()
    #check_wn18rrcinv_splitkg()

    # Step 8: (1) make the new dataset WN18RRc-inv-Disjont-I (2) check WN18RRc-inv-Disjoint-I
    #make_wn18rrcinv_disjoint1()
    #check_wn18rrcinv_disjoint1()

    # Step 9: (1) make the new dataset WN18RRc-inv-Disjont-II (2) check WN18RRc-inv-Disjoint-II
    #make_wn18rrcinv_disjoint2()
    check_wn18rrcinv_disjoint2()