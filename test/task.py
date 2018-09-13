from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import defaultdict

import random
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class IntegerMaze(object):
    def __init__(self, min_int=0, max_int=9999, operand_min=1, operand_max=100, n_ops_per_type=3, n_paths=1, path_length=4, random_seed=1234):
        self.min_int = min_int
        self.max_int = max_int + 1
        self.operand_min = operand_min
        self.operand_max = operand_max + 1
        self.n_ops_per_type = n_ops_per_type
        self.n_paths = n_paths
        self.path_length = path_length

        random.seed(random_seed)
        np.random.seed(random_seed)

        self.op_pool = self._build_op_pool()
        self.paths = self._build_op_paths()
        self.maze = self._build_maze()
        self.train_data, self.test_data = self._build_dataset()

    def _build_op_pool(self):
        pool = []
        operand_min = self.operand_min
        operand_max = self.operand_max
        n_ops_per_type = self.n_ops_per_type
        for operand in random.sample(range(operand_min, operand_max), n_ops_per_type):
            pool.append(lambda x, i=operand: ('+%d' % i, (x+i - self.min_int) % (self.max_int - self.min_int) + self.min_int))
        for operand in random.sample(range(operand_min, operand_max), n_ops_per_type):
            pool.append(lambda x, i=operand: ('-%d' % i, (x-i - self.min_int) % (self.max_int - self.min_int) + self.min_int))
        for operand in random.sample(range(operand_min, operand_max), n_ops_per_type):
            pool.append(lambda x, i=operand: ('*%d' % i, (x*i - self.min_int) % (self.max_int - self.min_int) + self.min_int))
        for operand in random.sample(range(operand_min, operand_max), n_ops_per_type):
            pool.append(lambda x, i=operand: ('/%d' % i, (int(x/i) - self.min_int) % (self.max_int - self.min_int) + self.min_int))
        return pool

    def _build_op_paths(self):
        paths = []
        for i in range(self.n_paths):
            paths.append(np.random.choice(self.op_pool, self.path_length))
        return paths

    def _build_maze(self):
        edges_dict = defaultdict(list)
        for v1 in range(self.min_int, self.max_int):
            for op_id, op in enumerate(self.op_pool):
                v2 = op(v1)[1]
                edges_dict[(v1, v2)].append(op_id)
        folded_edges = sorted(edges_dict.keys())
        unfolded_edges = []
        for e_id, v1_v2 in enumerate(folded_edges):
            for op_id in edges_dict[v1_v2]:
                unfolded_edges.append([e_id, op_id])
        return (unfolded_edges, folded_edges)

    def _build_dataset(self):
        nodes = range(self.min_int, self.max_int)
        random.shuffle(nodes)
        n_nodes = len(nodes)
        n_train_nodes = int(n_nodes*3/4)
        train_nodes = nodes[:n_train_nodes]
        test_nodes = nodes[n_train_nodes:]

        train_data = []
        for v1 in train_nodes:
            for p_id, path in enumerate(self.paths):
                v2 = self._connect_to(v1, path)
                train_data.append((p_id, v1, v2))
        random.shuffle(train_data)

        test_data = []
        for v1 in test_nodes:
            for p_id, path in enumerate(self.paths):
                v2 = self._connect_to(v1, path)
                test_data.append((p_id, v1, v2))
        return train_data, test_data

    def _connect_to(self, v1, path):
        v2 = v1
        for op in path:
            v2 = op(v2)[1]
        return v2

    def folded_edges(self, target):
        if target == 'v1_list':
            return np.array(self.maze[1])[:, 0]
        elif target == 'v2_list':
            return np.array(self.maze[1])[:, 1]
        else:
            raise ValueError('Wrong `target` in folded_edges')

    def unfolded_edges(self, target):
        if target == 'rel_id_list':
            return np.array(self.maze[0])[:, 1]
        elif target == 'edge_id_list':
            return np.array(self.maze[0])[:, 0]
        else:
            raise ValueError('Wrong `target` in unfolded_edges')

    @property
    def n_nodes(self):
        return self.max_int - self.min_int

    @property
    def n_relations(self):
        return len(self.op_pool)

    @property
    def n_edges(self):
        return len(self.maze[1])

    @property
    def n_train(self):
        return len(self.train_data)

    @property
    def n_test(self):
        return len(self.test_data)

    def get_batch(self, i, batch_size, target='train', with_path_id=True):
        if target == 'train':
            if with_path_id:
                return self.train_data[(i*batch_size):((i+1)*batch_size)]
            else:
                return np.array(self.train_data)[(i*batch_size):((i+1)*batch_size), 1:]
        elif target == 'test':
            if with_path_id:
                return self.test_data[(i*batch_size):((i+1)*batch_size)]
            else:
                return np.array(self.test_data)[(i*batch_size):((i+1)*batch_size), 1:]
        else:
            raise ValueError('Wrong `target` in get_batch')

    def n_batches(self, batch_size, target='train'):
        if target == 'train':
            return int(len(self.train_data) / batch_size)
        elif target == 'test':
            return int(len(self.test_data) / batch_size)
        else:
            raise ValueError('Wrong `target` in n_batches')

    def _recursion(self, v1, v2, l):
        c = 0
        for op in self.op_pool:
            v = op(v1)[1]
            if l == 1:
                c += 1 if v==v2 else 0
            else:
                c += self._recursion(v, v2, l-1)
        return c

    def _path_str(self, path):
        return ' -> '.join(map(lambda op: op(0)[0], path))

    def stats(self):
        print('#nodes: %d' % self.n_nodes)
        print('#relations: %d' % self.n_relations)
        print('#edges: %d' % self.n_edges)
        print('#train: %d' % len(self.train_data))
        print('#test: %d' % len(self.test_data))

        for i, op in enumerate(self.op_pool):
            print('relation (op) %d: %s' % (i, op(0)[0]))

        for i, path in enumerate(self.paths):
            print('path %d: %s' % (i, self._path_str(path)))

        n_paths_train, n_paths_test = [], []
        n = 0
        for p_id, v1, v2 in self.train_data:
            if n % 100 == 0:
                print(n)
            c = self._recursion(v1, v2, self.path_length)
            n_paths_train.append(c)
            n += 1
        for p_id, v1, v2 in self.test_data:
            if n % 100 == 0:
                print(n)
            c = self._recursion(v1, v2, self.path_length)
            n_paths_test.append(c)
            n += 1
        print('n_paths_train: min %d, max %d' % (np.min(n_paths_train), np.max(n_paths_train)))
        print('n_paths_test: min %d, max %d' % (np.min(n_paths_test), np.max(n_paths_test)))

        plt.figure(1)
        plt.subplot(211)
        n, bins, _ = plt.hist(n_paths_train, 100, facecolor='g')
        print(n)
        print(bins)
        plt.xlabel('#paths from src to dst')
        plt.ylabel('Count')
        plt.title('Histogram for `n_paths_train`')
        plt.grid(True)
        plt.show()

        plt.subplot(212)
        n, bins, _ = plt.hist(n_paths_test, 50, facecolor='g')
        print(n)
        print(bins)
        plt.xlabel('#paths from src to dst')
        plt.ylabel('Count')
        plt.title('Histogram for `n_paths_test`')
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    maze = IntegerMaze()
    #maze.stats()
