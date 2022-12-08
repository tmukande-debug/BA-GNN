import numpy as np, sys, math, os
import json
import pickle
import copy
import utils
import re
from utils import args, tqdm
import time

data_home = 'run_time/data'
class DatasetReader:
    def __init__(self, ds):
        self.ds = ds
        if ds == 'yc64':
            self.N = 144527
            self.reader = self.yc64
            self.min_ts, self.max_ts = 1411604904, 1412017199  # 412295
        elif ds == 'yc4':
            self.N = 2312432
            self.reader = self.yc4
            self.min_ts, self.max_ts = 1408507486, 1412017199 # 3509713
        elif ds == 'yc':
            self.N = 9249729 #

            self.reader = self.yc
            self.min_ts, self.max_ts = 1396292400, 1412017199  # 15724799
        elif ds == 'kaggledec':
            self.N = 358565
            self.reader = self.kaggledec
            self.min_ts, self.max_ts = 1575129605, 1577776237
        elif ds == 'kaggle':
            self.N = 1356762
            self.reader = self.kaggle
            self.min_ts, self.max_ts = 1569859226, 1582523045
        dt = self.max_ts - self.min_ts
        dt2 = dt // 7 * 3
        #dt3 = dt // 7 * 6 # 913361  5480166
        self.train_ts = self.min_ts + dt2 * 2
        #self.train_ts = self.min_ts + dt3  #1522894637
        #self.train_ts = 1523801969  #full：1523016413 #1521615650  #128：1523801969

    def yc(self, frac=1):
        pbar = tqdm(desc='read data', total=self.N)
        f = open(f'{data_home}/yc_1_{frac}/data.txt', 'r')
        for line in f:
            pbar.update(1)
            line = line[:-1]
            sid, vid_list_str = line.split()
            vid_list = []
            for vid in vid_list_str.split(','):
                vid, cls, ts = vid.split(':')
                cls = int(cls)
                ts = int(ts)
                vid_list.append([vid, cls, ts])
            yield vid_list
        print("vid_list:", len(vid_list))
        print(vid_list)
        f.close()
        pbar.close()

    def yc4(self):
        yield from self.yc(4)

    def yc64(self):
        yield from self.yc(64)

    def kaggle(self, frac=1):
        pbar = tqdm(desc='read data', total=self.N)
        f = open(f'{data_home}/kaggle_1_{frac}/data.txt', 'r')
        for line in f:
            pbar.update(1)
            line = line[:-1]
            sid, vid_list_str = line.split()
            vid_list = []
            for vid in vid_list_str.split(','):
                vid, cls, ts = vid.split(':')
                cls = int(cls)
                ts = int(ts)
                vid_list.append([vid, cls, ts])
            yield vid_list
        print("vid_list:", len(vid_list))
        print(vid_list)
        f.close()
        pbar.close()

    def kaggledec(self, frac=1):
        pbar = tqdm(desc='read data', total=self.N)
        f = open(f'{data_home}/kaggledec_1_{frac}/data.txt', 'r')
        for line in f:
            pbar.update(1)
            line = line[:-1]
            sid, vid_list_str = line.split()
            vid_list = []
            for vid in vid_list_str.split(','):
                vid, cls, ts = vid.split(':')
                cls = int(cls)
                ts = int(ts)
                vid_list.append([vid, cls, ts])
            yield vid_list
        print("vid_list:", len(vid_list))
        print(vid_list)
        f.close()
        pbar.close()

class DataProcess:
    def __init__(self, ds, adj_length, seq_length):
        self.ds = ds
        # self.adj_length = adj_length
        # self.seq_length = seq_length

        self.vid2node = {}
        self.vid2node['[MASK]'] = 0

        self.DR = DatasetReader(ds)
        self.G_in, self.G_out, self.train_data, self.test_data = self.build_graph(seq_length)

        print('train_data:',len(self.train_data))
        print('test_data:',len(self.test_data))

        rdm = np.random.RandomState(777)
        rdm.shuffle(self.train_data)

        rdm = np.random.RandomState(333)
        rdm.shuffle(self.test_data)

        args.update(nb_nodes=len(self.vid2node))
        args.update(nb_edges_0=self.G_in[0].nb_edges()) #0是buy
        args.update(nb_edges_1=self.G_in[1].nb_edges()) #1是click

        #0:buy  1:click
        self.adj_in_0 = self.build_adj(self.G_in[0], adj_length)
        self.adj_out_0 = self.build_adj(self.G_out[0], adj_length)
        self.adj_in_1 = self.build_adj(self.G_in[1], adj_length)
        self.adj_out_1 = self.build_adj(self.G_out[1], adj_length)

        self.adjs_tmp = [self.adj_in_0, self.adj_out_0, self.adj_in_1, self.adj_out_1]
        self.adjs = [a[0] for a in self.adjs_tmp]

    def build_graph(self, seq_length):
        test_seq = []
        G_in = [utils.Graph() for i in range(2)]
        G_out = [utils.Graph() for i in range(2)]
        train_data = []
        test_data = []

        for num_data, vid_list in enumerate(self.DR.reader()):#

            vid_list_for_graph = [[] for i in range(2)]
            vid_list_for_train = [[] for i in range(2)]
            first_pos = [{} for i in range(2)]

            for i, (vid, typ, ts) in enumerate(vid_list):
                if vid not in self.vid2node:
                    self.vid2node[vid] = len(self.vid2node)

                for_train = False
                if ts < self.DR.train_ts: #
                    for_train = True

                if for_train:
                    vid_list_for_graph[typ].append(vid) #

                if typ == 0 and vid not in first_pos[0]:
                    buy_history = vid_list_for_train[0]
                    if vid not in first_pos[1]: #
                        click_history = vid_list_for_train[1]

                    else: #
                        k = first_pos[1][vid]
                        click_history = vid_list_for_train[1][:k]

                    if len(click_history) >= 5 and len(buy_history) >= 1:
                        seq_buy = [buy_history[-seq_length: ], click_history[-seq_length: ], vid]
                        if for_train:
                            train_data.append(seq_buy)
                        else:
                            test_data.append(seq_buy)

                if vid not in first_pos[typ]:
                    first_pos[typ][vid] = len(vid_list_for_train[typ])
                vid_list_for_train[typ].append(vid)


            for typ in range(2):
                for i, vid in enumerate(vid_list_for_graph[typ]):
                    if i == 0:
                        continue
                    now_node = self.vid2node[vid]
                    pre_node = self.vid2node[vid_list_for_graph[typ][i - 1]]
                    if now_node != pre_node:
                        G_in[typ].add_edge(pre_node, now_node)
                        G_out[typ].add_edge(now_node, pre_node)
                    else:
                        pass

        return G_in, G_out, train_data, test_data

    def build_adj(self, G, M):
        # M: number of adj per node
        N = args.nb_nodes
        # adj shape: [N, M]
        adj = [None] * N
        adj[0] = [0] * M
        w = [None] * N
        w[0] = [0] * M

        rdm = np.random.RandomState(555)
        pbar = tqdm(total=N - 1, desc='building adj')
        for node in range(1, N):
            pbar.update(1)
            adj_list = G.get_adj(node)
            if len(adj_list) > M:
                adj_list = rdm.choice(adj_list, size=M, replace=False).tolist()
            mask = [0] * (M - len(adj_list))
            adj_list = adj_list[:] + mask
            adj[node] = adj_list
            w_list = [G.edge_cnt.get((node, x), 0) for x in adj_list]
            w[node] = w_list
        pbar.close()
        return [adj, w]

class Data:
    def __init__(self):
        self.dp = DataProcess(args.ds, args.adj_length, args.seq_length)

        self.adjs = self.dp.adjs
        self.vid2node = self.dp.vid2node

        self.load_data()
        self.status = 'train'

    def load_data(self):
        self.data = self.dp.train_data + self.dp.test_data
        nb_train = len(self.dp.train_data)
        nb_non_train = len(self.dp.test_data)
        nb_vali = nb_non_train // 3
        nb_test = nb_non_train - nb_vali

        nb_data = len(self.data)
        assert nb_data > 0
        args.update(nb_data=nb_data, nb_train=nb_train, nb_vali=nb_vali, nb_test=nb_test)

    def pad_seq(self, node_list):
        L = args.seq_length
        if len(node_list) < L:
            node_list = node_list + [0] * (L - len(node_list))
        return node_list

    def sample_neg(self, pos, rdm):
        neg = set()
        while len(neg) < args.num_neg:
            n = rdm.randint(args.nb_nodes)
            if n != 0 and n != pos and n not in neg:
                neg.add(n)
        neg = sorted(neg)
        return neg

    def get_data_by_idx(self, idx, rdm):#
        buy_history, click_history, pos = self.data[idx]

        pos = self.vid2node[pos]

        buy_seq = [self.vid2node[vid] for vid in buy_history]
        click_seq = [self.vid2node[vid] for vid in click_history]

        buy_list = self.pad_seq(buy_seq)
        click_list = self.pad_seq(click_seq)

        ret = [buy_list, click_list, pos]

        if self.status == 'train':
            neg = self.sample_neg(pos, rdm)
            ret.append(neg)
        return ret

    def get_batch_by_idxs(self, idxs, rdm=None):
        data = None
        for idx in idxs:
            d = self.get_data_by_idx(idx, rdm)
            n = len(d)
            if data is None:
                data = [[] for _ in range(n)]
            for i in range(n):
                data[i].append(d[i])

        # data: [0-seq, 1-typ, 2-len, 3-nxt, 4-label]
        batch = [np.array(d) for d in data]
        return batch

    def gen_train_batch_for_train(self, batch_size):
        rdm = np.random.RandomState(333)
        while True:
            idxs = list(range(args.nb_train))
            rdm.shuffle(idxs)
            for i in range(0, args.nb_train, batch_size):
                batch = self.get_batch_by_idxs(idxs[i: i + batch_size], rdm)
                yield batch

    def get_data_idxs(self, name):
        if name == 'train':
            return 0, args.nb_train
        if name == 'vali':
            return args.nb_train, args.nb_train + args.nb_vali
        if name == 'test':
            return args.nb_train + args.nb_vali, args.nb_data

    def gen_metric_batch(self, name, batch_size):
        self.status = 'metric'
        begin_idx, end_idx = self.get_data_idxs(name)
        yield from self.gen_data_batch(begin_idx, end_idx, batch_size)
        self.status = 'train'

    def gen_all_batch(self, batch_size):
        begin_idx = 0
        end_idx = args.nb_data
        yield from self.gen_data_batch(begin_idx, end_idx, batch_size)

#
    def gen_data_batch(self, begin_idx, end_idx, batch_size):
        for i in range(begin_idx, end_idx, batch_size):
            a, b = i, min(end_idx, i + batch_size)
            batch = self.get_batch_by_idxs(range(a, b))
            yield batch


def main():
    pass


if __name__ == '__main__':
    main()
