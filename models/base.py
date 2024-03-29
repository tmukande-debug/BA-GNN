#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import utils
from utils import args


eps = 1e-7
inf = 1e31


class Base:
    deep = True
    args = utils.Object()


    def __init__(self, data): 
        self.raw_adjs = data.adjs

        self.save_dir = f'{utils.save_dir}/{args.run_name}'
        self.tb_name = f'{utils.save_dir}/{args.run_name}' 


        self.graph = tf.Graph()
        with self.graph.as_default():
            ##tf.set_random_seed(args.seed)
            self.compile()
        self.fit_step = 0 

    def compile(self):
        self.make_io() #
        self.make_model()
        if args.run_tb: #
            self.all_summary = tf.summary.merge_all() #
            self.tbfw = tf.summary.FileWriter(self.tb_name, self.sess.graph) #

    def placeholder(self, dtype, shape, name, to_list):
        ph = tf.placeholder(dtype, shape, name)
        self.placeholder_dict[name] = ph
        to_list.append(ph) 
        return ph

    
    def make_io(self):
        self.placeholder_dict = {} 
        self.inputs = [] 
        L = args.seq_length
        self.placeholder(tf.int32, [None, L], 'buy_seq', self.inputs)
        self.placeholder(tf.int32, [None, L], 'click_seq', self.inputs)

        self.placeholder(tf.int32, [None], 'pos', self.inputs)
        self.placeholder(tf.int32, [None, None], 'neg', self.inputs)
        self.adjs = [tf.constant(adj, dtype=tf.int32) for adj in self.raw_adjs]  # [N, M] * 4, in_0, out_0, in_1, out_1

  
    def get_data_map(self, data):
        data_map = dict(zip(self.inputs, data))
        return data_map

 
    def make_model(self):
        with tf.variable_scope('Graph', reuse=tf.AUTO_REUSE, regularizer=self.l2_loss('all')) as self.graph_scope: 
            n = args.nb_nodes 
            k = args.dim_k 
            self.embedding_matrix = tf.get_variable(name='emb_w', shape=[n, k]) 
            #embedding_matrix = [nb_nodes,dim_k]
            with tf.variable_scope('graph_agg', reuse=tf.AUTO_REUSE) as self.graph_agg_scope:
                pass

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE, regularizer=self.l2_loss('all')):
            score, label = self.forward(*self.inputs) 
            seq_loss = tf.losses.softmax_cross_entropy(label, score)
            tf.summary.scalar('seq_loss', seq_loss) 
        self.loss = seq_loss
        self.loss += tf.losses.get_regularization_loss() 
        opt = tf.train.AdamOptimizer(learning_rate=args.lr)
        self.minimizer = opt.minimize(self.loss) 
        tf.summary.scalar('loss', self.loss)


        graph_var_list = tf.trainable_variables(scope='^Graph/') 
        network_var_list = tf.trainable_variables(scope='^Network/') 
        for v in graph_var_list:
            print('graph', v)
        for v in network_var_list:

            print('network', v)

        self.saver = tf.train.Saver()
        self.sess = self.get_session()
        self.sess.run(tf.global_variables_initializer()) 
    
    def get_session(self):
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=1,
            visible_device_list=args.gpu,
            allow_growth=True,
        )
        config = tf.ConfigProto(gpu_options=gpu_options)
        session = tf.Session(config=config)
        return session

 
    def fit(self, data): 
        data = dict(zip(self.inputs, data))
        if args.run_tb:
            _, loss, summary = self.sess.run([self.minimizer, self.loss, self.all_summary], data) 
            self.tbfw.add_summary(summary, self.fit_step) 
        else:
            _, loss = self.sess.run([self.minimizer, self.loss], data)
        self.fit_step += 1
        return loss

    def topk(self, data):
        data = self.get_data_map(data)
        return self.sess.run([self.topkV, self.topkI], data)

    def save(self):
        name = f'{self.save_dir}/model.ckpt'
        self.saver.save(self.sess, name)

    def restore(self): #存储
        name = f'{self.save_dir}/model.ckpt'
        try:
            self.saver.restore(self.sess, name)
        except Exception as e:
            print(f'can not restore model: {name}')
            raise e

    def l2_loss(self, name):
        alpha = args.get(f'l2_{name}', 0)
        if alpha < 1e-7:
            return None
        return lambda x: alpha * tf.nn.l2_loss(x)

    def forward(self, buy_seq, click_seq, pos, neg):
        pos2 = tf.expand_dims(pos, -1) 
        nxt = tf.concat([pos2, neg], -1)  # [BS, M + 1] 
        label = tf.concat([tf.ones_like(pos2, dtype=tf.int32), tf.zeros_like(neg, dtype=tf.int32)], -1)  # [BS, M + 1]  
                                                                                                       

        seq_emb = self.merge_seq(buy_seq, click_seq) 
        seq_emb = tf.layers.dense(seq_emb, args.dim_k, name='dense_W', use_bias=False) 
        score = tf.matmul(seq_emb, self.embedding_matrix, transpose_b=True) 

        topk = tf.math.top_k(score, k=500) 
        self.topkV = topk.values
        self.topkI = topk.indices 

        nxt_embs, _ = self.Embedding(nxt)  # [BS, M + 1, k]  
        nxt_score = tf.reduce_sum(tf.expand_dims(seq_emb, 1) * nxt_embs, -1)
        return nxt_score, label

    def merge_seq(self, buy_seq, click_seq):
        with tf.variable_scope(f'merge_seq', reuse=tf.AUTO_REUSE):
            buy_seq_embs, buy_mask = self.node_embedding(buy_seq)
            buy_emb = self.seq_embedding(buy_seq_embs, buy_mask, 'buy')
            click_seq_embs, click_mask = self.node_embedding(click_seq)
            click_emb = self.seq_embedding(click_seq_embs, click_mask, 'click')

            emb = self.gate(buy_emb, click_emb, 'merge_buy_and_click_seq')
            return emb

    def node_embedding(self, node):
        # node: [BS, L]
        embs, mask = self.Embedding(node, mask_zero=True)
        return embs, mask

    def seq_embedding(self, seq, mask, name):
        # seq: [BS, L, k]
        with tf.variable_scope(f'seq_embedding_{name}', reuse=tf.AUTO_REUSE):
            seq_emb = self.Mean(seq, mask=mask)
        return seq_emb

    def Mean(self, seq, seq_length=None, mask=None, name=None):
        # seq: (None, L, k), seq_length: (None, ), mask: (None, L)
        # ret: (None, k)
        if seq_length is None and mask is None:
            with tf.variable_scope('Mean'):
                return tf.reduce_sum(seq, -2)

        with tf.variable_scope('MaskMean'): 
            if mask is None: 
                mask = tf.sequence_mask(seq_length, maxlen=tf.shape(seq)[1], dtype=tf.float32)
       
            mask = tf.expand_dims(mask, -1)  
            seq = seq * mask
            seq = tf.reduce_sum(seq, -2)  # (None, k)  
            seq = seq / (tf.reduce_sum(mask, -2) + eps)
        return seq

    def gate(self, a, b, name):
        with tf.variable_scope(name):
            alpha = tf.layers.dense(tf.concat([a, b], -1), 1, activation=tf.nn.sigmoid, name='gateW') 
            ret = alpha * a + (1 - alpha) * b  
        return ret

    def Embedding(self, node, name='node', mask_zero=False):
        # node: [BS]
        with tf.variable_scope(f'Emb_{name}'):
            emb_w = self.embedding_matrix  #self.embedding_matrix = tf.get_variable(name='emb_w', shape=[n, k]) 
                                        
            t = tf.gather(emb_w, node) 
                                        
            if mask_zero: 
                mask = tf.not_equal(node, 0) 
                mask = tf.cast(mask, tf.float32) 
            else:
                mask = None
        return t, mask


class GNN(Base):
    args = Base.args.copy().update()
    def node_embedding(self, node):
        # node: [BS, L] node:buy_seq/click_seq
        with tf.variable_scope(self.graph_scope):
            embs, mask = self.node_list_aggregate(node, depth=args.gnnt, mask_zero=True)
        return embs, mask

    def node_list_aggregate(self, nodes, depth, mask_zero, name='buy'):
        # nodes: [BS, M]
        bs = tf.shape(nodes)[0] 
        m = tf.shape(nodes)[1]  
        nodes = tf.reshape(nodes, [bs * m])
        emb0, mask = self.single_node_aggregate0(nodes, depth, mask_zero, name)
        emb1, mask = self.single_node_aggregate1(nodes, depth, mask_zero, name)
        # k = tf.shape(emb)[1]
        emb = self.gate(emb0, emb1, 'merge_buy_and_click_emb')
        k = args.dim_k
        emb = tf.reshape(emb, [bs, m, k])

        if mask is not None:
            mask = tf.reshape(mask, [bs, m])
        return emb, mask

    def single_node_aggregate0(self, node, depth, mask_zero, name='buy'):
        # node: [BS], adj: [N, M]
        if depth <= 0:
            return self.Embedding(node, mask_zero=mask_zero)
        with tf.variable_scope(f'agg{depth}layer_{name}'):
            bs = tf.shape(node)[0]
            cur, cur_mask = self.single_node_aggregate0(node, depth - 1, mask_zero, name)  # [BS, k]

            nxt_in_0 = tf.gather(self.adjs[0], node)  # [BS, M]
            nxt_in_0, nxt_in_0_mask = self.node_list_aggregate(nxt_in_0, depth - 1, mask_zero, name)  # [BS, M, k]

            nxt_out_0 = tf.gather(self.adjs[1], node)  # [BS, M]
            nxt_out_0, nxt_out_0_mask = self.node_list_aggregate(nxt_out_0, depth - 1, mask_zero, name)  # [BS, M, k]

      
           
            h0 = self._aggregate0(cur,nxt_in_0, nxt_in_0_mask, nxt_out_0, nxt_out_0_mask,name)
    
            return h0, cur_mask

    def single_node_aggregate1(self, node, depth, mask_zero, name='click'):
        # node: [BS], adj: [N, M]
        if depth <= 0:
            return self.Embedding(node, mask_zero=mask_zero)
        with tf.variable_scope(f'agg{depth}layer_{name}'):
            bs = tf.shape(node)[0]
            cur, cur_mask = self.single_node_aggregate1(node, depth - 1, mask_zero, name)  # [BS, k]

            nxt_in_1 = tf.gather(self.adjs[2], node)  
            nxt_in_1, nxt_in_1_mask = self.node_list_aggregate(nxt_in_1, depth - 1, mask_zero, name)  # [BS, M, k]

            nxt_out_1 = tf.gather(self.adjs[3], node)  
            nxt_out_1, nxt_out_1_mask = self.node_list_aggregate(nxt_out_1, depth - 1, mask_zero, name)  # [BS, M, k]

           

            h1 = self._aggregate1(cur, nxt_in_1, nxt_in_1_mask, nxt_out_1, nxt_out_1_mask, name)

            return h1, cur_mask

    def scaled_dot_product_attention(self, Q, K, V,
                                     causality=False, dropout_rate=0.,
                                     training=True,
                                     scope="scaled_dot_product_attention"):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            d_k = Q.get_shape().as_list()[-1]
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
            outputs /= d_k ** 0.5
            # key_masks: [N, key_seqlen]
            # outputs = mask(outputs, key_masks=key_masks, type="key")
            # causality or future blinding masking
            if causality:
                outputs = mask(outputs, type="future")
   
            outputs = tf.nn.softmax(outputs)

            attention = tf.transpose(outputs, [0, 2, 1])

            tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

            # dropout
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

        return outputs

    def sparsemax(self,inputs, epsilon=1e-8, scope="sparsemax"):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
          
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())

            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())

            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))

            outputs = gamma * normalized + beta

        return outputs

    def multihead_attention(self, queries, keys, values,
                            num_heads=8,
                            dropout_rate=0,
                            training=True,
                            causality=False,
                            scope="multihead_attention"):
        d_model = queries.get_shape().as_list()[-1]

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Linear projections
            # 前向传播，Q,K,V计算
            Q = tf.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
            K = tf.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
            V = tf.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

            # Q_, K_, V_ 计算Attention
            outputs = self.scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

            # Residual connection
            outputs += queries

            # Normalize (N, T_q, d_model)
            outputs = self.sparsemax(outputs)

        return outputs

    def _aggregate0(self, cur, nxt_in, nxt_in_mask, nxt_out, nxt_out_mask, name):
        with tf.variable_scope(self.graph_agg_scope): 

            nxt_in = self._agg(nxt_in, nxt_in_mask)
            nxt_out = self._agg(nxt_out, nxt_out_mask)

            nxt = nxt_in + nxt_out

            o = self._merge(cur, nxt)
            return o

    def _aggregate1(self, cur, nxt_in, nxt_in_mask, nxt_out, nxt_out_mask, name):
        with tf.variable_scope(self.graph_agg_scope):  

            nxt_in = self.multihead_attention(nxt_in, nxt_in, nxt_in)
            nxt_out = self.multihead_attention(nxt_out, nxt_out, nxt_out)
            nxt_in = self._agg(nxt_in, nxt_in_mask)
            nxt_out = self._agg(nxt_out, nxt_out_mask)

            nxt = nxt_in + nxt_out

            o = self._merge(cur, nxt)
            return o

    def _agg(self, nxt, mask):
        return self.Mean(nxt, mask=mask)

    def _merge(self, cur, nxt):
        return cur + nxt
