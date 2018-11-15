#coding=utf-8
'''
Created on 2018.11.3

@author: DUTIRLAB
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.optimizers import RMSprop

from keras.preprocessing import sequence

import pickle as pkl
import gzip

from keras import utils



from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf

import tensorflow_hub as hub

import keras.layers as layers



class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+10) * (1 - x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention():

    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
                return x

            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = [];
            attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head);
                attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        outputs = Add()([outputs, q])
        return self.layer_norm(outputs), attn



class AttentionWithContext(Layer):


    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):

        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        mul_a = uit * self.u
        ait = K.sum(mul_a, axis=2)

        a = K.exp(ait)

        if mask is not None:

            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):

        return (input_shape[0], input_shape[-1])

def ElmoEmbedding(x):
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x[0], tf.string)),
                            "sequence_len":tf.squeeze(tf.cast(x[1], tf.int32))
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]


def categorical_F(y_true, y_pred, false_index):
    assert len(y_true), len(y_pred)

    pre01_matrix = np.zeros_like(y_true, dtype=np.int8)
    premax_indexs = np.argmax(y_pred, -1)
    ans01_matrix = np.zeros_like(y_true, dtype=np.int8)
    ansmax_indexs = np.argmax(y_true, -1)

    for i in range(len(premax_indexs)):
        pre01_matrix[i][premax_indexs[i]] = 1
        ans01_matrix[i][ansmax_indexs[i]] = 1
    nb_class = y_true.shape[-1]

    result_matrixs = np.zeros((7, nb_class + 2))

    avgp = avgr = 0.0
    for j in range(nb_class):
        pre01_column = np.array(pre01_matrix[:, j])
        ans01_column = np.array(ans01_matrix[:, j])
        tp = result_matrixs[0][j] = np.sum(pre01_column * ans01_column)  # tp
        fp = result_matrixs[1][j] = np.sum(pre01_column * (1 - ans01_column))  # fp
        fn = result_matrixs[2][j] = np.sum((1 - pre01_column) * ans01_column)  # fn +fnaray[j]
        if (tp + fp) == 0.:
            p = result_matrixs[3][j] = 0.  # p
        else:
            p = result_matrixs[3][j] = float(tp) / (tp + fp)  # p

        if (tp + fn) == 0.:
            r = result_matrixs[4][j] = 0.
        else:
            r = result_matrixs[4][j] = float(tp) / (tp + fn)  # r

        if (p + r) == 0:
            result_matrixs[5][j] = 0.
        else:
            result_matrixs[5][j] = 2 * p * r / (p + r)

        positive = result_matrixs[6][j] = np.sum(ans01_column)  # positive instance    #micro average
        if j != (false_index):
            avgp = avgp + p
            avgr = avgr + r
            result_matrixs[0][nb_class] = result_matrixs[0][nb_class] + tp
            result_matrixs[1][nb_class] = result_matrixs[1][nb_class] + fp
            result_matrixs[2][nb_class] = result_matrixs[2][nb_class] + fn
            result_matrixs[6][nb_class] = result_matrixs[6][nb_class] + positive

    # macro average
    avgp = avgp / (nb_class - 1)  # (nb_class-1)
    avgr = avgr / (nb_class - 1)  # (nb_class-1)
    if (avgp + avgr) == 0:
        avgf = 0.
    else:
        avgf = (2 * avgp * avgr) / (avgp + avgr)
    # mincro average
    sumtp = result_matrixs[0][nb_class]
    sumfp = result_matrixs[1][nb_class]
    sumfn = result_matrixs[2][nb_class]
    sumpositive = result_matrixs[6][nb_class]

    if (sumtp + sumfp) == 0:
        microp = 0.
    else:
        microp = float(sumtp) / (sumtp + sumfp)
    if sumpositive == 0:
        micror = 0.
    else:
        micror = float(sumtp) / sumpositive
    if (microp + micror) == 0.:
        microF = 0.
    else:
        microF = (2 * microp * micror) / (microp + micror)
    result_matrixs[3][nb_class] = microp
    result_matrixs[4][nb_class] = micror
    result_matrixs[5][nb_class] = microF

    result_matrixs[3][nb_class + 1] = avgp
    result_matrixs[4][nb_class + 1] = avgr
    result_matrixs[5][nb_class + 1] = avgf
    return result_matrixs, premax_indexs, ansmax_indexs


if __name__ == '__main__':

        s = {

             'batch_size':64,
             'epochs':60,
             'class_num':6,
            'emb_dropout':0.5,
            'dense_dropout':0.5,
            'train_file': "./chemprot_train.pkl.gz",
            'development_file': "./chemprot_development.pkl.gz",
            'test_file': "./chemprot_test.pkl.gz",

            'rnn_unit':300,

            }


        f_Train = gzip.open(s['train_file'], 'rb')

        train_labels_vec = pkl.load(f_Train)
        train_all_words = pkl.load(f_Train)

        train_all_dis1 = pkl.load(f_Train)
        train_all_dis2 = pkl.load(f_Train)
        train_part_sequence = pkl.load(f_Train)
        train_pos = pkl.load(f_Train)


        f_Train.close()

        train_all_words_length=[0]*len(train_all_words)
        for i in range(len(train_all_words)):
            train_all_words_length[i]=len(train_all_words[i])

        f_Develop = gzip.open(s['development_file'], 'rb')

        develop_labels_vec = pkl.load(f_Develop)
        develop_all_words = pkl.load(f_Develop)

        develop_all_dis1 = pkl.load(f_Develop)
        develop_all_dis2 = pkl.load(f_Develop)
        # train_entity_sequence = pkl.load(f_Train)

        develop_part_sequence = pkl.load(f_Develop)
        develop_pos = pkl.load(f_Develop)

        f_Develop.close()

        develop_all_words_length = [0] * len(develop_all_words)
        for i in range(len(develop_all_words)):
            develop_all_words_length[i] = len(develop_all_words[i])


        train_labels_vec+=develop_labels_vec
        train_all_words+=develop_all_words
        train_all_dis1+=develop_all_dis1
        train_all_dis2+=develop_all_dis2
        train_pos+=develop_pos

        train_part_sequence += develop_part_sequence

        train_all_words_length+=develop_all_words_length




        f_Test = gzip.open(s['test_file'], 'rb')
        test_labels_vec = pkl.load(f_Test)
        test_all_words = pkl.load(f_Test)

        test_all_dis1 = pkl.load(f_Test)
        test_all_dis2 = pkl.load(f_Test)

        test_part_sequence = pkl.load(f_Test)
        test_pos = pkl.load(f_Test)

        f_Test.close()


        test_all_words_length = [0] * len(test_all_words)
        for i in range(len(test_all_words)):
            test_all_words_length[i] = len(test_all_words[i])

        train_labels = train_labels_vec
        test_labels=test_labels_vec

        pos_dic={}
        pos_index=0
        for sentence in train_pos:
            for instance in sentence:
                if instance not in pos_dic:
                    pos_dic[instance]=pos_index
                    pos_index+=1

        for sentence in test_pos:
            for instance in sentence:
                if instance not in pos_dic:
                    pos_dic[instance]=pos_index
                    pos_index+=1

        new_train_pos=[]
        new_test_pos=[]
        for sentence in train_pos:
            temp_list=[]
            for instance in sentence:
                temp_list.append(pos_dic[instance])
            new_train_pos.append(temp_list)

        for sentence in test_pos:
            temp_list=[]
            for instance in sentence:
                temp_list.append(pos_dic[instance])
            new_test_pos.append(temp_list)

        train_pos=new_train_pos

        test_pos=new_test_pos

        max_length_all_words=max(train_all_words_length+test_all_words_length)


        new_train_all_words = []
        for seq in train_all_words:
            new_seq = []
            for i in range(max_length_all_words):
                try:
                    new_seq.append(seq[i])
                except:
                    new_seq.append("__PAD__")
            new_train_all_words.append(new_seq)
        train_all_words = new_train_all_words

        new_test_all_words = []
        for seq in test_all_words:
            new_seq = []
            for i in range(max_length_all_words):
                try:
                    new_seq.append(seq[i])
                except:
                    new_seq.append("__PAD__")
            new_test_all_words.append(new_seq)
        test_all_words = new_test_all_words

        train_all_words = np.array(train_all_words)
        test_all_words = np.array(test_all_words)

        train_all_words_length = np.array(train_all_words_length)
        test_all_words_length = np.array(test_all_words_length)

        train_y = utils.to_categorical(train_labels, num_classes=s['class_num'])
        #print (train_y)

        test_y = utils.to_categorical(test_labels, num_classes=s['class_num'])

        # initialize elmo_model
        sess = tf.Session()
        K.set_session(sess)

        tf.logging.set_verbosity(tf.logging.ERROR)


        elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        print("elmo_model initialized finished")


        train_all_dis1 = sequence.pad_sequences(train_all_dis1, maxlen=max_length_all_words,
                                               truncating='post', padding='post')
        test_all_dis1 = sequence.pad_sequences(test_all_dis1, maxlen=max_length_all_words,
                                              truncating='post',
                                              padding='post')

        train_all_dis2 = sequence.pad_sequences(train_all_dis2, maxlen=max_length_all_words,
                                               truncating='post', padding='post')
        test_all_dis2 = sequence.pad_sequences(test_all_dis2, maxlen=max_length_all_words,
                                              truncating='post',
                                              padding='post')

        train_pos = sequence.pad_sequences(train_pos, maxlen=max_length_all_words,   truncating='post', padding='post')
        test_pos = sequence.pad_sequences(test_pos, maxlen=max_length_all_words,  truncating='post',  padding='post')

        result_out = open("./CPI_extraction_output.txt", 'w+')

        p_list = []
        r_list = []
        f_list = []

        #### training repeat 10 times to reduce the selection bias
        for random_times in range(10):

            ##position embedding

            disembedding = Embedding(650,
                                     100,
                                     )


            posembedding = Embedding(100,
                                     100
                                     )

            input_all_dis1 = Input(shape=(max_length_all_words,), dtype='int32', name='input_all_dis1')
            all_dis_fea1 = disembedding(input_all_dis1)

            input_all_dis2 = Input(shape=(max_length_all_words,), dtype='int32', name='input_all_dis2')
            all_dis_fea2 = disembedding(input_all_dis2)

            input_pos = Input(shape=(max_length_all_words,), dtype='int32', name='input_pos')
            pos_fea = posembedding(input_pos)


            input_all_word_string = layers.Input(shape=(max_length_all_words,), dtype=tf.string)
            input_all_word_max_length = layers.Input(shape=(1,), dtype=tf.int32)

            input_all_word_string_embedding = layers.Lambda(ElmoEmbedding, output_shape=(max_length_all_words,1024))([input_all_word_string,input_all_word_max_length])

            emb_merge = layers.concatenate([input_all_word_string_embedding, all_dis_fea1, all_dis_fea2, pos_fea],
                                           axis=-1)
            emb_merge = Dropout(0.5)(emb_merge)

            left_lstm = LSTM(output_dim=s['rnn_unit'],
                                      init='orthogonal',
                                      activation='tanh',
                                      inner_activation='sigmoid',
                                      recurrent_dropout=0.2, dropout=0.2,
                                        return_sequences=True,
                                      )(emb_merge)

            right_lstm = LSTM(output_dim=s['rnn_unit'],
                                       init='orthogonal',
                                       activation='tanh',
                                       inner_activation='sigmoid',
                                       recurrent_dropout=0.2, dropout=0.2,
                                        return_sequences=True,
                                       go_backwards=True)(emb_merge)

            emb_lstm = layers.concatenate([left_lstm, right_lstm],
                                          axis=-1)

            att_layer = MultiHeadAttention(6, 600, 100, 100, dropout=0.2,mode=1)
            att_output, attn = att_layer(emb_lstm, emb_lstm, emb_lstm, mask=None)

            double_att = AttentionWithContext()(att_output)

            classify_drop = Dropout(0.5)(double_att)
            classify_output = Dense(s['class_num'])(classify_drop)
            classify_output = Activation('softmax')(classify_output)

            model = Model(
                inputs=[input_all_word_string,input_all_word_max_length,input_all_dis1,input_all_dis2,input_pos],
                outputs=classify_output)

            keras_opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)

            model.compile(loss='categorical_crossentropy', optimizer=keras_opt,metrics=['accuracy'])
            model.summary()

            batch_size=s['batch_size']

            history = []
            max_f = max_p = max_r = 0

            early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')

            print('-----------------Begin of training---------->'  + '\n')


            History = model.fit(
                [train_all_words,train_all_words_length,train_all_dis1,train_all_dis2,train_pos ],
                train_y,
                batch_size=s['batch_size'], epochs=s['epochs'], verbose=1,
                callbacks=[early_stopping], validation_split=0.1)

            pred_test = model.predict(
                [test_all_words,test_all_words_length,test_all_dis1,test_all_dis2,test_pos],
                batch_size=s['batch_size'])
            print(test_all_words.shape)


            resultF_matrix, premax_indexs, ansmax_indexs = categorical_F(test_y, pred_test, 0)

            precision, recall, F1=resultF_matrix[3][6],resultF_matrix[4][6],resultF_matrix[5][6]


            print("random times:" + str(random_times) + ' precision:' + str(
                np.round(precision, 5)) + ' recall:' + str(
                np.round(recall, 5)) + ' F1:' + str(np.round(F1, 5)))


            p_list.append(precision)
            r_list.append(recall)
            f_list.append(F1)

        p_array = np.array(p_list)
        r_array = np.array(r_list)
        f_array = np.array(f_list)
        avg_p = np.average(p_array)
        avg_r = np.average(r_array)
        avg_f = np.average(f_array)
        std_p = np.std(p_array)
        std_r = np.std(r_array)
        std_f = np.std(f_array)


        print( ' average_precision:' + str(
            np.round(avg_p, 5)) + ' average_recall:' + str(
            np.round(avg_r, 5)) + ' average_F1:' + str(np.round(avg_f, 5)) + ' std_precision:' + str(
            np.round(std_p, 5)) + ' std_recall:' + str(
            np.round(std_r, 5)) + ' std_F1:' + str(np.round(std_f, 5)))

        result_out.write( ' average_precision:' + str(
            np.round(avg_p, 5)) + ' average_recall:' + str(
            np.round(avg_r, 5)) + ' average_F1:' + str(np.round(avg_f, 5)) + ' std_precision:' + str(
            np.round(std_p, 5)) + ' std_recall:' + str(
            np.round(std_r, 5)) + ' std_F1:' + str(np.round(std_f, 5)))

        result_out.close()




