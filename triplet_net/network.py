import tensorflow as tf
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display
import time
from sklearn.model_selection import train_test_split
import random
import pandas as pd
from data import *

'''
- loss = cosine, problem with euclidean
- net = Vanilla/Denset
- layers: number of layers
- neurons: number of neurons per layer
- embedding length: 8, 16, 32
- epoch: 50, 75, 100, 200
- (densenet param: subject to results)
- embedding name
'''


class siamese:

    # Create model
    def __init__(self, loss='cos', network='net', num_layers=10, neuron=100, emb_len=32, dropout=0):
        tf.reset_default_graph()

        self.x1 = tf.placeholder(tf.float32, [None, 978], "input1")
        self.x2 = tf.placeholder(tf.float32, [None, 978], "input2")
        self.x3 = tf.placeholder(tf.float32, [None, 978], "input3")

        if network == 'net':
            self.network = self.normalnet
        if network == 'densenet':
            self.network = self.dense_network

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1, num_layers, neuron, emb_len, dropout)
            scope.reuse_variables()
            self.o2 = self.network(self.x2, num_layers, neuron, emb_len, dropout)
            scope.reuse_variables()
            self.o3 = self.network(self.x3, num_layers, neuron, emb_len, dropout)

        if loss == 'cos':
            self.loss = self.loss_with_cosine()
        if loss == 'euc':
            self.loss = self.loss_with_euclid()

    def normalnet(self, x, num_layers, neuron, emb_len, dropout):

        input = x
        for l in range(num_layers):
            name = "fc" + str(l)
            fc1 = self.fc_layer(input, neuron, name)
            ac1 = tf.nn.relu(fc1)
            if (dropout):
                d1 = tf.nn.dropout(ac1, dropout)
                input = d1
            else:
                input = ac1

        fc_last = self.fc_layer(input, emb_len, "fc_embedding")
        fc_last = tf.nn.l2_normalize(fc_last, axis=1, name = "fc_normal")
        return fc_last

    # TODO: fill up densenet
    def dense_network(self, x, num_layers, neuron, emb_len, dropout):
        fc1 = self.fc_layer(x, 400, "fc1")
        ac1 = tf.nn.relu(fc1)
        op_concat1 = tf.concat([x, ac1], axis=1)

        fc2 = self.fc_layer(op_concat1, 400, "fc2")
        ac2 = tf.nn.relu(fc2)
        op_concat2 = tf.concat([op_concat1, ac2], axis=1)

        fc3 = self.fc_layer(op_concat2, 400, "fc3")
        ac3 = tf.nn.relu(fc3)
        op_concat3 = tf.concat([op_concat2, ac3], axis=1)

        fc4 = self.fc_layer(op_concat3, 32, "fc4")
        fc4 = tf.nn.l2_normalize(fc4, axis=1,name = "fc_normal")
        return fc4

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name + 'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name + 'b', dtype=tf.float32,
                            initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def loss_with_euclid(self):
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.sqrt(tf.reduce_sum(eucd2, axis=1))
        eucd2_mean = tf.reduce_mean(eucd2, axis=0)

        eucd3 = tf.pow(tf.subtract(self.o1, self.o3), 2)
        eucd3 = tf.sqrt(tf.reduce_sum(eucd3, axis=1))
        eucd3_mean = tf.reduce_mean(eucd3, axis=0)

        return eucd2_mean, eucd3_mean, eucd2, eucd3

    def loss_with_cosine(self):
        eucd2 = tf.multiply(self.o1, self.o2)
        eucd2 = 1 - tf.pow(tf.reduce_sum(eucd2, axis=1), 2)
        eucd2_mean = tf.reduce_mean(eucd2, axis=0)

        eucd3 = tf.multiply(self.o1, self.o3)
        eucd3 = 1 - tf.pow(tf.reduce_sum(eucd3, axis=1), 2)
        eucd3_mean = tf.reduce_mean(eucd3, axis=0)

        return eucd2_mean, eucd3_mean, eucd2, eucd3


'''
- Median
- AUC (write code)
- Top-x recall
'''


def run_network(input):
    # print(input)
    # initialise variables
    model_name = input["model_name"]
    emb_name = input["emb_name"]
    s = input["siamese"]
    epoch = input["epoch"]
    X = input["X_train"]
    test = input["X_test"]
    full = input["full"]
    test_pert = input["test_pert"]
    train_pert = input["train_pert"]

    print("=== Running Network")
    saver = tf.train.Saver(max_to_keep=4)
    saving_multiple = 50
    optim_pos = tf.train.AdamOptimizer(0.005).minimize(s.loss[0])
    optim_neg = tf.train.AdamOptimizer(0.005).minimize(-s.loss[1])
    with tf.Session() as session:
        # tf.global_variables_initializer()
        tf.initialize_all_variables().run()
        print("Initialized")

        # feed_dict = {s.x1: X[0], s.x2: X[1], s.x3: X[2]}

        p_loss = []
        n_loss = []
        train_acc_l = []
        test_acc_l = []

        print("Epoch\t\t\t+ Dist\t- Dist\t\tTrain\tTest")
        for a in range(epoch):
            p_loss.append([])
            n_loss.append([])
            for b in range(len(X[0])):
                sys.stdout.write("\rEpoch %d:\t%d/%d\t" % (a, b + 1, len(X[0])))
                feed_dict = {s.x1: X[0][b:b + 1], s.x2: X[1][b:b + 1], s.x3: X[2][b:b + 1]}
                _, _, l = session.run([optim_pos, optim_neg, s.loss], feed_dict=feed_dict)
                p_loss[-1].append(l[0])
                n_loss[-1].append(l[1])
            p_loss[-1] = sum(p_loss[-1]) / len(p_loss[-1])
            n_loss[-1] = sum(n_loss[-1]) / len(n_loss[-1])

            trained = session.run([s.loss], feed_dict={s.x1: X[0], s.x2: X[1], s.x3: X[2]})
            pred = session.run([s.loss], feed_dict={s.x1: test[0], s.x2: test[1], s.x3: test[2]})

            train_acc = 100 * (np.sum(trained[0][2] <= 0.5) + np.sum(trained[0][3] > 0.5)) / len(X[0]) / 2
            test_acc = 100 * (np.sum(pred[0][2] <= 0.5) + np.sum(pred[0][3] > 0.5)) / len(test[0]) / 2

            train_acc_l.append(train_acc)
            test_acc_l.append(test_acc)

            print("%2.3f\t%2.3f\t\t%2.3f\t%2.3f" % (p_loss[-1], n_loss[-1], train_acc, test_acc))

            if a % saving_multiple == 0:
                saver.save(session, '../Models/' + model_name + "/" + model_name, global_step=a)

        saver.save(session, '../Models/' + model_name + "/" + model_name, global_step=epoch)
        print("==== Saved Model")

        # Get embeddings
        embeddings = session.run([s.o1, s.o2, s.o3], feed_dict={s.x1: X[0], s.x2: X[1], s.x3: X[2]})
        trained = session.run([s.loss], feed_dict={s.x1: X[0], s.x2: X[1], s.x3: X[2]})
        pred = session.run([s.loss], feed_dict={s.x1: test[0], s.x2: test[1], s.x3: test[2]})

        # full_embedding
        full_dataset_embeddings = session.run([s.o1], feed_dict={s.x1: full.iloc[:, 0:978]})
        train_data, _, test_data, y_test = generate_data_2(full, train_pert, test_pert)

        #TODO: change name
        pickle.dump(test_data, open('../Data/SNN_triplet_X_test', 'wb'))
        pickle.dump(y_test, open('../Data/SNN_triplet_y_test', 'wb'))

        # train_embedding
        train_embeddings = session.run([s.o1], feed_dict={s.x1: train_data})

        # test_embedding
        test_embeddings = session.run([s.o1], feed_dict={s.x1: test_data})

        # # Save Embeddings
        # embfile = "../Embeddings/" + emb_name
        # cols = ['e' + str(a) for a in range(1, len(full_dataset_embeddings[0][0]) + 1)]
        # # print(cols)
        # e = pd.DataFrame(test_embeddings[0], columns=cols)
        # e['pert_id'] = list(y_test)
        # # print(e.head())
        # e.to_csv(embfile)
        # print("==== Saved Embedding")

        # Return Dictionary
        return_dict = dict()
        return_dict["full_embedding"] = full_dataset_embeddings
        return_dict["train_embedding"] = train_embeddings
        return_dict["test_embedding"] = test_embeddings
        return_dict["train_loss"] = trained
        return_dict["test_loss"] = pred
        return_dict["p_loss"] = p_loss
        return_dict["n_loss"] = n_loss
        return_dict["train_acc"] = train_acc_l
        return_dict["test_acc"] = test_acc_l

        return return_dict


def graph(epoch, p_loss, n_loss, train_acc_l=None, test_acc_l=None):
    plt.figure(figsize=(20, 6))
    plt.plot(np.arange(epoch), p_loss, label='positive')
    plt.plot(np.arange(epoch), n_loss, label='negative')
    plt.legend()
    plt.title("Loss over epoch")
    plt.show()

    if train_acc_l != None and test_acc_l != None:
        plt.figure(figsize=(20, 6))
        plt.plot(np.arange(epoch), train_acc_l, label='train')
        plt.plot(np.arange(epoch), test_acc_l, label='test')
        plt.legend()
        plt.title("Accuracy over epoch")
