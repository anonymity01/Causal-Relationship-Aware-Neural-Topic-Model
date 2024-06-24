# This code is modified based on https://github.com/huawei-noah/trustworthyAI/tree/master/research/CausalVAE
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
import math
import os
import utils as utils
import sys
import argparse
import pickle
class MaskLayer(tf.keras.layers.Layer):
    def __init__(self, dag_dim=4, z2_dim=1, y_dim=0, dropout=0.25):
        super(MaskLayer, self).__init__()
        self.dag_dim = dag_dim
        self.elu = tf.keras.layers.ReLU()
        self.in_dim = self.dag_dim
        self.z2_dim = z2_dim
        self.netlist = []
        for i in range(self.dag_dim):
            net = tf.keras.Sequential([
                tf.keras.layers.Dense(32, input_dim=self.z2_dim),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(self.z2_dim)
            ])
            self.netlist.append(net)

    def mix(self, z):
        zy = tf.reshape(z, [-1, self.dag_dim*self.z2_dim])
        rx_list = []
        zy_list = tf.split(zy, self.dag_dim, axis=1)
        for i in range(self.dag_dim):
            temp = self.netlist[i](zy_list[i])
            rx_list.append(temp)
        h = tf.concat(rx_list, axis=1)
        return h


class Attention(tf.keras.layers.Layer):
    def __init__(self, in_features):
        super(Attention, self).__init__()
        self.M = self.add_weight(shape=(in_features, in_features), initializer='random_normal')
        self.sigmd = tf.keras.layers.Activation('sigmoid')

    def attention(self, z, e):
        a = tf.multiply(tf.transpose(tf.matmul(tf.transpose(z, perm=[0, 2, 1]), self.M), perm=[0, 2, 1]), e)
        a = self.sigmd(a)
        A = tf.nn.softmax(a, axis=1)
        e = tf.multiply(A, e)
        return e, A
class DagLayer(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, i=False, use_bias=False):
        super(DagLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A = self.add_weight(name='A',shape=(out_features, out_features), initializer='random_normal', trainable=True)
        self.I = tf.eye(out_features)
        if use_bias:
            self.bias = self.add_weight(shape=(out_features,), initializer='random_normal', trainable=True)
        else:
            self.bias = None

    def mask_z(self, x):
        x = tf.matmul(tf.transpose(self.A), tf.reshape(x, [-1, self.out_features, 1]))
        x = tf.reshape(x, [-1, self.out_features])
        return x

    def calculate_dag(self, x, v):
        if len(x.shape) > 2:
            x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.linalg.matmul(x, tf.linalg.inv(self.I - tf.transpose(self.A)))
        if self.bias is not None:
            x = x + self.bias
        if len(x.shape) > 2:
            x = tf.transpose(x, perm=[0, 2, 1])
        return x, v
