#Reference : https://github.com/hunkim/DeepLearningZeroToAll


import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

num_input = 1
num_seq = 10
num_class = 2
num_hidden = 2
batch_size = 100
learning_rate = 0.001
epoch = 100

x = np.loadtxt("train_data.csv", delimiter=",", dtype=np.float32)
y = np.loadtxt("train_lable.csv", delimiter=",", dtype=np.float32)


X = tf.placeholder(tf.float32, [None, num_seq, num_input])
Y = tf.placeholder(tf.float32, [None, num_class])

cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)
output, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
