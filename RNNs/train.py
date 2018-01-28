#Reference : https://github.com/hunkim/DeepLearningZeroToAll

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
tf.set_random_seed(777)  # for reproducibility


'''
parameter 

- data : numpy array

--------------------
returns 

- Normalized numpy array

'''
def MinMaxScaler(data, label_pos):
    # to avoid divide by zero, add noise
    data = (data - np.min(np.abs(data), axis=0)) / (np.max(np.abs(data),axis=0) - np.min(np.abs(data), axis=0) + 1e-8)
    return (data,np.min(data[:,label_pos]), np.max(data[:, label_pos]))


'''
decode scaled(0~1) predict value to original value

parameter

- predict : decoding target
- min : minimum value for data label
- max : maximum value for data label

------------------
returns 

- ret : decoded value
'''
def decodePredict(predict, min, max):
    ret = predict * (max - min + 1e-8)
    ret = ret + min

    return ret


'''
parameter

- data : numpy array
- num_seq : number of sequence for input data
- pos : index of label in data

------------------
returns 

- x : input dataset 
- y : label dataset
'''
def MakeDataSet(data, num_seq, pos):
    x = []
    y = []

    for i in range(len(data) - num_seq):
        x.append(data[i:i+num_seq])
        y.append([data[i+num_seq][pos]])

    return (x,y)

class Model():
    num_input = 3
    num_seq = 100
    num_output = 1
    num_hidden = 2
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, [None, num_seq, num_input])
    Y = tf.placeholder(tf.float32, [None, num_output])

    # RNN Model + fully-connected
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True, activation=tf.tanh)
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    predict = tf.contrib.layers.fully_connected(outputs[:, -1], num_output,
                                                activation_fn=None)  # use last cell's output
    cost = tf.reduce_sum(tf.square(predict - Y))  # MSE
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # RMSE for accuracy test
    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

def load_model(path, sess):
    file_name = "saved_model_epoch_"  # prefix of file name that will be loaded

    print("Do you want to restore your model? (Y/N)")
    ans = input()

    old_epoch = 0  # epoch of restored model.

    saver = tf.train.Saver()

    if (ans == 'y') or (ans == 'Y'):
        num = 0
        model_list = []

        # list all models saved.
        for f in listdir(path):
            if f.find(".ckpt.meta") != -1:
                model_list.append(f.replace(".meta", ''))
                print(str(num + 1) + " - " + model_list[num])
                num += 1

        if num == 0:
            print("No models found")
        else:
            print("Select model by number : ")
            num = int(input())
            saver.restore(sess, path + model_list[num - 1])
            print(model_list[num - 1], "is restored")
            old_epoch = int((model_list[num - 1].replace(file_name, "")).replace(".ckpt", ""))
            print("Restored epoch : ", old_epoch)

    return old_epoch, sess


def train(path):
    data_split_rate = 0.7  # dataset split rate for train data. Others will be test data
    label_pos = 0

    model = Model()

    file_name = "saved_model_epoch_"  # prefix of file name that will be saved
    path = path + "/saved/"  # path of files

    x = np.loadtxt('./Crawler/data.csv', delimiter=',', usecols=(1, 2, 3), skiprows=1)

    if model.num_input == 1:
        x = np.reshape(x, (-1, 1))

    x, label_min, label_max = MinMaxScaler(x, label_pos)

    x, y = MakeDataSet(x, model.num_seq, label_pos)  # shape = [None, num_seq, num_input]

    train_len = int(len(x) * data_split_rate)

    train_x = x[:train_len]
    train_y = y[:train_len]
    test_x = x[train_len:]
    test_y = y[train_len:]


    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        old_epoch, sess = load_model(path, sess)


        print("Input training epoch : ")
        epoch = int(input())

        # Training
        for i in range(epoch):
            _, loss = sess.run([model.optimizer, model.cost], feed_dict={model.X: train_x, model.Y: train_y})
            if ((i + 1) % 50 == 0):
                print("Epoch ", i + 1, " : ", loss)

        # Testing
        test_predict = sess.run(model.predict, feed_dict={model.X: test_x})
        rmse_val = sess.run(model.rmse, feed_dict={model.targets: test_y, model.predictions: test_predict})
        print("RMSE for test data : ", rmse_val)

        # Save the variables to disk.
        save_path = saver.save(sess, path + file_name + str(epoch + old_epoch) + ".ckpt")
        print("Models saved in : %s" % save_path)

        # Show test accuracy by matplotlib
        plt.plot(test_y)
        plt.plot(test_predict)
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.show()



