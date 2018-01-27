#Reference : https://github.com/hunkim/DeepLearningZeroToAll

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
tf.set_random_seed(777)  # for reproducibility

def hello():
    print("!!!!")

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





num_input = 3
label_pos = 0
num_seq = 100
num_output = 1
num_hidden = 2
learning_rate = 0.01
data_split_rate = 0.7	# dataset split rate for train data. Others will be test data

file_name = "saved_model_epoch_"    # prefix of file name that will be saved
path = "./saved/"   # path of files


x = np.loadtxt('../Crawler/data.csv', delimiter=',', usecols=(1,2,3), skiprows=1)

if(num_input == 1):
    x = np.reshape(x, (-1,1))


x = MinMaxScaler(x, label_pos)

x, y = MakeDataSet(x, num_seq, label_pos)	# shape = [None, num_seq, num_input]

train_len = int(len(x) * data_split_rate)
test_len = len(x) - train_len

train_x = x[:train_len]
train_y = y[:train_len]
test_x = x[train_len:]
test_y = y[train_len:]

X = tf.placeholder(tf.float32, [None, num_seq, num_input])
Y= tf.placeholder(tf.float32, [None, num_output])

#RNN Model + fully-connected
cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True, activation=tf.tanh)
outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
predict = tf.contrib.layers.fully_connected(outputs[:,-1], num_output, activation_fn=None)  # use last cell's output
cost = tf.reduce_sum(tf.square(predict - Y))    # MSE
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# RMSE for accuracy test
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    print("Do you want to restore your model? (Y/N)")
    ans = input()

    old_epoch = 0   # epoch of restored model.

    if (ans == 'y') or (ans== 'Y'):
        num = 0
        model_list = []

        # list all models saved.
        for f in listdir(path):
            if f.find(".ckpt.meta") != -1:
                model_list.append(f.replace(".meta", ''))
                print(str(num+1) + " - " + model_list[num])
                num += 1

    if num == 0:
        print("No models found")
    else:
        print("Select model by number : ")
        num = int(input())
        saver.restore(sess, path + model_list[num-1])
        print(model_list[num-1], "is restored")
        old_epoch = int((model_list[num-1].replace(file_name, "")).replace(".ckpt",""))
        print("Restored epoch : ", old_epoch)


    print("Input training epoch : ")
    epoch = int(input())

    # Training
    for i in range(epoch):
        _, loss = sess.run([optimizer, cost], feed_dict = {X : train_x, Y:train_y})
        if((i+1) % 50 == 0):
            print("Epoch ", i+1, " : ", loss)



    # Testing
    test_predict = sess.run(predict, feed_dict={X:test_x})
    rmse_val = sess.run(rmse, feed_dict={targets:test_y, predictions:test_predict})
    print("RMSE for test data : ", rmse_val)

    # Save the variables to disk.
    save_path = saver.save(sess, path + file_name + str(epoch + old_epoch) +  ".ckpt")
    print("Models saved in : %s" % save_path)

    # Show test accuracy by matplotlib
    plt.plot(test_y)
    plt.plot(test_predict)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.show()



