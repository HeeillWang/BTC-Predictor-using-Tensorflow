#Reference : https://github.com/hunkim/DeepLearningZeroToAll

import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

'''
parameter 

- data : numpy array

--------------------
returns 

- Normalized numpy array

'''
def MinMaxScaler(data):
    # to avoid divide by zero, add noise
    data *= 1 / (np.max(np.abs(data),axis=0) + 1e-8)
    return data

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
        temp = []
        if(data[i+num_seq-1][pos] > data[i+num_seq][pos]):
            y.append([1])   # up
        else:
            y.append([0])   # down

    return (x,y)

num_input = 5
num_seq = 100
num_output = 2  # number of output class by one_hot
num_hidden = 2
learning_rate = 0.01
epoch = 10000
data_split_rate = 0.7	# dataset split rate for train data. Others will be test data


x = np.loadtxt('hourly_data.csv', delimiter=',', skiprows=1)
x = MinMaxScaler(x)


x, y = MakeDataSet(x, num_seq, num_input-1)	# shape = [None, num_seq, num_input]

train_len = int(len(x) * data_split_rate)
test_len = len(x) - train_len

train_x = x[:train_len]
train_y = y[:train_len]
test_x = x[train_len:]
test_y = y[train_len:]


X = tf.placeholder(tf.float32, [None, num_seq, num_input])
Y= tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, num_output)  # one hot
Y_one_hot = tf.reshape(Y_one_hot, [-1, num_output])

#RNN Model + fully-connected
cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True, activation=tf.tanh)
outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
W = tf.Variable(tf.random_normal([num_hidden, num_output]))
b = tf.Variable(tf.random_normal([num_output]))
predict = tf.matmul(outputs, W) + b

#predict = tf.contrib.layers.fully_connected(outputs, num_output, activation_fn=None)  # use last cell's output
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y_one_hot))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)



with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training
    for i in range(epoch):
        _, loss = sess.run([optimizer, cost], feed_dict = {X : train_x, Y:train_y})
        if((i+1) % 100 == 0):
          print("Epoch ", i+1, " : ", loss)

    # Testing
    correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))
    print('outputs : ', sess.run(outputs[:30], feed_dict={X:test_x}))
    print('predict : ', sess.run(predict[:30], feed_dict={X: test_x}))
    print('label : ', test_y[:30])



