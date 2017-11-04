#Reference : https://github.com/hunkim/DeepLearningZeroToAll

import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

def MinMaxScaler(tensor):
    tensor = tf.div(
       tf.subtract(
          tensor, 
          tf.reduce_min(tensor)
       ), 
       tf.subtract(
          tf.reduce_max(tensor), 
          tf.reduce_min(tensor)
       )
    )
    
    return tensor


filename_queue = tf.train.string_input_producer(
    ['train_data.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.], [0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# Basic parameters
num_input = 7
num_output = 2
batch_size = 100
learning_rate = 0.0001
training_epochs = 100
layer1 = 200
layer2 = 200
# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing

#normalize
#for i in range(num_input):
#    xy[i] = tf.to_float(xy[i])
#    xy[i] = MinMaxScaler(xy[i])


# collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=batch_size)



# Input and output placeholders. It will be fed
X = tf.placeholder(tf.float32, shape=[None, num_input])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_one_hot = tf.one_hot(Y, num_output)  # one hot
Y_one_hot = tf.reshape(Y_one_hot, [-1, num_output])


# Weight variables
#W = tf.Variable(tf.random_normal([9, 1]), name='weight', dtype=tf.float32)
#b = tf.Variable(tf.random_normal([1]), name='bias', dtype=tf.float32)

# Hypothesis
W1 = tf.get_variable("W1", shape=[num_input, layer1],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([layer1]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("W2", shape=[layer1, layer2],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([layer2]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable("W3", shape=[layer2, num_output],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([num_output]))
hypothesis = tf.matmul(L2, W3) + b3

# cost/loss function
#cost = tf.reduce_mean(tf.square(hypothesis - Y))
#cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
#                       tf.log(1 - hypothesis))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y_one_hot))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(2193 / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = sess.run([train_x_batch, train_y_batch])
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

coord.request_stop()
coord.join(threads)


print('Learning Finished!')

#
#Test
#
data = np.loadtxt("test_data.csv", delimiter=",", dtype=np.float32)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X:data[0:, :-1], Y: data[0:, -1:]}))

