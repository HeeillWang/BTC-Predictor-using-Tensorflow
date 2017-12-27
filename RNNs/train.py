#Reference : https://github.com/hunkim/DeepLearningZeroToAll

import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

'''
parameter 

data : numpy array

--------------------
returns 

Normalized numpy array

'''
def MinMaxScaler(data):
    # to avoid divide by zero, add noise
    data *= 1 / (np.max(np.abs(data),axis=0) + 1e-8)
    return data

def MakeDataSet(data, num_seq):
    ret = []
    
    for i in range(len(data) - num_seq):
        ret.append(data[i:i+num_seq])

    return ret

num_input = 5
num_seq = 10
num_output = 1
num_hidden = 2
batch_size = 100
learning_rate = 0.01
epoch = 100

x = np.loadtxt('data.csv', delimiter=',', skiprows=1)
x = MinMaxScaler(data)

y = xy[:,[-1]]	# close price will be label


