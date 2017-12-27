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

'''
parameter

data : numpy array

------------------
returns 

created dataset numpy array.
'''
def MakeDataSet(data, num_seq):
    ret = []
    
    for i in range(len(data) - num_seq):
        ret.append(data[i:i+num_seq])

    return ret

num_input = 5
num_seq = 10
num_output = 1
num_hidden = 2
learning_rate = 0.01
epoch = 100
data_split_rate = 0.7	# dataset split rate for train data. Others will be test data


x = np.loadtxt('hourly_data.csv', delimiter=',', skiprows=1)
x = MinMaxScaler(x)

y = x[:,[-1]]	# close price will be label


x = MakeDataSet(x, num_seq)	# shape = [None, num_seq, num_input]
y = MakeDataSet(y, num_seq)	# shape = [None, num_seq, num_output]


