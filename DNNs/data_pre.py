import numpy as np

size_train = 1800
scale_factor = 1

data = np.loadtxt("data.csv", delimiter=",", dtype=np.float32)

#Normalize
data *= scale_factor / np.max(np.abs(data),axis=0)

#Save train data
np.savetxt("train_data.csv", data[0:size_train], delimiter=",", fmt="%f")

#Save test data
np.savetxt("test_data.csv", data[size_train:], delimiter=",", fmt="%f")

