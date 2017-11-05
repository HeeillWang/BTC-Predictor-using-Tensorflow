import numpy as np

size_train = 1800	# train data size
normal_columns = 9	# number of columns that sholud be normailized
scale_factor = 1	# datas are normalized 0 ~ scale_factor

data = np.loadtxt("data.csv", delimiter=",", dtype=np.float32)
size_test = data.shape[0] - size_train
print(size_test)

#normal_data = data[0:, :normal_columns]
#other_data = data[0:, normal_columns:]


#Normalize
#normal_data *= scale_factor / np.max(np.abs(normal_data),axis=0)
#data = np.append(normal_data, other_data)

data *= scale_factor / np.max(np.abs(data),axis=0)

#Save test data
np.savetxt("test_data.csv", data[0:size_test], delimiter=",", fmt="%f")

#Save train data
np.savetxt("train_data.csv", data[size_test:], delimiter=",", fmt="%f")


