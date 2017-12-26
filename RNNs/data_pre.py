import numpy as np

size_train = 2800	# train data size
normal_columns = 1	# number of columns that sholud be normailized
scale_factor = 1	# datas are normalized [0,scale_factor]

data = np.loadtxt("hourly_data.csv", delimiter=",", dtype=np.float32, skiprows=1)
size_test = data.shape[0] - size_train
num_input = 100

#Normalize
data *= scale_factor / np.max(np.abs(data),axis=0)
new_data = []
label = []

for i in range(len(data) - num_input):
    new_data.append(data[i:i+num_input])
    
    if(data[i+num_input-1] < data[i+num_input]):
        label.append(1)
    else:
        label.append(0)

#Save test data
np.savetxt("train_data.csv", new_data[0:size_train], delimiter=",", fmt="%f")
np.savetxt("train_label.csv", label[0:size_train], delimiter=",", fmt="%d")

#Save train data
np.savetxt("test_data.csv", new_data[size_train:], delimiter=",", fmt="%f")
np.savetxt("test_label.csv", label[size_train:], delimiter=",", fmt="%d")


