import  sys
sys.path.append('./Crawler')
sys.path.append('./RNNs')

import crawler_upbit
import train
import numpy as np


while 1:
    print("=============Main Controller=============")
    print("0 - Exit")
    print("1 - Collect Data")
    print("2 - Train")
    print("3 - Predict")
    print("Enter by number : ")
    num = int(input())

    if num == 0:
        exit()
    elif num == 1:
        crawler_upbit.collect_data('data.csv', ['btc'])
    elif num == 2:
        train.train('./RNNs')
    elif num == 3:
        data = np.loadtxt('data.csv', delimiter=',',usecols=(1), skiprows=1)
        train.predict("./RNNs/saved/", data)

