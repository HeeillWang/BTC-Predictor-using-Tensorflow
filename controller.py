import  sys
sys.path.append('./Crawler')
sys.path.append('./RNNs')

import crawler
import train


while 1:
    print("0 - Exit")
    print("1 - Collect Data")
    print("2 - Train")
    print("3 - Predict")
    print("Enter by number : ")
    num = int(input())

    if num == 0:
        exit()
    elif num == 1:
        crawler.collect_data('data.csv', ['btc'])
    elif num == 2:
        train.train('./RNNs')

