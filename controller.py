import  sys
sys.path.append('./Crawler')
sys.path.append('./RNNs')

import crawler
import train


train.train_predict('./RNNs')