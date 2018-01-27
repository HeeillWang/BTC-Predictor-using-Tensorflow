import  sys
sys.path.append('./Crawler')
sys.path.append('./RNNs')

import crawler
#import train


crawler.collect_data("./Crawler/data.csv", ['btc', 'xrp', 'eth'])