from datetime import datetime

import requests as rq
import time
import datetime
import numpy as np
from time import gmtime, strftime
from pip.compat import total_seconds
import json

'''
Collect cryptocurrency data

parameter
- path : .csv file path that will be saved
- coins : coin list that want to collect data

returns
- none
'''
def collect_data(path, coins):
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    flag = False

    url = 'https://crix-api-endpoint.upbit.com/v1/crix/candles/minutes/60'
    data = np.loadtxt(path, delimiter=',', dtype='str')
    last_time = data[-1][0]
    last_time = datetime.datetime.strptime(last_time, "%a %b %d %H:%M:%S %Y")
    last_time = last_time.timestamp() * 1000    # in millisecond

    collect_list = []
    target = cur_time
    while 1:
        while 1:
            try:
                res = rq.get(url, params={
                    'code': 'CRIX.UPBIT.USDT-BTC',
                    'count': '100',
                    'to': target,
                })
            except:
                time.sleep(1)
                print('except')
                continue
            else:
                if res.status_code == 200 and len(res.json()) != 0:
                    print(res)
                    break
                else:
                    time.sleep(1)
                    print('Response code is not 200')
                    continue

        for obj in res.json():
            if obj['timestamp'] <= last_time:
                print('collecting completed')
                flag = True
                break
            else:
                collect_list.append([time.ctime(obj['timestamp'] / 1000), obj['tradePrice']])

        if flag:
            collect_list = collect_list[::-1]   # reverse
            data = np.append(data, collect_list, axis=0)
            np.savetxt('../data2.csv', data, delimiter=',', fmt="%s")
            break
        else:
            # update target time
            target = res.json()[-1]['timestamp']
            target = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(target / 1000))





collect_data('../data.csv','df')