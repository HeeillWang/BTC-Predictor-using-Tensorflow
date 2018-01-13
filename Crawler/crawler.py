import requests as rq
import time
import numpy as np

interval = 1638000000
start = 1356966000000
end = 1514732400000
url = 'http://index.bithumb.com/api/coinmarketcap/localAPI.php'

target = start
coins = [
    'btc',
    'xrp',
    'eth',
]

arr = []
title = []  # title of columns
title.append("time")

for coin in coins:
    title.append(coin)

arr.append(title)


while(target + interval <= 1514732400000):
    json_list = []
    json_len = 0

    for coin in coins:
        res = rq.get(url, params={
            'api': 'graph',
            'coin': coin,
            'subject': 'price_usd',
            'start': target,
            'end': target + interval
        })

        if(coin == 'btc'):
            json_len = len(res.json())

        temp = []
        if len(res.json()) < json_len:  # add default values
            for i in range(json_len - len(res.json())):
                temp.append([0, 0])

            temp = temp + res.json()
            json_list.append(temp)
        else:
            json_list.append(res.json())

    target = target + interval

    for i in range(json_len):
        row = []
        row.append(time.ctime(json_list[0][i][0] / 1000)) # add timestamp

        for j in range(len(json_list)):
            row.append(json_list[j][i][1])

        arr.append(row)


np.savetxt("data.csv", arr,delimiter=',', fmt="%s")
