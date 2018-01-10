import requests as rq
import time
import numpy as np

interval = 1638000000
start = 1356966000000
end = 1514732400000
url = 'http://index.bithumb.com/api/coinmarketcap/localAPI.php'

target = start
arr = []

while(target + interval <= 1514732400000):
    res = rq.get(url, params={
        'api': 'graph',
        'coin': 'btc',
        'subject': 'price_usd',
        'start': target,
        'end': target + interval
    })

    target = target + interval

    for i in range(len(res.json())):
        #arr.append(res.json()[i][1])
        arr.append([time.ctime(res.json()[i][0] / 1000), res.json()[i][1]])



np.savetxt("data.csv", arr,delimiter=',', fmt="%s")
