import requests as rq
import time

url = 'http://index.bithumb.com/api/coinmarketcap/localAPI.php'
res = rq.get(url, params={
    'api': 'graph',
    'coin': 'btc',
    'subject': 'price_usd',
    'start': 1451617200000,
    'end': 1452826800000
})

for i in range(len(res.json())):
    print(time.ctime(res.json()[i][0] / 1000), res.json()[i][1])