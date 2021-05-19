import time
import requests
import json


def request(data):
    url = 'http://127.0.0.1:8080/tccapi'
    start = time.time()
    try:
        res = requests.post(url=url, data=data)
        res_batch = json.loads(res.text)
        res = json.dumps(res_batch)
    except Exception as e:
        index_list = data['index']
        res_batch = {'results': []}
        for index_str in index_list:
            res_elem = {'ok': False, 'index': index_str, 'predict': 0,
                        'msg': 'get result from tccapi failed, set default predict to 0'}
            res_batch['results'].append(res_elem)
        res = json.dumps(res_batch)
    end = time.time()
    cost = end - start
    return res, cost


if __name__ == '__main__':
    res = total = 0
    for _ in range(1000):
        data_json = {'input': ['444 29 19\t421 39 9'], 'index': ['1']}
        res, cost = request(data_json)
        total += cost
    print(res)
    print(total)
