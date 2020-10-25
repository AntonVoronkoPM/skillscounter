import requests
import json

f = open('30k-hr-linkedin.json', 'rb')
# print('here')
hr = json.load(f)


# res = requests.post('http://localhost:5000/predict', json=json.dumps(hr))
res = requests.post('https://future-skills.herokuapp.com/', json=json.dumps(hr))
if res.ok:
    print (res.json())

f.close