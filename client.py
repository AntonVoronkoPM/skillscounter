import requests
import json
from models import MongoAPI
from bson import json_util

# f = open('30k-hr-linkedin.json', 'rb')
# # print('here')
# hr = json.load(f)


# # res = requests.post('http://localhost:5000/predict', json=json.dumps(hr))
# res = requests.post('https://future-skills.herokuapp.com/', json=json.dumps(hr))
# if res.ok:
#     print (res.json())

# f.close

data = {
    "database": "sm-web",
    "collection": "vacancies",
}
mongo_obj = MongoAPI(data)
print(json.dumps(mongo_obj.read(), indent=4, default=json_util.default))