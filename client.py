import json

import requests
from bson import json_util
from models import MongoAPI
from prediction import prediction
from preprocessing import dataset_preparation, json_to_dataframe

# f = open('30k-hr-linkedin.json', 'rb')
# # print('here')
# hr = json.load(f)


# # res = requests.post('http://localhost:5000/predict', json=json.dumps(hr))
# res = requests.post('https://future-skills.herokuapp.com/', json=json.dumps(hr))
# if res.ok:
#     print (res.json())

# f.close

# data = {
#     "database": "sm-web",
#     "collection": "vacancies",
# }
# mongo_obj = MongoAPI(data)
# print(json.dumps(mongo_obj.read(), indent=4, default=json_util.default))


jobstr = {
    "database": "sm-web",
    "collection": "jobstrings",
    "filter": {"target": None},
    "projection": {"tag": 1, "text": 1},
}

jobstr_db = MongoAPI(jobstr)
new_jobstr = jobstr_db.read()

df = json_to_dataframe(new_jobstr)
dataset = dataset_preparation(df)
targets = prediction(dataset)


res = []

for i in range(len(new_jobstr)):
    new_jobstr[i]["target"] = int(targets[i])
    data = {
        "filter": {"_id": new_jobstr[i]["_id"]},
        "updated_data": {"$set": {"target": new_jobstr[i]["target"]}},
    }
    res.append(jobstr_db.update(data))
