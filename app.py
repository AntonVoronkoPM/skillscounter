import os
from flask import Flask, jsonify, request
from preprocessing import json_to_dataframe, dataset_preparation
from prediction import prediction
import json
from models import MongoAPI
from pprint import pprint

app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])

print(os.environ['APP_SETTINGS'])

@app.route('/predict', methods=['GET'])
def classifier():

  #Config
  vac = {'database': 'sm-web', 'collection': 'vacancies', 'filter': {'analized': False}, 'projection': {'analized': 1}}
  
  vac_db = MongoAPI(vac)
  new_vacancies = vac_db.read()

  if len(new_vacancies) == 0:
    return Response(response=json.dumps({"Warning": "Nothing to analyze"}), 
                    status=200,
                    mimetype='application/json')

  new_vacancies_id = []

  # if len(new_vacancies) < 50:
  for i in new_vacancies:
  	new_vacancies_id.append(str(i['_id']))
  # else:
  # 	for i in range(50):
  # 	  new_vacancies_id.append(str(new_vacancies[i]['_id']))

  #Config ?
  jobstr = {'database': 'sm-web', 'collection': 'jobstrings', 'filter': {'vacancyId': {'$in': new_vacancies_id}}, 'projection': {'tag': 1, 'text': 1, 'target': 1, 'vacancyId': 1}}
  
  jobstr_db = MongoAPI(jobstr)
  new_jobstr = jobstr_db.read()

  df = json_to_dataframe(new_jobstr)
  dataset = dataset_preparation(df)
  targets = prediction(dataset)


  res = []

  for i in range(len(new_jobstr)):
    new_jobstr[i]['target'] = int(targets[i])
    data = {'filter': {'_id': new_jobstr[i]['_id']}, 'updated_data': {'$set': {'target': new_jobstr[i]['target']}}}
    res.append(jobstr_db.update(data))


  res_analyze = []
  if res.count('Nothing was updated') == 0:
    for i in new_vacancies:
      data = {'filter': {'_id': i['_id']}, 'updated_data': {'$set': {'analized': True}}}
      res_analyze.append(vac_db.update(data))
  else:
  	return Response(response=json.dumps({"Warning": "Nothing was updated"}),
                    status=200,
                    mimetype='application/json')

  if res_analyze.count('Nothing was updated') == 0:
  	return Response(response=json.dumps({"Status": "Targets set successfully"}),
                    status=200,
                    mimetype='application/json')
  else:
  	return Response(response=json.dumps({"Error": "Analyzed status wasn't updated"}),
                    status=400,
                    mimetype='application/json')

@app.route('/analyze/<position>', methods=['GET'])
def analyzer():
  pass

if __name__ == '__main__':
    app.run()