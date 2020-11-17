import os
from flask import Flask, jsonify, request
from preprocessing import json_to_dataframe, dataset_preparation
from prediction import prediction
import json
from models import MongoAPI
from ngrams import ngram
from datetime import datetime
from worker import redis_conn
from rq.job import Job
from rq import Queue

app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])

q = Queue(connection=redis_conn)

print(os.environ['APP_SETTINGS'])

@app.route('/predict', methods=['GET'])
def classifier():

  #Config
  # vac = {'database': 'sm-web', 'collection': 'vacancies', 'filter': {'analyzed': False}, 'projection': {}}
  
  # vac_db = MongoAPI(vac)
  # new_vacancies = vac_db.read()

  # if len(new_vacancies) == 0:
  #   return {"Warning": "Nothing to analyze"}

  # new_vacancies_id = []

  # if len(new_vacancies) < 10:
  #   for i in new_vacancies:
  # 	  new_vacancies_id.append(str(i['_id']))
  # else:
  # 	for i in range(10):
  # 	  new_vacancies_id.append(str(new_vacancies[i]['_id']))

  #Config ?
  # jobstr = {'database': 'sm-web', 'collection': 'jobstrings', 'filter': {'vacancyId': {'$in': new_vacancies_id}}, 'projection': {'tag': 1, 'text': 1, 'target': 1, 'vacancyId': 1}}

  jobstr = {'database': 'sm-web', 'collection': 'jobstrings', 'filter': {'target': None}, 'projection': {'tag': 1, 'text': 1, 'target': 1, 'vacancyId': 1}}
  
  jobstr_db = MongoAPI(jobstr)
  new_jobstr = jobstr_db.read()

  # new_jobstr = new_jobstr[:600]

  # if len(new_jobstr) == 0:
  #   for i in new_vacancies:
  #     data = {'filter': {'_id': i['_id']}, 'updated_data': {'$set': {'analyzed': True}}}
  #     vac_db.update(data)
  #   return {"Status": "Analyzed status was updated"}

  df = json_to_dataframe(new_jobstr)
  dataset = dataset_preparation(df)
  targets = prediction(dataset)


  res = []

  for i in range(len(new_jobstr)):
    new_jobstr[i]['target'] = int(targets[i])
    data = {'filter': {'_id': new_jobstr[i]['_id']}, 'updated_data': {'$set': {'target': new_jobstr[i]['target']}}}
    res.append(jobstr_db.update(data))


  # res_analyze = []
  # if res.count('Nothing was updated') == 0:
  #   for i in new_vacancies:
  #     data = {'filter': {'_id': i['_id']}, 'updated_data': {'$set': {'analyzed': True}}}
  #     res_analyze.append(vac_db.update(data))
  # else:
  # 	return {"Warning": "Nothing was updated"}

  # if res_analyze.count('Nothing was updated') == 0:
  # 	return {"Status": "Targets set successfully"}
  # else:
  # 	return {"Error": "Analyzed status wasn't updated"}

  return {'Status': 'Done'}

@app.route('/analyze/<position_id>', methods=['GET'])
def analyzer(position_id):
  job = q.enqueue(ngram, position_id)
  return {'job_id': job.get_id()} 

@app.route('/result/<job_key>', methods=['GET'])
def get_results(job_key):
  job = Job.fetch(job_key, connection=redis_conn)

  if job.is_finished:
  	return str(job.result)
  else:
  	return {'Status': job.get_status()}

if __name__ == '__main__':
    app.run()