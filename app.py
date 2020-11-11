import os
from flask import Flask, jsonify, request
from preprocessing import json_to_dataframe, dataset_preparation
from prediction import prediction
import json
from models import MongoAPI
from ngrams import generate_ngrams

app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])

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

  new_jobstr = new_jobstr[:600]

  if len(new_jobstr) == 0:
    for i in new_vacancies:
      data = {'filter': {'_id': i['_id']}, 'updated_data': {'$set': {'analyzed': True}}}
      vac_db.update(data)
    return {"Status": "Analyzed status was updated"}

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

@app.route('/analyze/<position>', methods=['GET'])
def analyzer(position):
  two_words = position.split('_')
  for i in range(len(two_words)):
  	two_words[i] = two_words[i].capitalize()
  position = ' '.join(two_words)

  #Config
  position_vacancies = {'database': 'sm-web', 'collection': 'vacancies', 'filter': {'position': position}, 'projection': {}}
  pv_db = MongoAPI(position_vacancies)
  relevant_postions = pv_db.read()
  positions_processed = len(relevant_postions)
  if positions_processed == 0:
  	return {"Warning": "No vacancies for position"}
  
  positions_id = []

  for i in relevant_postions:
  	positions_id.append(str(i['_id']))

  #Config ?
  posstr = {'database': 'sm-web', 'collection': 'jobstrings', 'filter': {'vacancyId': {'$in': positions_id}, 'target': 1}, 'projection': {'text': 1, '_id': 0}}
  
  posstr_db = MongoAPI(posstr)
  new_posstr = posstr_db.read()

  if len(new_posstr) == 0:
  	return {"Warning": "No data for position"}

  #Generate unigram for data analyst
  data_1gram = generate_ngrams(new_posstr, 1, 40)
  data_2gram = generate_ngrams(new_posstr, 2, 40)
  
  #Config ?
  ngrams = {'database': 'sm-web', 'collection': 'ngrams', 'documents': {'position': position, 'vacancies_number': positions_processed, 'unigrams': data_1gram, 'digrams': data_2gram}}
  ngrams_db = MongoAPI(ngrams)
  post_ngrams = ngrams_db.write()

  return post_ngrams

if __name__ == '__main__':
    app.run()