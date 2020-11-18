from joblib import load
from preprocessing import json_to_dataframe, dataset_preparation
from models import MongoAPI

def prediction(X):

  """ Function takes dataset and predicts which lines are content important information"""

  clf = load('classifier_model/position_text_classifier.joblib')

  y = clf.predict(X)

  return y

def classifier():
  # Config
  vac = {'database': 'sm-web', 'collection': 'vacancies', 'filter': {'analyzed': False}, 'projection': {}}
  
  vac_db = MongoAPI(vac)
  new_vacancies = vac_db.read()

  if len(new_vacancies) == 0:
    return {"Warning": "Nothing to analyze"}

  new_vacancies_id = []

  for i in new_vacancies:
  	new_vacancies_id.append(str(i['_id']))


  #Config ?
  jobstr = {'database': 'sm-web', 'collection': 'jobstrings', 'filter': {'vacancyId': {'$in': new_vacancies_id}}, 'projection': {'tag': 1, 'text': 1}}

  
  jobstr_db = MongoAPI(jobstr)
  new_jobstr = jobstr_db.read()

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


  res_analyze = []
  if res.count('Nothing was updated') == 0:
    for i in new_vacancies:
      data = {'filter': {'_id': i['_id']}, 'updated_data': {'$set': {'analyzed': True}}}
      res_analyze.append(vac_db.update(data))
  else:
  	return {"Warning": "Nothing was updated"}

  if res_analyze.count('Nothing was updated') == 0:
  	return {"Status": "Targets set successfully"}
  else:
  	return {"Error": "Analyzed status wasn't updated"}