import re
from collections import defaultdict
import unicodedata
import numpy as np
import pandas as pd
from wordcloud import STOPWORDS
from models import MongoAPI
from datetime import datetime

stopwords = set(STOPWORDS)

stopwords_req = {'database': 'sm-web', 'collection': 'stopwords', 'filter': {}, 'projection': {'unigrams': 1, 'digrams': 1, '_id': 0}}
stopwords_db = MongoAPI(stopwords_req)
all_stopwords = stopwords_db.read()[0]

stopwords_unigrams = stopwords.union(set(all_stopwords['unigrams']))
stopwords_digrams = stopwords.union(set(all_stopwords['digrams']))

def clean_text(skills):
  print('text cleaning entered')
  skills = pd.DataFrame(skills)
  #Normalize text
  skills.text = skills.text.map(lambda x: unicodedata.normalize("NFKD", x))
  print('normalization finished')
  #Select text only
  skills = skills['text']
  skills.replace('--', np.nan, inplace=True)
  skills_na = skills.dropna()
  print('emptyness reduced')
  #convert list elements to lower case
  skills_na_cleaned = [item.lower() for item in skills_na]
  print('lowercase done')
  #remove html links from list 
  skills_na_cleaned =  [re.sub(r"http\S+", "", item) for item in skills_na_cleaned]
  print('web artifacts removed')
  #remove special characters left
  skills_na_cleaned = [re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", item) for item in skills_na_cleaned]
  print('special symbols removed')
  #convert to dataframe
  skills_na_cleaned = pd.DataFrame(np.array(skills_na_cleaned).reshape(-1))
  print('dataframe reshaped')
  #Squeeze dataframe to obtain series
  data_cleaned = skills_na_cleaned.squeeze()
  return data_cleaned

#ngram function
def ngram_extractor(text, n_gram, stopwords):
  token = [token for token in text.lower().split(" ") if token != "" if token not in stopwords]
  ngrams = zip(*[token[i:] for i in range(n_gram)])
  return [" ".join(ngram) for ngram in ngrams]

# Function to generate a dataframe with n_gram and top max_row frequencies
def generate_ngrams(df, n_gram, max_row):
  print('generation entered')
  df = clean_text(df)
  print('text cleaned up')
  temp_dict = defaultdict(int)
  if n_gram == 1:
    stopwords = stopwords_unigrams
  elif n_gram == 2:
    stopwords = stopwords_digrams
  for question in df:
    print('extraction entered')
    for word in ngram_extractor(question, n_gram, stopwords):
      temp_dict[word] += 1
  print('extraction finished')
  temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(max_row)
  temp_df.columns = ["word", "wordcount"]
  return temp_df.to_dict('records')

#function for calling ngram functionality from api
def ngram(position_id):
  print('ngrams started')

  #Config
  position_title = {'database': 'sm-web', 'collection': 'positions', 'filter': {}, 'projection': {'positions': 1}}
  pt_db = MongoAPI(position_title)
  relevant_title = pt_db.read()[0]['positions'][int(position_id)]
  if relevant_title == 0:
    return {"Warning": "No such a position"}

  print('title received')

  position = relevant_title['title']
  position_vacancies = {'database': 'sm-web', 'collection': 'vacancies', 'filter': {'position': position}, 'projection': {}}
  pv_db = MongoAPI(position_vacancies)
  relevant_postions = pv_db.read()
  positions_processed = len(relevant_postions)
  if positions_processed == 0:
    return {"Warning": "No vacancies for position"}
  
  print('positions received')

  positions_id = []

  for i in relevant_postions:
    positions_id.append(str(i['_id']))

  #Config ?
  posstr = {'database': 'sm-web', 'collection': 'jobstrings', 'filter': {'vacancyId': {'$in': positions_id}, 'target': 1}, 'projection': {'text': 1, '_id': 0}}
  
  posstr_db = MongoAPI(posstr)
  new_posstr = posstr_db.read()

  if len(new_posstr) == 0:
    return {"Warning": "No data for position"}
  
  print('jobstrings received')

  #Generate unigram for data analyst
  data_1gram = generate_ngrams(new_posstr, 1, 40)
  data_2gram = generate_ngrams(new_posstr, 2, 40)
  
  print('ngrams generated')

  #Config ?
  ngrams = {'database': 'sm-web', 'collection': 'ngrams', 'documents': {'position': position, 'vacancies_number': positions_processed, 'unigrams': data_1gram, 'digrams': data_2gram, 'createdAt': datetime.now()}}
  ngrams_db = MongoAPI(ngrams)
  post_ngrams = ngrams_db.write()

  return post_ngrams