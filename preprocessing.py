import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from joblib import load


def json_to_dataframe(req):

  """ Function takes json and returns pandas DataFrame"""

  first_level_normalizing = json_normalize(req['jobs'])
  first_level_columns = list(first_level_normalizing.columns)
  
  works_data = json_normalize(data=req['jobs'], record_path='parsedArray', meta=first_level_columns)
  works_data.drop(columns=['parsedArray', 'createdAt', 'updatedAt', '__v', 'keyword', 'isHeader', '_id'], inplace=True)

  years = works_data[works_data.text.str.contains('years', regex=False)]
  years = years['text'].str.extract('(\d+)').dropna().rename(columns={0: 'experience'})
  years['experience'] = years['experience'].map(int)
  years = years.drop(years[years['experience'] > 7].index)

  works_data = works_data.join(years, how='outer')
  works_data['experience'] = works_data['experience'].fillna(works_data.groupby('job_id')['experience'].transform('max')).fillna(0)
  works_data['experience'] = works_data['experience'].map(int)

  works_data['target'] = np.nan
  works_data.drop_duplicates(subset=['text', 'job_id'], inplace=True)

  print(works_data.columns)

  return works_data


def dataset_preparation(df):

  """ Function takes DataFrame with text data and returns numeric matrixes for prediction model"""

  tfidf_text = load('tfidf_models/tfidf_text.joblib')
  tfidf_tag = load('tfidf_models/tfidf_tag.joblib')

  text = tfidf_text.transform(df.text)
  tag = tfidf_tag.transform(df.tag)

  X = hstack([text, tag])

  return X