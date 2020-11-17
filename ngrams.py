import re
from collections import defaultdict
import unicodedata
import numpy as np
import pandas as pd
from wordcloud import STOPWORDS
from models import MongoAPI

stopwords = set(STOPWORDS)

stopwords_req = {'database': 'sm-web', 'collection': 'stopwords', 'filter': {}, 'projection': {'unigrams': 1, 'digrams': 1, '_id': 0}}
stopwords_db = MongoAPI(stopwords_req)
all_stopwords = stopwords_db.read()[0]

stopwords_unigrams = stopwords.union(set(all_stopwords['unigrams']))
stopwords_digrams = stopwords.union(set(all_stopwords['digrams']))

def clean_text(skills):
  skills = pd.DataFrame(skills)
  #Normalize text
  skills.text = skills.text.map(lambda x: unicodedata.normalize("NFKD", x))
  #Select text only
  skills = skills['text']
  skills.replace('--', np.nan, inplace=True)
  skills_na = skills.dropna()
  #convert list elements to lower case
  skills_na_cleaned = [item.lower() for item in skills_na]
  #remove html links from list 
  skills_na_cleaned =  [re.sub(r"http\S+", "", item) for item in skills_na_cleaned]
  #remove special characters left
  skills_na_cleaned = [re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", item) for item in skills_na_cleaned]
  #convert to dataframe
  skills_na_cleaned = pd.DataFrame(np.array(skills_na_cleaned).reshape(-1))
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
  df = clean_text(df)
  temp_dict = defaultdict(int)
  if n_gram == 1:
    stopwords = stopwords_unigrams
  elif n_gram == 2:
    stopwords = stopwords_digrams
  for question in df:
    for word in ngram_extractor(question, n_gram, stopwords):
      temp_dict[word] += 1
  temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(max_row)
  temp_df.columns = ["word", "wordcount"]
  return temp_df.to_dict('records')