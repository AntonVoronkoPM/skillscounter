import re
import string

# from collections import Counter
import unicodedata
from datetime import datetime

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.probability import FreqDist
from nltk.stem import SnowballStemmer
from nltk.util import ngrams

from skillscounter.models import MongoAPI

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

nltk.data.path.append("./nltk_data")

# Retrieving stopwords from database
stopwords_req = {
    "database": "sm-web",
    "collection": "stopwords",
    "filter": {},
    "projection": {"unigrams": 1, "digrams": 1, "_id": 0},
}
stopwords_db = MongoAPI(stopwords_req)
all_stopwords = stopwords_db.read()[0]


def clean_text(skills):
    print("text cleaning entered")
    skills = pd.DataFrame(skills)
    # Normalize text
    skills.text = skills.text.map(lambda x: unicodedata.normalize("NFKD", x))
    print("normalization finished")
    # Select text only
    skills = skills["text"]
    skills.replace("--", np.nan, inplace=True)
    skills_na = skills.dropna()
    print("emptyness reduced")
    # convert list elements to lower case
    skills_na_cleaned = [item.lower() for item in skills_na]
    print("lowercase done")
    # remove html links from list
    skills_na_cleaned = [re.sub(r"http\S+", "", item) for item in skills_na_cleaned]
    print("web artifacts removed")
    # remove special characters left
    # skills_na_cleaned = [re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", item) for item in skills_na_cleaned]
    skills_na_cleaned = [re.sub(r"[-()\"@;:<>{}`=~|!?,]", "", item) for item in skills_na_cleaned]
    print("special symbols removed")
    return skills_na_cleaned


def token_extractor(text, n_gram, isTechnical):
    # print(text)
    nltk_stopwords = nltk.corpus.stopwords.words("english")
    nltk_stopwords = set(nltk_stopwords)
    if n_gram == 1:
        if isTechnical:
            nltk_stopwords.update(all_stopwords["unigramsTech"])
        else:
            nltk_stopwords.update(all_stopwords["unigrams"])
    elif n_gram == 2:
        if isTechnical:
            nltk_stopwords.update(all_stopwords["digramsTech"])
        else:
            nltk_stopwords.update(all_stopwords["digrams"])
    stemmer = SnowballStemmer("english")
    corpus_nltk = [pos_tag(word_tokenize(pos)) for pos in text]
    corpus_clean_nltk = [[], []]
    for j in corpus_nltk:
        for i in j:
            if i[0] not in nltk_stopwords and i[0] not in string.punctuation:
                stem = stemmer.stem(i[0])
                corpus_clean_nltk[0].append(i)
                corpus_clean_nltk[1].append(stem)
    return corpus_clean_nltk


# ngram function
def digram_extractor(tokens):
    digrams_i = ngrams(tokens[0], 2)
    digrams_stem = ngrams(tokens[1], 2)
    di_i = list(digrams_i)
    di_stem = list(digrams_stem)
    di_i_adj = []
    di_stem_adj = []
    for i in range(len(di_i)):
        # print('check digrams')
        if (
            (di_i[i][0][1] == di_i[i][1][1] and (di_i[i][0][1] == "JJ" or di_i[i][0][1] == "VB"))
            or (di_i[i][0][1]) == "POS"
            or (di_i[i][1][1]) == "POS"
        ):
            continue
        di_i_adj.append(di_i[i])
        di_stem_adj.append(di_stem[i])
    return [di_i_adj, di_stem_adj]


# Function to generate a dataframe with n_gram and top max_row frequencies
def generate_ngrams(df, n_gram, max_row, isTechnical):
    print("ngram value")
    # print(df[0:5])
    print("generation entered")
    df = clean_text(df)
    print("text cleaned up")
    # print(df[0:5])
    tokens = token_extractor(df, n_gram, isTechnical)
    if n_gram == 2:
        print("digrams")
        tokens = digram_extractor(tokens)
    freq = FreqDist(tokens[1]).most_common(max_row)
    if n_gram == 2:
        for i in range(len(tokens[0])):
            tokens[0][i] = [(tokens[0][i][0][0] + " " + tokens[0][i][1][0])]
    top_freq = []
    for i in range(max_row):
        top_freq.append(
            {"word": tokens[0][tokens[1].index(freq[i][0])][0], "wordcount": freq[i][1]}
        )
    return top_freq


# function for calling ngram functionality from api
def ngram(position_id):
    print("starting ngram functionality")
    print("position id", position_id)

    # Config
    position_title = {
        "database": "sm-web",
        "collection": "positions",
        "filter": {},
        "projection": {"positions": 1},
    }
    pt_db = MongoAPI(position_title)
    relevant_title = pt_db.read()[0]["positions"][int(position_id)]
    if len(relevant_title) == 0:
        return {"Warning": "No such a position"}

    isTechnical = relevant_title["isTechnical"]
    position = relevant_title["title"]
    position_vacancies = {
        "database": "sm-web",
        "collection": "vacancies",
        "filter": {"position": position},
        "projection": {},
    }
    pv_db = MongoAPI(position_vacancies)
    relevant_postions = pv_db.read()
    positions_processed = len(relevant_postions)
    if positions_processed == 0:
        return {"Warning": "No vacancies for position"}

    vacancies_id = []

    for i in relevant_postions:
        vacancies_id.append(str(i["_id"]))

    # Config ?
    posstr = {
        "database": "sm-web",
        "collection": "jobstrings",
        "filter": {"vacancyId": {"$in": vacancies_id}, "target": 1},
        "projection": {"text": 1, "_id": 0},
    }

    posstr_db = MongoAPI(posstr)
    new_posstr = posstr_db.read()

    if len(new_posstr) == 0:
        return {"Warning": "No data for position"}

    print("jobstrings received")

    # Generate ngram
    data_1gram = generate_ngrams(new_posstr, 1, 40, isTechnical)
    data_2gram = generate_ngrams(new_posstr, 2, 40, isTechnical)

    print("ngrams generated")

    # Config ?
    # ngrams = {'database': 'sm-web', 'collection': 'ngrams', 'documents': {'position': position, 'positionId': position_id, 'vacancies_number': positions_processed, 'unigrams': data_1gram, 'digrams': data_2gram, 'createdAt': datetime.now()}}
    ngrams_conn = {"database": "sm-web", "collection": "ngrams"}
    ngrams_db = MongoAPI(ngrams_conn)
    ngrams_data = {
        "filter": {"position": position},
        "updated_data": {
            "$set": {
                "vacancies_number": positions_processed,
                "unigrams": data_1gram,
                "digrams": data_2gram,
                "updatedAt": datetime.now(),
            },
            "$setOnInsert": {
                "position": position,
                "positionId": position_id,
                "createdAt": datetime.now(),
            },
        },
        "upsert": True,
    }
    post_ngrams = ngrams_db.update(ngrams_data, upsert=True)

    return post_ngrams
