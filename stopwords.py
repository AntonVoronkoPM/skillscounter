from models import MongoAPI
from datetime import datetime

#define additional stop words that are not contained in the dictionary
unigrams = []
digrams = ['project', 'management', 'manager', 'ability', 'preferred', 'required', 'excellent', 
                  'will', 'new', 'working', 'years', 'experience', 'including', 'e', 'g', 'strong', 'work', 
                  'education', 'must', 'within', 'year', 'sexual', 'gender', 'candidate', 'veteran', 
                  'equal', 'best', 'related', 'employer', 'origin', 'race', 'color', 'religion', 'sex', 
                  'employment', 'applicants', 'consideration', 'receive', 'regard', 'thanbless', 
                  'thanpless', 'thanliless', 'marital', 'thanp', 'disability', 'status', 'job', 'without', 
                  'identity', 'orientation', 'age', 'thanli', 'dirltrless', 'let', 'duties', '&', 'fulltime', 
                  'monday', 'friday']


stopwords = {'database': 'sm-web', 'collection': 'stopwords', 'documents': {'unigrams': unigrams, 'digrams': digrams, 'createdAt': datetime.now(), 'updatedAt': datetime.now()}}


stopwords_db = MongoAPI(stopwords)
post_stopwords = stopwords_db.write()