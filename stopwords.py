from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)

#define additional stop words that are not contained in the dictionary
unigrams = {}
digrams = {'project', 'management', 'manager', 'ability', 'preferred', 'required', 'excellent', 
                  'will', 'new', 'working', 'years', 'experience', 'including', 'e', 'g', 'strong', 'work', 
                  'education', 'must', 'within', 'year', 'sexual', 'gender', 'candidate', 'veteran', 
                  'equal', 'best', 'related', 'employer', 'origin', 'race', 'color', 'religion', 'sex', 
                  'employment', 'applicants', 'consideration', 'receive', 'regard', 'thanbless', 
                  'thanpless', 'thanliless', 'marital', 'thanp', 'disability', 'status', 'job', 'without', 
                  'identity', 'orientation', 'age', 'thanli', 'dirltrless', 'let', 'duties', '&', 'fulltime', 
                  'monday', 'friday'}

stopwords_unigrams = stopwords.union(unigrams)
stopwords_digrams = stopwords.union(digrams)