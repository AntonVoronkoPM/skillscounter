from wordcloud import STOPWORDS
import re
from collections import defaultdict
import unicodedata

def clean_text(skills):
  skills.text = skills.text.map(lambda x: unicodedata.normalize("NFKD",x))
  #Select descriptions from requirements
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


stopwords = set(STOPWORDS)
#define additional stop words that are not contained in the dictionary
more_stopwords = {'project', 'management', 'manager', 'ability', 'preferred', 'required', 'excellent', 
                  'will', 'new', 'working', 'years', 'experience', 'including', 'e', 'g', 'strong', 'work', 
                  'education', 'must', 'within', 'year', 'sexual', 'gender', 'candidate', 'veteran', 
                  'equal', 'best', 'related', 'employer', 'origin', 'race', 'color', 'religion', 'sex', 
                  'employment', 'applicants', 'consideration', 'receive', 'regard', 'thanbless', 
                  'thanpless', 'thanliless', 'marital', 'thanp', 'disability', 'status', 'job', 'without', 
                  'identity', 'orientation', 'age', 'thanli', 'dirltrless', 'let', 'duties', '&', 'fulltime', 'monday', 'friday'}
stopwords = stopwords.union(more_stopwords)


#ngram function
def ngram_extractor(text, n_gram):
    token = [token for token in text.lower().split(" ") if token != "" if token not in stopwords]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

# Function to generate a dataframe with n_gram and top max_row frequencies
def generate_ngrams(df, n_gram, max_row):
    temp_dict = defaultdict(int)
    for question in df:
        for word in ngram_extractor(question, n_gram):
            temp_dict[word] += 1
    temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(max_row)
    temp_df.columns = ["word", "wordcount"]
    return temp_df


data_cleaned = clean_text(java)

#Generate unigram for data analyst
data_1gram = generate_ngrams(data_cleaned, 1, 40)