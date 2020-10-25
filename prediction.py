from joblib import load

def prediction(X):

  """ Function takes dataset and predicts which lines are content important information"""

  clf = load('classifier_model/position_text_classifier.joblib')

  y = clf.predict(X)

  return y