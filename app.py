import os
from flask import Flask
from flask_cors import CORS
from prediction import classifier
from ngrams import ngram
from datetime import datetime
from worker import redis_conn
from rq.job import Job
from rq import Queue

app = Flask(__name__)
cors = CORS(app, resources=r'/*')
app.config.from_object(os.environ['APP_SETTINGS'])

q = Queue(connection=redis_conn)

print(os.environ['APP_SETTINGS'])

@app.route('/predict', methods=['GET'])
def prediction():
  job = q.enqueue(classifier)
  return {'job_id': job.get_id()}

@app.route('/analyze/<position_id>', methods=['GET'])
def analyzer(position_id):
  print('analyze')
  job = q.enqueue(ngram, position_id)
  return {'job_id': job.get_id()} 

@app.route('/result/<job_key>', methods=['GET'])
def get_results(job_key):
  job = Job.fetch(job_key, connection=redis_conn)

  if job.is_finished:
  	return str(job.result)
  else:
  	return {'Status': job.get_status()}

if __name__ == '__main__':
    app.run()