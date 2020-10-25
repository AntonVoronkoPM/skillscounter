import os
from flask import Flask, jsonify, request
from preprocessing import json_to_dataframe, dataset_preparation
from prediction import prediction
import json

app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTING'])

print(os.environ['APP_SETTINGS'])

@app.route('/predict', methods=['POST'])
def classifier():

  req = request.get_json()
  req = json.loads(req)
  print(type(req))
  df = json_to_dataframe(req)
  dataset = dataset_preparation(df)
  targets = prediction(dataset)
  for i in range(len(req['jobs'])):
    req['jobs'][i]['target'] = int(targets[i])
  print(type(req))
  return jsonify(req)

if __name__ == '__main__':
    app.run()
