import os
import pandas as pd
import json
import pickle
#from azureml.core import Model
from sklearn.externals import joblib
import azureml.train.automl
from azureml.core import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType


def init():
    global deploy_model
    model_name = 'best_model_output'
    #model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_fit_model.pkl')
    model_path = Model.get_model_path(model_name)
    deploy_model = joblib.load(model_path)

def run(data):
    try:
        temp = json.loads(data)
        data = pd.DataFrame(temp['data'])
        result = deploy_model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error