import sys
import os
import io
import json


# import cv2
import mxnet as mx

import warnings
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'deploy'))
import face_model
from types import SimpleNamespace

warnings.filterwarnings("ignore",category=FutureWarning)

# sys.path.append(os.path.join(os.path.dirname(__file__), '/opt/ml/code/package'))

import pickle
# from io import StringIO
from timeit import default_timer as timer
from collections import Counter

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    # from autogluon import ImageClassification as task

import flask

# import sagemaker
# from sagemaker import get_execution_role, local, Model, utils, fw_utils, s3
# import boto3
import tarfile

# import pandas as pd

prefix = '/opt/ml/'
# model_path = os.path.join(prefix, 'model')
model_path = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            print(os.listdir("/opt/program/"))

#             parser = argparse.ArgumentParser(description='face model test')
#             # general
#             parser.add_argument('--image-size', default='112,112', help='')
#             parser.add_argument('--mtcnn-model', default='/opt/program/mtcnn-model/', help='path to load mtcnn model.')
#             parser.add_argument('--model', default='/opt/program/model-r100-ii/model,0', help='path to load model.')
#             parser.add_argument('--ga-model', default='', help='path to load model.')
#             parser.add_argument('--gpu', default=0, type=int, help='gpu id')
#             parser.add_argument('--use-gpu', default=False, type=bool, help='use gpu')
#             parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
#             parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
#             parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
#             args = parser.parse_args()
            
            args = {'image_size': '112,112', 'mtcnn_model': '/opt/program/mtcnn-model/', 'model': '/opt/program/model-r100-ii/model,0', 'ga_model': '', 'gpu': 0, 'use_gpu': False, 'det': 0, 'flip': 0, 'threshold': 1.24}
            args = SimpleNamespace(**args)

            cls.model = face_model.FaceModel(args)

        return cls.model

    @classmethod
    def predict(cls, input_np):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        net = cls.get_model()
        img = net.get_input(input_np)
        if img is None:
            return None
        feature = net.get_feature(img)
        return feature

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    # Convert from CSV to pandas
    if flask.request.content_type == 'application/json':
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)
        data_np = np.asarray(data['data'], dtype=np.uint8)
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.keys()))

    # Do the prediction
    feature = ScoringService.predict(data_np)
    output_dict = {"Output": {"Feature": str(feature.tolist()),},}
    output_dict_str = json.dumps(output_dict)
    print(output_dict_str)

    # Convert from numpy back to CSV
    out = io.StringIO()
    # pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    out.write(output_dict_str)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='application/json')
