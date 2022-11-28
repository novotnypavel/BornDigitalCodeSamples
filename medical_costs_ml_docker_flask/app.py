# Local imports
import datetime
# Third part imports
from flask import request
import pandas as pd

from ms import app
from ms.functions import get_model_response
from ms.functions import preprocess_data


model_name = "Patient's Medical Costs Prediction"
model_file = 'rf_model.sav'
version = "v1.0.0"


@app.route('/info', methods=['GET'])
def info():
    """Return model information, version, how to call"""
    result = {}

    result["name"] = model_name
    result["version"] = version

    return result


@app.route('/health', methods=['GET'])
def health():
    """Return service health"""
    return 'ok'


@app.route('/predict', methods=['POST'])
def predict():
    feature_dict = request.get_json()
    if not feature_dict:
        return {
            'error': 'Body is empty.'
        }, 500
    try:
        X = preprocess_data(feature_dict)
        response = get_model_response(X)
    except ValueError as e:
        return {'error': str(e).split('\n')[-1].strip()}, 500

    return response, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0')