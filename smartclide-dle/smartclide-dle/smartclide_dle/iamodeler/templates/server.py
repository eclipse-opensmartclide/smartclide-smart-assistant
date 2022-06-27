#!/usr/bin/env python3
#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************

import pickle

import pandas as pd
from flask import Flask, request
from flask_cors import CORS
from flask_restx import Api, Resource, fields

estimator, encoder, input_names = pickle.load(open("model.pkl", "rb"))


def predict(data):
    if not isinstance(data, pd.DataFrame):
        # Add column name for preprocessing
        attributes = input_names
        data = pd.DataFrame(data, columns=attributes)

    predictions = estimator.predict(data)

    if encoder is None:
        return predictions
    else:
        return encoder.inverse_transform(predictions)


app = Flask(__name__, static_url_path='')
CORS(app)

api = Api(app=app, version="1.0", title="My predictor", description="Created with SmartCLIDE")
api_namespace = api.namespace("api", description="General methods")

predict_model = api.model('Prediction request', {
    'data': fields.List(fields.List(fields.Raw), description={{description}},
                        example={{example}})
})

predict_result_model = api.model('Prediction results', {
    'predictions': fields.List(fields.Raw, description="List of values predicted", example={{output_example}})
})


@api_namespace.route('/predict')
class Predict(Resource):
    @api.expect(predict_model)
    @api.marshal_with(predict_result_model, code=200, description='OK')
    def post(self):
        """Make predictions using the exported model"""
        data = request.json['data']

        predictions = predict(data)

        return {"predictions": predictions.tolist()}


def main():
    app.run(debug=False, host='0.0.0.0')


if __name__ == '__main__':
    main()
