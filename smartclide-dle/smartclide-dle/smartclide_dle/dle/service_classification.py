#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.


import requests
from typing import Tuple, List

from servclassify import PredictServiceClassModel


class ServiceClassification:

    def __init__(self):
        self.service_classification = PredictServiceClassModel()

    def predict(self, service_id: str, service_name: str, service_description: str, method:str = 'Advanced') -> Tuple[List[str],str,str]:

        # predict
        result = self.service_classification.predict(service_name, service_description, method=method)

        # extract results
        method = result['Method']
        service_id = result['Service_id']
        categories = result['Service_class']

        categories = list(set(categories))
        categories = ['Generic Service'] if not categories or (len(categories) == 1 and not categories[0]) else categories

        return categories, method, service_id