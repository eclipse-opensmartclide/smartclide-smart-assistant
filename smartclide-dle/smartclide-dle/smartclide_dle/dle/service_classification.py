#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.


from typing import Tuple

from smartclide_service_classification_autocomplete import PredictServiceClassModel


class ServiceClassification:

    def predict(self, service_id: str, service_name: str, service_description: str, method:str = 'default') -> Tuple[str,str]:

        # predict
        service_classification = PredictServiceClassModel()
        result = service_classification.predict(service_name, service_description, method=method)

        # extract results
        method = result['result'][0]['Method']
        category = result['result'][0]['Service_class']
        category = 'Generic Service' if not category else category

        return category, method