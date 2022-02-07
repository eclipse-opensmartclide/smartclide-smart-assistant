#!/usr/bin/python3
# Eclipse Public License 2.0

from .ServiceClassification import *
from flask import jsonify


class PredictServiceClassModel():
    """
    These trained models need a gateway between the trained models and user interfaces.
    This Class provide interface for models and DLE apis
    """
    def predict(self, serviceName, serviceDesc, serviceID=None, method="Default"):
        """
        :param serviceName:      string param specifies service Name 
        :param serviceDesc:      string param specifies service Description 
        :param serviceID:        int param specifies specifies service ID in service registery, optinal
        :param method:           string param specifies using trined models, "Default" use ML model and "Advanced" use DL model
        """
        result = None
        errorMsg = None
        serviceClass = ''
        if not method in (ServiceClassificationModel.method):
            result = {"Error": "The Method is invalid"}
            return result
        if len(serviceDesc) > 2:
            if method == 'Advanced':
               #TODO
               serviceClass="Under develope"
                
            if method == 'Default':
                serviceObjML = ServiceClassificationModel(True, 'Description', 'Category')
                pred = serviceObjML.predictBOWML(serviceDesc)
                serviceClass = pred[0]
            results = []
            if not errorMsg == None:
                result = {
                    "Error": errorMsg,
                }
            else:
                result = {
                    "Service_name": serviceName,
                    "Method": method,
                    "Service_id": serviceID,
                    "Service_class": serviceClass
                }
        else:
            result = {
                "Error": "Minimum length should be 3",
            }
        results.append(result)
        return ({'result': results})
