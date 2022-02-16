#!/usr/bin/python3
# Eclipse Public License 2.0

from .ServiceClassification import *
from flask import jsonify


class PredictServiceClassModel():

    def __init__(self,load_model=False):
        if load_model:
          service_classifier_obj= ServiceClassificationModel()
          service_classifier_obj.loadTrainedClassifier()
          self.classifier = service_classifier_obj
    
    def predict(self, serviceName, serviceDesc, serviceID=None, method="Default"):
        result = None
        errorMsg = None
        serviceClass = ''
    
        if not method in (ServiceClassificationModel.method):
            result = {"Error": "The Method is invalid"}
            return result

        if len(serviceDesc) > 2:
          
            if method == 'BSVM':
                serviceObjBSVM = ServiceClassificationModel(True)
                pred = serviceObjBSVM.predictBSVMModel(serviceDesc)

            if method == 'Default':
                serviceObjML = ServiceClassificationModel(True, 'Description', 'Category')
                serviceClass = serviceObjML.predictBOWML(serviceDesc)

            if method == 'Advanced':
                serviceClass="Under develope,waiting for upload git lgfs file ..."
                # serviceClass =self.classifier.get_prediction(serviceDesc,k=2)
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

