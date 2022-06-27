#!/usr/bin/python3
#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************


from .ServiceClassification import *
from flask import jsonify


class PredictServiceClassModel():
    classifier=None
    
    def __init__(self,load_model=True):
        if load_model:
            service_classifier_obj= ServiceClassificationModel()
            if service_classifier_obj.loadTrainedClassifier():
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
                #return a top predicted service category
                serviceObjML = ServiceClassificationModel(True, 'Description', 'Category')
                serviceClasses = serviceObjML.predictBOWML(serviceDesc)

            if method == 'Advanced':
                #return two top predicted service categories
                if self.classifier is not None:
                    serviceClasses =self.classifier.get_prediction(serviceDesc,k=2)
                else:
                    serviceClasses=["Under develope,waiting for upload git lgfs file ...",""]


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
                    "Service_class": serviceClasses
                }
        else:
            result = {
                "Error": "Minimum length should be 3",
            }
        results.append(result)
        return ({'result': results})
