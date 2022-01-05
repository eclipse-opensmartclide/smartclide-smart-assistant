# !/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.

from .ServiceClassification import *
from flask import jsonify


class PredictServiceClassModel():
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
                if len(pred[0]) < 1:
                    errorMsg = 'Training need more resource'
                else:
                    serviceClass = pred[0]
            if method == 'Default':
                serviceObjML = ServiceClassificationModel(True, 'Description', 'Category')
                pred = serviceObjML.predictBOWML(serviceDesc)
                #                         if len(pred[0]) < 1:
                #                             errorMsg = 'Training need more resource'
                #                         else:
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
