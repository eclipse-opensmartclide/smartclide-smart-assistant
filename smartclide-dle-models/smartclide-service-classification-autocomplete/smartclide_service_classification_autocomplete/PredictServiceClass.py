# !/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.

from .ServiceClassification import *
from flask import jsonify

class PredictServiceClassModel():
     def predict(self,serviceName,serviceDesc,serviceID,method="Default"):
            result=None
            errorMsg=None
            serviceClass=''
            print(ServiceClassificationModel.method)
            if not method in (ServiceClassificationModel.method): 
                     result = {"Error": "The Method is invalid" }
                     return  result
            if len(serviceDesc) > 2:
                    if method == 'BSVM':
                        serviceObjBSVM = ServiceClassificationModel(True)
                        print('yessss')
                        pred = serviceObjBSVM.predictBSVMModel(serviceDesc)
                        if len(pred[0]) < 1:
                            errorMsg = 'Training need more resource'
                        else:
                            serviceClass = pred[0]
                    if method == 'Default':
                        serviceObjML = ServiceClassificationModel(True, 'Description','Category_lable')
                        pred = serviceObjML.predictBOWML(serviceDesc)
#                         if len(pred[0]) < 1:
#                             errorMsg = 'Training need more resource'
#                         else:
                        serviceClass = pred[0]
                    results = []
                    if not errorMsg==None:
                        result = {
                           "Error": errorMsg,
                        }
                    else:
                        result = {
                         "service_name": serviceName,
                         "Method": method,
                         "service_id":serviceID,
                         "Service_class": serviceClass
                       }
                    results.append(result)
            return ({'result': results})






