# !/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.

from .ServiceClassification import *
from flask import jsonify

class PredictServiceClassModel():
     def predict(self,serviceName,serviceDesc,method="Default"):
         error=''
         serviceClass=''
         if len(serviceDesc) > 2:
             if method == 'Fasttext':
                 serviceObjFastText = ServiceClassificationModel(True)
                 serviceObjFastText.loadData()
                 pred = serviceObjFastText.predictFastTextModel(serviceDesc)
                 if not pred:
                     error = 'Predicting is failed'
                 else:
                     serviceClass = pred[0]
             if method == 'BSVM':
                 serviceObjBSVM = ServiceClassificationModel(True)
                 serviceObjBSVM.loadData()
                 pred = serviceObjBSVM.predictBSVMModel(serviceDesc)
                 if not pred:
                     error = 'Training need more resource'
                 else:
                     serviceClass = pred[0]
             if method == 'Default':
                 serviceObjML = ServiceClassificationModel(True)
                 serviceObjML.loadData()
                 pred = serviceObjML.predictBOWML(serviceDesc)
                 # if pred:
                 #     error = 'Training need more resource'
                 # else:
                 serviceClass = pred

             results = []

             if not len(error) < 1:
                 result = {
                     "Error": error,
                 }
             else:
                 result = {
                     "service_name": serviceName,
                     "Method": method,
                     "Service_class": serviceClass
                 }
             results.append(result)
             return ({'result': results})






