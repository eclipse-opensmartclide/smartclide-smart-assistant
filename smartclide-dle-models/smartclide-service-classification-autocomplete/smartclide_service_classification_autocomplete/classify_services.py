#!/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.

# from .PredictServiceClass import *
from smartclide_service_classification_autocomplete.PredictServiceClass import PredictServiceClassModel


X6="Tipalti removes the friction of financial operations, utils invoices, and reconciling payments.  We help over 1,500 customers pay over 4 million suppliers. With a 99% retention rate, Tipalti is both an award-winning and best-reviewed payables automation platform for hyper-growth businesses. "

serviceName='test'
serviceDesc=''
method="Default"
serviceID=11

serviceObj = PredictServiceClassModel()
serviceDesc=X6
pred = serviceObj.predict(serviceName,serviceDesc,serviceID,method="BSVM")
print(pred)

