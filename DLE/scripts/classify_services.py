#!/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.

from configparser import ConfigParser
#Read config.ini file
config_object = ConfigParser()
config_object.read("config.ini")
#Get the _PATH_ROOT_
_PATH_ROOT_ = config_object["Path"]["root"]


import sys
sys.path.insert(1, _PATH_ROOT_ +'/DLE/utils')
sys.path.insert(1, _PATH_ROOT_ +'/DLE/models')
from ImportingModules import *
from ServiceClassification import *



X2 = "If you are still coming to grips with the very basics of programming, you really want to work your way through a few tutorials first. The Python Wiki lists several options for you. Stack Overflow may not be the best place to ask for help with your issues."
X3 = "This mashup combines job listings from Indeed with company reviews from Glassdoor for job searches within the data science"
X4 = "This Python example demonstrates how to perform datatype binding using the Numpy library. Also, this test module performs table-related operations."
X5="This independent report, Embracing the Global Open Banking Opportunity, is the summary and analysis of research conducted by the research firm Twimbit. This report covers the overall landscape of global open banking initiatives."
X6="Tipalti removes the friction of financial operations, utils invoices, and reconciling payments.  We help over 1,500 customers pay over 4 million suppliers. With a 99% retention rate, Tipalti is both an award-winning and best-reviewed payables automation platform for hyper-growth businesses. "

# serviceObjML = ServiceClassification("Clustered_Services_Bert_V_1.csv")
# serviceObjML.loadData()
# pried2=serviceObj.predictFastTextModel(X6)
# print('predictFastText: '+pred2[0])


serviceObjML = ServiceClassification("Clustered_Services_Bert_V_1.csv",True)
serviceObjML.loadData()

pred = serviceObjML.predictBOWML(X6)
print('predictBOWML: '+pred[0])