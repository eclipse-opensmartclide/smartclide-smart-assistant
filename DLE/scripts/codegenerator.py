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
from CodeGeneration import *


#sample Code
codeGenObj=CodeGeneration("top_poject_source_codes.csv",False)

#sample1
seed_text='import'
predictionList=codeGenObj.generate_code(seed_text,1)