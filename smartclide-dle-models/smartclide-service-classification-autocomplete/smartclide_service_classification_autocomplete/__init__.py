#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************

import io
import os
import requests
from .ServiceClassification import  ServiceClassificationModel
from .AutocompleteCode import AutocompleteCodeModel
from .PredictServiceClass import PredictServiceClassModel

_ROOT = os.path.abspath(os.path.dirname(__file__))

def getPackagePath():
    """
    Returns Package path.
    """
    return _ROOT

def getFilePath_(path):
    """
    Returns file path in Package.
    """
    return os.path.join(_ROOT, path)



