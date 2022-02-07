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



