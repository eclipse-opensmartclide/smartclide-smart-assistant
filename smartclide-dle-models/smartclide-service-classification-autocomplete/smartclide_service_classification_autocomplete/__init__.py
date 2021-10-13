from .ServiceClassification import  ServiceClassificationModel
from .AutocompleteCode import AutocompleteCodeModel
from .PredictServiceClass import PredictServiceClassModel
import io
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
def getPackagePath():
    return _ROOT

def getFilePath_(path):
    return os.path.join(_ROOT, path)
