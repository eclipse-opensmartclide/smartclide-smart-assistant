import io
import os
import requests
from .ServiceClassification import  ServiceClassificationModel
from .AutocompleteCode import AutocompleteCodeModel
from .PredictServiceClass import PredictServiceClassModel

modelTrainedInfo = {
        0: {"file_id":"1zmm7337MJp3ddODtscqLXMwOOlQP5H1r",
            "file_name":"service_classification_bow_svc.pkl"},   
#         1: {"file_id":"1zmm7337MJp3ddODtscqLXMwOOlQP5H1r",
#             "file_name":"service_classification_bert_svc.pkl"},
#         2:{"file_id":"1e4Qy1glbdH9EYoaZXAzsbc7U0z8OWAzo",
#            "file_name":"code_generation_trained_distilgpt2.pt"}, 
        1:{"file_id":"12BWvgzhnyD8R2lNC5h00x27-kujevmCh",
           "file_name":"GPTgenerator.pkl"},     
    
    }

_ROOT = os.path.abspath(os.path.dirname(__file__))

def getPackagePath():
    return _ROOT

def getFilePath_(path):
    return os.path.join(_ROOT, path)

def getmodelTrainedFileArray():
    """
    Returns trained models file information list.
    """
    return modelTrainedFile

def download_file_from_google_drive(id, destination):
    """
    Download heavy models from google drive
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

                

    
trainedModelDir=path=getPackagePath()+'/trained_models/' 
for i in range(0,2):    
    path= trainedModelDir+modelTrainedInfo[i]['file_name']    
    if not os.path.isfile(path):
        fileId= trainedModelDir+modelTrainedInfo[i]['file_id']
        print(modelTrainedInfo[i]['file_name']+" is downloading .....")
        download_file_from_google_drive(fileId, path)
        print(modelTrainedInfo[i]['file_name']+" is Downloaded.")    




  
    
    
   
