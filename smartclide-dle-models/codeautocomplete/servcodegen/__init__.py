import io
import os
import wget
import logging
import requests
from .CodeGeneration import CodeGenerationModel
from .AutocompleteCode import AutocompleteCodeModel

try:
    import coloredlogs
    coloredlogs.install() 
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
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


def install(package):
    import pip
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

class DownloadFile():

    def __init__(self, code_generator_url='', service_classifier_url=''):
        '''
         Use S3 as default public object url
        '''
        self.s3_bucket_url = 'https://smartclide.s3.eu-central-1.amazonaws.com/'
        self.codeGenObj = CodeGenerationModel()
        self.code_generator_url = self.s3_bucket_url + self.codeGenObj.trained_model
        self.distilGPT2_code_generator_url = self.s3_bucket_url + 'distilGPT2/pytorch_model.bin'
        # self.distilGPT2_code_generator_url = 'https://smartclidetemp.s3.eu-central-1.amazonaws.com/pytorch_model.bin'

    def download_object(self, url, out_dir, model_name):
        '''
        Download objects from url 
        '''
        try:
            logging.info("Downloading " + str(model_name) + " Model from S3 is started...")
            url=str(url)
            out_dir=str(out_dir)
            file_name = wget.download(url, out_dir)
            logging.info("Downloading " + str(model_name) + " Model from S3 is finished")
        except:
            logging.error("DLE: Download "+model_name+" Failed: Check URL or access permission")

    def download_Models(self):
        '''
        Download large models from S3 
        '''
        try:
            trained_models_folder = self.codeGenObj.getTrainedModelsDirectory()
        except:
            logging.error("DLE: Train folder not found.")

        '''
        Download DL Code generator from s3
        '''
        model_codegen_path = os.path.join(trained_models_folder, self.codeGenObj.trained_model)
        if not (os.path.exists(model_codegen_path)):
            logging.info("The models need to download to " + str(trained_models_folder))
            self.download_object(self.code_generator_url, trained_models_folder,'DL CodeGeneration')


        '''
        Download distilGPT2 DL Code generator from s3
        '''
        model_distilGPT2_path = os.path.join(trained_models_folder,self.codeGenObj.trained_model_distilGPT2_folder)+'pytorch_model.bin'
        output_dir_distil = os.path.join(trained_models_folder, self.codeGenObj.trained_model_distilGPT2_folder) 
        if not (os.path.exists(model_distilGPT2_path)):
            logging.info("The models need to download to " + str(output_dir_distil))
            self.download_object(self.distilGPT2_code_generator_url,output_dir_distil,'distilGPT2 CodeGeneration')





download_file_obj = DownloadFile()
download_file_obj.download_Models()
