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
import wget
import logging
import requests
from .ServiceClassification import ServiceClassificationModel
from .PredictServiceClass import PredictServiceClassModel

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
        self.serviceClsObj = ServiceClassificationModel()
        self.service_classifier_url = str(self.s3_bucket_url) + 'pytorch_model.bin'
        self.service_ml_classifier_url = str(self.s3_bucket_url) + 'service_classification_bow_svc.pkl'
  
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
            trained_models_folder = self.serviceClsObj.getTrainedModelsDirectory()
        except:
            logging.error("DLE: Train folder not found.")

        '''
        Download DL Service classification from s3
        '''
        model_classifier_path = os.path.join(trained_models_folder,self.serviceClsObj.trained_model_folder)+'pytorch_model.bin'
        output_dir = os.path.join(trained_models_folder, self.serviceClsObj.trained_model_folder)
        if not (os.path.exists(model_classifier_path)):
            logging.info("The models need to download to " + str(output_dir))
            self.download_object(self.service_classifier_url,output_dir, 'Service DL classifier')

        '''
        Download ML Service classification from s3
        '''
        # model_ml_classifier_path = os.path.join(trained_models_folder, 'service_classification_bow_svc.pkl')
        # if not (os.path.exists(model_classifier_path)):
        #     logging.info("The models need to download to " + + str(trained_models_folder))
        #     self.download_object(self.service_ml_classifier_url,trained_models_folder, 'Service ML classifier')


download_file_obj = DownloadFile()
download_file_obj.download_Models()
