#!/usr/bin/python3
# Eclipse Public License 2.0

from .ServiceClassification import *
#from flask import jsonify

from .AIPipelineConfiguration  import *
   
import logging    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

class PredictServiceClassModel():
    classifier=None
    cnf_models=None

    def __init__(self):
        self.cnf_models= AIPipelineConfiguration()
        if self.cnf_models.service_classification_method=="Advanced":
            service_classifier_obj= ServiceClassificationModel()
            if service_classifier_obj.loadTrainedClassifier():
                self.classifier = service_classifier_obj
    
    def predict(self, serviceName, serviceDesc, serviceID=None, method="Default"):
        results=[]
        result = {
                    "Service_name": '',
                    "Method": '',
                    "Service_id": '',
                    "Service_class": ['','']
                }        
        errorMsg = None
        serviceClass = ''
    
        if not method in (ServiceClassificationModel.method):
            logging.error("DLE: The Method parameter input is invalid it should be 'Default' or 'Advanced' ")
            return result

        if (method=="Advanced" and self.cnf_models.service_classification_method=="Default"):
            logging.error("DLE: The \"AIPipelineConfiguration\" class is configured for Default mode; Please use \"Default\" or upgrade the configuration in the AIPipelineConfiguration file to use advanced mode.")
            return result

        if len(serviceDesc) > 1:
          
            if method == 'BSVM':
                serviceObjBSVM = ServiceClassificationModel(True)
                pred = serviceObjBSVM.predictBSVMModel(serviceDesc)

            if method == 'Default':
                #return a top predicted service category
                serviceObjML = ServiceClassificationModel(True, 'text', 'category')
                serviceClasses = serviceObjML.predictBOWML(serviceDesc)

            if method == 'Advanced':
                #return two top predicted service categories
                if self.classifier is not None:
                    serviceClasses =self.classifier.get_prediction(serviceDesc,k=2)
                else:
                    serviceClasses=["Under develope,waiting for upload git lgfs file ...",""]



            result = {
                    "Service_name": serviceName,
                    "Method": method,
                    "Service_id": serviceID,
                    "Service_class": serviceClasses
                }
        else:
            logging.error("The service description can not be empty.")
            return result          

        results.append(result)
        return ({'result': results})

