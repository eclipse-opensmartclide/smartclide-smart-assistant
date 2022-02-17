from typing import Tuple 
from typing import List
from smartclide_service_classification_autocomplete import PredictServiceClassModel

class Classify_service:
    def __init__(self):
        '''
        The DL models  input parameter for PredictServiceClassModel mention loading service model, if the trained models havenot placed in trained_models folder, you need new the class like: PredictServiceClassModel(False)
        '''
        self.predict_service_obj = PredictServiceClassModel()

    def predict(self, service_id: str, service_name: str, service_description: str, method:str = 'Default') -> Tuple[str,str]:
        # predict
        result = self.predict_service_obj.predict(service_name, service_description, method=method)
        return result
 


'''
Loading model recommended to execute on background
Note: The DL models  input parameter for PredictServiceClassModel mention loading service model; if the trained models have not placed in the trained_models folder, you need a new the class like PredictServiceClassModel(False)
'''
model2 = Classify_service()


service_id=1
service_name="service name text"
service_desc="service desc text"
method="Advanced"

result=model2.predict(service_id,service_name, service_desc,method)
print(result) 

'''
#OUTPUT:
#The advanced method will return the top 2 categories assigned to service metadata input. the format of output will be:

{'result': [{
    'Service_name': 'test service', 
    'Method': 'Advanced', 
    'Service_id': foo,
    'Service_class': '(predicted_category1,predicted_category2)'  
    }]
}

Warning : if the model file is not avalible you will recive :'Under develope,waiting for upload git lgfs file ...' as output
'''