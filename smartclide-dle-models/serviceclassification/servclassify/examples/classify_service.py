#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************

from typing import Tuple 
from typing import List
from servclassify import PredictServiceClassModel

class Classify_service:
    def __init__(self):
        '''
        The PredictServiceClassModel __init__  loading trained model in background
        '''
        self.predict_service_obj = PredictServiceClassModel()

    def predict(self, service_id: str, service_name: str, service_description: str, method:str) -> Tuple[str,str]:
        # predict
        result = self.predict_service_obj.predict(service_name, service_description, method=method)
        return result
 


'''
Loading model recommended to execute on background
'''
model2 = Classify_service()


service_id=1
service_name="service name text"
service_desc="find the distination on map"
method="Default"

result=model2.predict(service_id,service_name, service_desc,method)
print(result) 

'''


-------------------------------------------
# For smartclide-dle API need to get following parametters:(method can be "Default" or "Advanced")
-------------------------------------------
{
    "service_id":  34333,
    "method":  "Default",
    "service_name": " TransLoc openAPI",
    "service_desc":"The TransLoc OpenAPI is a public RESTful API which allows developers to access real-time vehicle tracking information and incorporate this data into their website or mobile application."
}

-------------------------------------------
# OUTPUT of For smartclide-dle API will be:
--------------------------------------------

#OUTPUT:
#The advanced method will return the top 2 categories assigned to service metadata input. the format of output will be:

{'result': [{
  'Service_name': 'service name text',
  'Method': 'Default',
  'Service_id': None, 
  'Service_class': ['predicted_class1', '']
  }]}

Warning : or using method= ‘Advanced’ you need change upgrade the configuration in the AIPipelineConfiguration file to set service_classification_method= ‘Advanced’in AIPipelineConfiguration.py and reinstall package.
'''
