<!--
   Copyright (C) 2021-2022 AIR Institute
   
   This program and the accompanying materials are made
   available under the terms of the Eclipse Public License 2.0
   which is available at https://www.eclipse.org/legal/epl-2.0/
   
   SPDX-License-Identifier: EPL-2.0
-->
# Intro

Smartclide provides an environment to support the development of service-oriented software. This service classification aims to classify the same web services based on their functionality, which can be helpful in later stages such as service composition. 

- [Requirements](#requirements)
- [Quick Installation](#quick-installation)
- [DLE Service Classification AI Model](#dle-service-classification-ai-model)
- [Usage](#usage)
  - [Simple Usage](#simple-usage)
  - [Singleton classes Usage](#singleton-classes-usage)




# Requirements

The list of the third-party library are listed on requirments.txt file; however, the two main used library and requirements are:

- Python 3.7+
- [HuggingFace](https://huggingface.co/)
- [scikit-learn](https://scikit-learn.org)
- System Requirements: CPU: 2cv RAM: 8 GB

Note: The minimum requirement for using this package is 43GB of storage. The reason is during package installation, and it uses temp storage and packages like a torch, which exceeds more spaces during the installation process.

However, if the package is installing on AWS EC2, the increasing storage or, configuration in TMPDIR is necessary to avoid reciving "Errror:No space left on device" [more info](https://stackoverflow.com/questions/55103162/could-not-install-packages-due-to-an-environmenterror-errno-28-no-space-left)

# DLE Service Classification AI Model
This service classification takes advantage of text classification trends and deep learning methods. After processing the collected dataset, techniques and algorithms have been applied to solve the service classification problem. The real-world service classification in the industry needs to consider performance and minimum resource usage. Therefore, this subcomponent has provided two models; the first model was trained by keyword-based feature extraction and SVM, which is set as the default model. The second model uses BERT, which is a transformer that Google has introduced. The significant feature of BERT is using BiLSTM, which is the most promising model for learning long-term dependencies. 

The following parameter is mandatory to route the requests to this implemented Model:
- Service Name: A string including web service name.
- Service Description: a service description in string format.

# Quick Installation
The trained models have been packaged using the Python Setuptools library. Therefore, this component need to install the related package by cloning the package, browsing to main directory, and executing “python3 -m pip install . --upgrade” command. 
```
git clone https://github.com/eclipse-researchlabs/smartclide-smart-assistant.git
cd smartclide-dle-models/serviceclassification 
python3 -m pip install . --upgrade
```
Testing the module installation
```
python3 servclassify/examples/classify_service.py
```


In SmartCLIDE platform, These models need a  gateway between the trained models and user interfaces. [smartclide-dle](https://github.com/eclipse-researchlabs/smartclide-smart-assistant/tree/main/smartclide-dle) component provides API for using these models.

# Usage
This library provides two trained models; first, the prediction by ML model. Second, predict using the DL model; the default configuration uses the ML model, which is lighter. You can select method="Default" for using ML model or method= 'Advanced' for using DL model. However, the "AIPipelineConfiguration" class is configured for Default mode; for using  method= 'Advanced', you need to change the configuration in the AIPipelineConfiguration file to set service_classification_method= 'Advanced' in AIPipelineConfiguration.py and reinstall the package.
### Simple Usage
```
from servclassify import PredictServiceClassModel

service_name="service name text"
service_desc="find the distination on map"
method="Default"

predict_service_obj = PredictServiceClassModel()
result = predict_service_obj.predict(service_name, service_description, method=method)
print(result) 

```

The above class demonstrate using service classification interface class whuich is PredictServiceClassModel. After defining this class we can use it for predicting service class:
```
{'result': [{'Service_name': 'service name text', 'Method': 'Default', 'Service_id': None, 'Service_class': ['Mapping', '']}]}

```
✨Note ✨
The advanced method will return the top 2 categories assigned to service metadata input. the format of output will be:

```
{'result': [{'Service_name': 'service name text', 'Method': 'Default', 'Service_id': None, 'Service_class': ['Mapping', 'Transportation']}]}

```
### Singleton Classes Usage
In SmartCLIDE, many tasks require to run in the background independently of the user interface (UI). AI Models is one of these tasks that need to serve requests in real-time and return results. Consequently, loading the AI model can be time-consuming due to late response. A strategy such as using singleton classes for loading the models can help minimize the application UI load, improve availability, and reduce interactive response times. 

```
from typing import Tuple 
from typing import List
from servclassify import PredictServiceClassModel

class Classify_service:
    def __init__(self):
        '''
        The DL models  input parameter for PredictServiceClassModel mention loading service model
        '''
        self.predict_service_obj = PredictServiceClassModel()

    def predict(self, service_id: str, service_name: str, service_description: str, method:str = 'Default') -> Tuple[str,str]:
        # predict
        result = self.predict_service_obj.predict(service_name, service_description, method=method)
        return result
        

#Loading model recommended to execute on background

model2 = Classify_service()
service_id=1
service_name="service name text"
service_desc="find the distination on map"
method="Advanced"
result=model2.predict(service_id,service_name, service_desc,method)
print(result) 
```

You can find the example code which are in python script in the example folder.

