# Smartclide-DLE

The SmartCLIDE DLE and smart assistant has brought together the IDE assistant features within one component. Proposed models try to provide a learning algorithm with the information that data carries, including internal data history and external online web services identified from online resources.
 After providing AI models, the smart assistant and DLE models are deployed as APIs REST encapsulated in a Python package, which guarantees the portability of this component between Operating Systems. Afterward, to expose the component functionality, we have chosen to visualize these requests and responses through the API swagger, which consists of an interactive web interface.



## Requirements 

The list of the third-party library are listed on requirments.txt files in each sub-components; however, the two main used library and requirements are:

- Python 3.7+
- [Pytorch](https://pytorch.org/)
- [HuggingFace](https://huggingface.co/)
- [scikit-learn](https://scikit-learn.org)

Note: The minimum requirement for installing each transformer learning models using this package is 30GB of disk storage, 2vCPU, 4GB RAM. The reason of disk storage is during package installation, and it uses temp storage and packages like a torch, which exceeds more spaces during the installation process.
To use less storage, you can disable caching behavior by using `--no-cache-dir` in pip install command. [more info](https://pip.pypa.io/en/stable/topics/caching/) 

## How to Build DLE component 
In SmartCLIDE platform, trained models need a gateway between the trained models and user interfaces. In this regard, the smart-assistant will support this option through Flask-restx APIs developed, which serve SmartCLIDE DLE (Deep Learning Engine) and Smart Assistant. Moreover, some statistical models are supported by smart-assistant as well.In this regard, DLE needs to install both trained models sub-components and also API gateway.

### API Gateway Installation

Install prerequisites :

```bash
sudo python3 -m pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
sudo pyton3 -m pip install git+https://github.com/Dih5/zadeh
```

install smartclid getway:
```bash
sudo apt update
sudo apt install python3 python3-pip nodejs npm -y
sudo npm install -g pm2
```

After installation, SmartCLIDE smart-assistant provides the API specification using Swagger specification on "http://<SmartCLIDE-host>/dle", and "http://<SmartCLIDE-host>/iamodeler". 
In summary, the available end-points are:
- http://<SmartCLIDE-host>/dle/codegen
- http://<SmartCLIDE-host>/dle/acceptance
- http://<SmartCLIDE-host>/dle/environment
- http://<SmartCLIDE-host>/dle/templatecodegen
- http://<SmartCLIDE-host>/dle/serviceclassification
- http://<SmartCLIDE-host>/dle/bpmnitemrecommendation
- http://<SmartCLIDE-host>/dle/predictivemodeltoolassistant
- http://<SmartCLIDE-host>/iamodeler/classification/bayes
- http://<SmartCLIDE-host>/iamodeler/classification/extra-trees
- http://<SmartCLIDE-host>/iamodeler/supervised/classification/forest
- http://<SmartCLIDE-host>/iamodeler/supervised/classification/gradient
- http://<SmartCLIDE-host>/iamodeler/supervised/classification/logistic
- http://<SmartCLIDE-host>/iamodeler/supervised/classification/mlp
- http://<SmartCLIDE-host>/iamodeler/supervised/classification/neighbors
- http://<SmartCLIDE-host>/iamodeler/supervised/classification/sv
- http://<SmartCLIDE-host>/iamodeler/supervised/classification/tree
- http://<SmartCLIDE-host>/iamodeler/supervised/regression/gradient
- http://<SmartCLIDE-host>/iamodeler/supervised/regression/linear
- http://<SmartCLIDE-host>/iamodeler/supervised/regression/mlp
- http://<SmartCLIDE-host>/iamodeler/supervised/regression/neighbors
- http://<SmartCLIDE-host>/iamodeler/supervised/regression/sv
- http://<SmartCLIDE-host>/iamodeler/supervised/regression/tree

### Sub-component Quick Installation
The trained models have been packaged using the Python Setuptools library. Therefore, this component need to install the related package by cloning the package, browsing to main directory, and executing “python3 -m pip install . --upgrade” command. 
```bash
git clone https://github.com/eclipse-opensmartclide/smartclide-smart-assistant.git
cd smartclide-dle-models/<sub-component> 
python3 -m pip install . --upgrade
```


## How to run DLE Component

### Configuration

The application configuration is set via enviroment variables:

- `SA_API_PORT`: Port to bind to (default: `5001`). 
- `SA_API_BIND`: Address to bind to (default: `0.0.0.0`).
- `SA_MONGODB_PORT`: MongoDB database to connect to (default:`27017`).
- `SA_MONGODB_HOST`: MongoDB database to connect to (default: `localhost`).
- `SA_MONGODB_USER`: MongoDB user to connect to db (default: `user`).
- `SA_MONGODB_PASSWROD`: MongoDB password to connect to db (default: `password`).
- `SA_MONGODB_DB`: MongoDB database to connect to (default: `smartclide-smart-assistant`).
- `DLE_BASE_URL`: Base URL for DLE connection (default: `http://smartclide.ddns.net:5001/smartclide/v1/dle`).
- `SMART_ASSISTANT_BASE_URL`: Base URL for Smart Assistant RabbitMQ connection (default: `http://smartclide.ddns.net:5000/smartclide/v1/smartassistant`).
- `RABBITMQ_HOST`: RabbitMQ connection string host (default: `localhost`).
- `RABBITMQ_PORT`: RabbitMQ connection string port (default: `5672`).
- `RABBITMQ_USER`: RabbitMQ connection string user (default: `user`).
- `RABBITMQ_PASSWORD`: RabbitMQ connection string password (default: `password`).
- `RABBITMQ_MAPPINGS`: RabbitMQ mappings between queue and API's endpoint to connect to. (default: `{
	    'acceptance_tests_queue': '{SMART_ASSISTANT_BASE_URL}/acceptance',
	    'bpmn_item_recommendation_queue': '{SMART_ASSISTANT_BASE_URL}/bpmnitemrecommendation',
	    'code_generation_queue': '{SMART_ASSISTANT_BASE_URL}/codegen',
	    'code_repo_recommendation_queue': '{SMART_ASSISTANT_BASE_URL}/coderepo',
	    'enviroment_queue': '{SMART_ASSISTANT_BASE_URL}/enviroment'
	}`). Note: All of them are prefixed with `{SMART_ASSISTANT_BASE_URL}/` before start the connection.

### Run application

Application can be launched with the launch script:

```bash
sudo bash launch.bash
```

Or using PM2:

```bash
sudo pm2 start pm2.json
```

Note: if the script `launch.bash` doesn't works, you can use `launch2.bash` instead.
   
## DLE Sub-components
SmartCLIDE primarily works with text data, therefore, these components have the advantage of text processing trends and deep learning methods. The earlier approaches mostly combined key-word based feature engineering and traditional ML. However,
the keyword-based approaches such as BoW mostly use one-
hot encoded vectors, which are high-dimensional and sparse.
The emergence of word-embedding techniques has im-
proved keyword-based feature engineering. Additionally, the
increasing word embedding of open-source projects such as
[Glove](https://nlp.stanford.edu/projects/glove/), [word2vec](https://www.nltk.org/howto/gensim.html), [BERT](https://huggingface.co/docs/transformers/model_doc/bert), [GPT2](https://huggingface.co/gpt2) help the fast and efficient
low-dimensional representation of text data. Thus, despite these technologies being resource-demanding, SmartcLIDE considered them for some key functinalities.


### Service classification model
Smartclide provides an environment to support the development of service-oriented softwares. The goal of this service classification is to classify the same web services based on their functionality which can be helpful in later stages such as service composition. 


The trained models have been packaged using the Python Setuptools library. Therefore, this component need to install the related package by cloning the package, browsing to main directory, and executing “python3 -m pip install . --upgrade” command. 
```
git clone https://github.com/eclipse-opensmartclide/smartclide-smart-assistant.git
cd smartclide-dle-models/serviceclassification 
python3 -m pip install . --upgrade
```
Testing the module installation
```
python3 servclassify/examples/classify_service.py
```
#### Usage
This library provides two trained models; first, the prediction by ML model. Second, predict using the DL model; the default configuration uses the ML model, which is lighter. You can select method="Default" for using ML model or method= 'Advanced' for using DL model. However, the "AIPipelineConfiguration" class is configured for Default mode; for using  method= 'Advanced', you need to change the configuration in the AIPipelineConfiguration file to set service_classification_method= 'Advanced' in AIPipelineConfiguration.py and reinstall the package.
##### Simple Usage
```python
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
##### Singleton Classes Usage
In SmartCLIDE, many tasks require to run in the background independently of the user interface (UI). AI Models is one of these tasks that need to serve requests in real-time and return results. Consequently, loading the AI model can be time-consuming due to late response. A strategy such as using singleton classes for loading the models can help minimize the application UI load, improve availability, and reduce interactive response times. 

```python
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

### Code completion model
This subcomponent is responsible for generating code based on internal templates. The API returns related code snippets based on templates to implement the workflow represented in BPMN in low code. The first version of this API is designed for finding Java codes.


The trained models have been packaged using the Python Setuptools library. Therefore, this component need to install the related package by cloning the package, browsing to main directory, and executing “python3 -m pip install . --upgrade” command. 
```
git clone https://github.com/eclipse-opensmartclide/smartclide-smart-assistant.git
cd smartclide-dle-models/codeautocomplete 
python3 -m pip install . --upgrade
```
Testing the module installation
```
python3 servcodegen/examples/generate_code.py
```
#### Usage

This library provides code-generator which uses language modeling, and after installation, the library can be used by importing the package. The model predicts the next tokens based on user input; in order to have better results, the following recommendation need to be considered:


- Max_sugges_line specifies max line suggestion; recommended value is between 1-3.
- Max_lenth specifies max length line suggestion, and the recommended value is between 15-20
- Use Singletone call for acceptable response time, which this method is explained in the next section.
- Handling client requests need access to sufficient computing
infrastructure. Therefore, it suggests calling code to autocomplete when the user uses "Tab" or "Dot." 


##### Simple Usage
```python
from servcodegen import AutocompleteCodeModel

model = AutocompleteCodeModel()
method="GPT-2"
lang="java"
max_lenth=20
max_sugges_line=3
code_input="import android."
    
result=model.generateCode(code_input, max_lenth, max_sugges_line,method)
print(result) 

```

The above code demonstrate using servcodegen interface class whuich is AutocompleteCodeModel. the result will be
```
{'result': {'code_sugg': ['import android.os.Bundle ;', 'import android.content.Intent ;', 'import android.content.Context ;'], 'Method': 'GPT-2', 'codeSuggLen': 20, 'codeSuggLines': 3, 'language': 'java'}}
```
✨Note ✨
loding model recommended to execute on background which is explained on singletone classes usage in below.

##### Singleton classes Usage
In SmartCLIDE, many tasks require to run in the background independently of the user interface (UI). AI Models is one of these tasks that need to serve requests in real-time and return results. Consequently, loading the AI model can be time-consuming due to late response. A strategy such as using singleton classes for loading the models can help minimize the application UI load, improve availability, and reduce interactive response times. 

```python
from typing import Tuple 
from typing import List
from servcodegen import AutocompleteCodeModel

class CodeCompletion:
    
    def __init__(self):
        self.model = AutocompleteCodeModel()

    
    def predict2(self, method:str, language:str, code_input:str, code_sugg_len:int, code_sugg_lines:int) -> List[str]:
        # predict
        result = self.model.generateCode(code_input, code_sugg_len, code_sugg_lines,method)
        return result
	
#Loading model recommended to execute on background
codecomplete_obj = CodeCompletion()

#Using loaded model
Method="GPT-2"
lang="java"
max_lenth=20
max_sugges_line=3
code_input="file=new"
result=codecomplete_obj.predict2(Method,lang,code_input,max_lenth,max_sugges_line)
print(result) 
```


### Acceptance test suggestions model
The acceptance test set suggestion system, based on collaborative filtering techniques, is responsible for providing the user with a set of tests defined in Gherkin format to be applied to the workflow defined in the BPMN and help verify if the expectations are met.

The trained models have been packaged using the Python Setuptools library. Therefore, this component need to install the related package by cloning the package, browsing to main directory, and executing “python3 -m pip install . --upgrade” command. 
```
git clone https://github.com/eclipse-opensmartclide/smartclide-smart-assistant.git
cd smartclide-dle-models/cbr-gherkin-recommendation 
python3 -m pip install . --upgrade
```
To install also the dependencies to run the tests or to generate the documentation install some of the extras like (Mind the quotes):

```bash
python3 -m pip install '.[docs]' --upgrade
```

#### Case database initialization

For that purpose, use the following command:

```bash
python3 initialize_cbr_db.py
```

#### Usage

The main class is CBR wich also needs the clases Casebase, Recovery and Aggregation. You need a frist load with all your base cases. After that first inicial load you can pass an empty array to the class initializer:

```python
import pycbr
cbr = pycbr.CBR([],"ghrkn_recommendator","smartclide.ddns.net")
```

#### Add case

The method to add a case must recibe a dictionary with this format:

```python
cbr.add_case({
    'name': "Sting with the file name",
    'text': "All the bpmn text",
    'gherkins': ["list with gherkins text"]
})
```

##### Get recommendation

The  method to get a recommendation must recibe a string with all the bpmn text:

```python
cbr.recommend(bpmn_text)
>>> {
        'gherkins': [["List of list with all the recomended gherkins for the first 5 matches"]],
        'sims': ["List of similarity scores from 0 to 1"]
    }
```

#### Documentation

To generate the documentation, the *docs* extra dependencies must be installed. Furthermore, **pandoc** must be available in your system.

To generate an html documentation with sphinx run:
```bash
make docs
```

To generate a PDF documentation using LaTeX:
```bash
make pdf
```



### Predictive model tool API
This subcomponent utilized the automated machine learning (AutoML) concept, allowing users to define ML actions sequences via an interface.  These sequences contain the Predictive model tool APIs, which include 4 primary steps. 1) Importing data 2)  Creating a supervised model based on regression or classification Model 3) Performing Prediction based on user input 4) Providing validation matric results which can use for visualization.

#### Installation

You probably to set up and use a virtualenv:

```
# Prepare a clean virtualenv and activate it
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
```

Remember to activate it whenever you are working with the package.

To install a **development** version clone the repo, cd to the directory and:

```
pip install -e .
```

Once installed, the *development* flask server might be started with the command:

```
iamodeler
```

For real deployment, gunicorn might be used instead:

```
pip install gunicorn
unicorn --workers 4 --bind 0.0.0.0:5000 --timeout 600 iamodeler.server:app
```

To use a celery queue system (see configuration below), a celery broker
like [RabbitMQ](https://www.rabbitmq.com/download.html) must also be installed.

With RabbitMQ installed and running, start the queue system by running:

```
celery -A iamodeler.tasks.celery worker
```

Note the gunicorn timeout parameter does not affect the celery queues.

In **Windows**, the default celery pool might not work. You might try to add `--pool=eventlet` to run it.

[//]: # (For instructions to set up a **cluster** to scale the number of workers, see [this file]&#40;docs/cluster.md&#41;.)

#### Configuration

Configuration is done with environment variables.

| Variable                 | Description                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------- |
| IAMODELER\_STORE         | Path to the local storage of the models. Defaults to a temporal directory.                                    |
| IAMODELER\_CELERY        | If set and not empty, use a local Celery queue system.                                                        |
| IAMODELER\_CELERY\_BROKER| Address of the Celery broker.                                                                                 |
| IAMODELER\_AUTH          | Authentication token for the server. Client request must set X-IAMODELER-AUTH to this token in their headers. |
| IAMODELER\_LOG           | A path to a yaml logging configuration file. Defaults to logging.yaml                                         |

The paths are *relative to the CWD*, provide full paths when needed.

Pro-tip: A .env file can be used installing the python-dotenv package.

An example of logging configuration file is provided in the root of the repo.


### BPMN Items suggestions
This AI-based approach provides recommendations during service composition. The suggestions are based on a selected service composition approach by (BPMN-based work-flow) data representation, existing/history BPMN work-flows, and provided service specification information.

#### Usage
This sub-module receives the information of the last selected node in the target BPMN diagram. This information is in JSON format, which can include unique node id and other node metadata such as name or user_id. Afterwards, the query compositor merges it with the incomplete BPMN file , developers are working with.

```json
{
  "dle": {
    "header": "bpmn suggestion",
    "state": "query",
    "previous node": [
      {
        "id": "_13BAF867-3CA8-4C6F-85C6-D3FD748D07D2"
      },
      {
        "name": "UserFound?"
      }
    ]
  }
}

```

The module performs four main steps on the received JSON request, which are: 1) Query Compositor 2) Current BPMN Extractor 3) BPMN semantic identifier 4) Numerical vector transformer and finally suggesting nexrtBPMN node which will be in JSON response format: 

```json

{
  "dle": {
    "header": "bpmn suggestion",
    "state": "true",
    "previous node": [
      {
        "id": "_13BAF867-3CA8-4C6F-85C6-D3FD748D07D2"
      },
      {
        "name": "UserFound?"
      }
    ],
    "suggestionnode": [
      {
        "id": "_E5D17755-D671-43ED-BD7D-F6538933069C"
      },
      {
        "name": "AuditUser"
      }
    ]
  }
}
{
  "dle": {
    "header": "bpmnsuggestion",
    "state": "false",
    "previousnode": [
      {
        "id": "_13BAF867-3CA8-4C6F-85C6-D3FD748D07D2"
      },
      {
        "name": "UserFound?"
      }
    ]
  }
}
```




[//]: # (### Code repository suggestions model)

[//]: # (This wizard is responsible for generating suggestions to the user to facilitate commits to the git repository. Receiving information from the Context Handling component, and with the help of the DLE, it will determine the best time to commit to the git repository. )

[//]: # (### Deployment environment suggestions model )

[//]: # (This subcomponent is responsible for generating suggestions for the sizing of the deployment environment. )

