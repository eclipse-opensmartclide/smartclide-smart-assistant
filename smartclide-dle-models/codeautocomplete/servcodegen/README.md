# Intro

Smartclide provides an environment to support the development of service-oriented softwares. The goal of this service code autocompletion is to classify the same web services based on their functionality which can be helpful in later stages such as service composition. 

- [Requirements](#requirements)
- [Quick Installation](#quick-installation)
- [DLE Service Classification AI Model](##DLE-Service-Classification-AI-Model)
- [Usage](#usage)
  - [Simple Usage](##Simple-Usage)
  - [Singleton classes Usage](##singleton-classes-Usage)
- [JSON Output](#json-output)



# Requirements

The list of thirdparty library are listed on requirments.txt file however the two main used library and requirments are:
- Python 3.7+
- [HuggingFace](https://huggingface.co/)
- [scikit-learn](https://scikit-learn.org)
- System Requirements: CPU: 2cv RAM: 8 GB

Note: The minimum requirment for using this package is 43GB storage. the reasen is during package installation, it use temp storage and packages like torch during installation, exceed more spaces.

However, if the package is installing on AWS EC2, the configuration in TMPDIR is necessary [more info](https://stackoverflow.com/questions/55103162/could-not-install-packages-due-to-an-environmenterror-errno-28-no-space-left)

# DLE Service Classification AI Model
SmartCLIDE tries to use a language modeling pipeline to generate one-line codes based on existing internal and external source codes. Language modeling is an active area in NLP, which uses different neural network architectures and Transformers.The proposed pipeline has used GPT2/DistilGPT-2, which is lighter than GPT3. In terms of training data, The best scenario is to learn from the internal SmartCLIDE codes project. However, deep learning algorithms train well with large data. Therefore, the proposed model needs to use external source codes and benchmark dataset.

# Quick Installation
The trained models have been packaged using the Python Setuptools library. Therefore, this component need to install the related package by cloning the package, browsing to main directory, and executing “python3 -m pip install . --upgrade” command. 
```
git clone https://github.com/eclipse-researchlabs/smartclide-smart-assistant.git
cd smartclide-dle-models/codeautocomplete 
python3 -m pip install . --upgrade
```
Testing the module installation
```
python3 servcodegen/examples/generate_code.py
```


In SmartCLIDE platform, These models need a  gateway between the trained models and user interfaces. [smartclide-dle](https://github.com/eclipse-researchlabs/smartclide-smart-assistant/tree/main/smartclide-dle) component provides API for using these models.

#Usage
This library provide two a code generator which using language modeling and after installation the library can be use by importing the package.To have better performance, max_lenth specify max lenth line suggestion, recommended value is between 15-20, and max_sugges_line specify max line suggestion, recommended value is between 1-3.
## Simple Usage
```
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

## Singleton classes Usage
In SmartCLIDE, many tasks require to run in the background independently of the user interface (UI). AI Models is one of these tasks that need to serve requests in real-time and return results. Consequently, loading the AI model can be time-consuming due to late response. A strategy such as using singleton classes for loading the models can help minimize the application UI load, improve availability, and reduce interactive response times. 

```
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

You can find the example code which are in python script in the example folder.

