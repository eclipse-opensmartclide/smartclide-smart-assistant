## DLE  Service Classification AI Model

Smartclide provides an environment to support the development of service-oriented softwares. The goal of this service classification is to classify the same web services based on their functionality which can be helpful in later stages such as service composition. 


This service classification takes advantage of text classification trends and deep learning methods. After processing the collected dataset, techniques and algorithms have been applied to solve the service classification problem. The real-world service classification in industry need to consider performance and minimum resource usage. Therefore, thiis subcomponent have provided two models, first model was trained by keyword-based featuer extraction and SVM  which is set as deafualt model. Secound model use BERT which is is a transformer that Google has introduced. The significant feature of BERT is using BiLSTM, which is the most promising model for learning long-term dependencies. 


The following parameter is mandatory to route the requests to this implemented Model:

1.Service Name: A string including web service name.

2.Service Description: a service description in string format.



## DLE   Code  Autocomplete AI Model

SmartCLIDE tries to use a language modeling pipeline to generate one-line codes based on existing internal and external source codes. Language modeling is an active area in NLP, which uses different neural network architectures and Transformers.The proposed pipeline has used GPT2, which is lighter than GPT3. In terms of training data, The best scenario is to learn from the internal SmartCLIDE codes project. However, deep learning algorithms train well with large data. Therefore, the proposed model needs to use external source codes. This model is under develope.



## Install

Both AI models are using same lib, therefore service classification and autocomplete model have packaged together, The list of thirdparty library are listed on requirments.txt file however the two main used library are:

* [HuggingFace](https://huggingface.co/)
* [scikit-learn](https://scikit-learn.org)


Service classification and Autocomplete models can be install with running following command:
```
python3 -m pip install . --upgrade
```
 

These models need a  gateway between the trained models and user interfaces. [smartclide-dle](https://github.com/eclipse-opensmartclide/smartclide-smart-assistant/tree/main/smartclide-dle) component provides API for using these models.


The minimum requirment for using this package is 43GB storage and 16GB RAM requirment. the reasen is the used packages after extraction is going to exceed more spaces.

However, if the package is installing on AWS EC2, the configuration in TMPDIR is necessary [more info](https://stackoverflow.com/questions/55103162/could-not-install-packages-due-to-an-environmenterror-errno-28-no-space-left)


### Usage

In SmartCLIDE, many tasks require to run in the background independently of the user interface (UI). AI Models is one of these tasks that need to serve requests in real-time and return results. Consequently, loading the AI model can be time-consuming due to late response. A strategy such as using singleton classes for loading the models can help minimize the application UI load, improve availability, and reduce interactive response times.
You can find the example code which describes how to import and use the package. The package is located in the example folder.

