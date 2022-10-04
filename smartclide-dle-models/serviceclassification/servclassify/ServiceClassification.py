#!/usr/bin/python3
# Eclipse Public License 2.0

"""
import required lib
"""

import os
import sys
import io
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.svm import LinearSVC
from .PreProcessTextData import *
from .AIPipelineConfiguration import *
from nltk.tokenize import RegexpTokenizer
import servclassify
from sklearn.feature_extraction.text import TfidfVectorizer


class ServiceClassificationModel(AIPipelineConfiguration):
    """
    This class is part of backend section of SmartCLIDE which include  training and prediction  using
    text classification models including ML and DL approches
    """

    X = []
    y = []
    device = 'cpu'
    model = None
    learner = None
    predictor = None
    lbMapped = None
    testDF = None
    max_len = 128
    trainDF = None
    tokenizer = None
    testLabels = None
    trainLabels = None
    testFeatures = None
    trainFeatures = None
    maxLenEmbedding = 80
    rawCodeList = [];
    classifier_config =None
    tokenizer_class = None
    classifier_model = None
    transformer_model_name = 'bert-base-uncased'
    trained_model_name = f'bert_service_classifier'
    trained_model_folder = 'bert_service_classifier/'
    svm_model_name = 'service_classification_svm_.pkl'
    method = ['Default', 'BSVM', 'Advanced']

    def __init__(self, useSavedModel=True,
                 targetCLMName='text',
                 classCLMName='category',
                 defaultServiceDataset='',
                 defaultServiceTrainDataset='',
                 defaultServiceTestDataset=''
                 ):
        """
        :param useSavedModel:               bool param specify using existing model .pkl or retrain
        :param targetCLMName:               string param specifies target colmn
        :param classCLMName:                string param specifies class colmn
        :param defaultServiceDataset:       string param specifies default service dataset file
        :param defaultServiceTrainDataset:  string param specifies default service train dataset file
        :param defaultServiceTestDataset:   string param specifies default service test dataset file
        """
        df = pd.DataFrame();
        tfidf = TfidfVectorizer()
        self.targetCLMName = targetCLMName
        self.classCLMName = classCLMName
        self.useSavedModel = useSavedModel
        if not (defaultServiceTrainDataset == ''):
            self.defaultServiceTrainDataset = defaultServiceTrainDataset
        if not (defaultServiceTestDataset == ''):
            self.defaultServiceTestDataset = defaultServiceTestDataset
        if (defaultServiceDataset == ''):
            self.defaultServiceDataset = self.defaultServiceDataset
        if not (self.useSavedModel):
            self.loadData()
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.textPreProcessOBJ = TextDataPreProcess()

    def loadData(self, path=''):
        """
        :param path: path text that the model uses as a deafualt target dataset
        :return: A dataframe object
        """
        if (path == ''):
            currentPath = os.path.abspath(os.path.dirname(__file__))
            self.df = pd.read_csv(
                os.path.join(currentPath, (self.defaultDatasetsFolder + '/' + self.defaultServiceDataset)))
            self.trainDF = pd.read_csv(
                os.path.join(currentPath, (self.defaultDatasetsFolder + '/' + self.defaultServiceTrainDataset)))
            self.testDF = pd.read_csv(
                os.path.join(currentPath, (self.defaultDatasetsFolder + '/' + self.defaultServiceTestDataset)))
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        return self.df

    def getCurrentDirectory(self):
        """
        returns current working directory of a process
        :return: string param specifies working directory
        """
        path = os.getcwd()
        return (path)

    def getParentDirectory(self):
        """
        returns current working directory parrent
        :return: string param specifies working directory parrent path
        """
        path = os.getcwd()
        return (os.path.abspath(os.path.join(path, os.pardir)))

    def getTrainedModelsDirectory(self):
        """
        returns default training models folder in order to load trained models files
        :return: string param specifies training models folder
        """
        from servclassify import getPackagePath
        packageRootPath = getPackagePath()
        return (packageRootPath + "/trained_models/")

    def getTrainedModel(self, modelName):
        """
        returns default training models path
        :return: string param specifies training models path
        """
        TrainrdModelPath = self.getTrainedModelsDirectory()
        path = TrainrdModelPath + '/' + modelName
        return path

    def IsTrainedModelExist(self, modelName):
        """
        Ensure trained model file is exist
        :return: bool param
        """
        TrainrdModelPath = self.getTrainedModelsDirectory()
        isfile = os.path.exists(TrainrdModelPath + '/' + modelName)
        return isfile

    def removeHTMLTags(self, clmName):
        """
        Preprocess function inorder to clean HTML tags from service metadata
        :param clmName: string param specifies target colmn
        """
        textPreProcessOBJ = TextDataPreProcess()
        self.df[clmName] = self.df[clmName].astype(str).apply(textPreProcessOBJ.removeHTMLTags)
        return (self.df)

    def removeNullrows(self, clmName):
        """
        Preprocess function inorder to clean Null value from service metadata
        :param clmName: string param specifies target colmn
        """
        self.df = self.df[pd.notnull(self.df[clmName])]
        return (self.df)

    def cleanPunc(self, clmName):
        """
        Preprocess function inorder to clean spechial character from service metadata
        :param clmName: string param specifies target colmn
        """
        textPreProcessOBJ = TextDataPreProcess()
        self.df[clmName] = self.df[clmName].astype(str).apply(textPreProcessOBJ.cleanPunc)
        return (self.df)

    def removeStopWords(self, clmName):
        """
        Preprocess function inorder to clean EN stop words from service metadata
        :param clmName: string param specifies target colmn
        """
        textPreProcessOBJ = TextDataPreProcess()
        self.df[clmName] = self.df[clmName].apply(textPreProcessOBJ.removeStopWords)
        return (self.df)

    def removeDuplicates(self, clmName):
        """
        Preprocess function inorder to remove Duplicate gathered service data
        :param clmName: string param specifies target colmn
        """
        textPreProcessOBJ = TextDataPreProcess()
        self.df[clmName] = self.df[clmName].astype(str).apply(textPreProcessOBJ.uniqueList)
        return (self.df)

    def tokenize(self, clmName):
        """
        Preprocess function inorder to tokenizer service metadata text
        :param clmName: string param specifies target colmn
        """
        tokenizer = RegexpTokenizer(r'\w+')
        self.df[clmName] = self.df[clmName].apply(lambda x: tokenizer.tokenize(x))
        return (self.df)

    def wordLemmatizer(self, clmName=''):
        """
        Preprocess function inorder to ldmmatization, Lemmatization is the process of grouping together
        the different inflected forms of a word
        :param clmName: string param specifies target colmn
        """
        if clmName == '':
            clmName = 'text'
        self.df[clmName] = self.df[clmName].apply(self.textPreProcessOBJ.wordLemmatizer)
        return (self.df)

    def getCommonWord(self, clmName, n=50):
        """
        Preprocess function inorder to remove common and most frequent words in gathered service data
        :param clmName: string param specifies target colmn
        """
        from sklearn.feature_extraction.text import CountVectorizer
        countVectorizerOBJ = CountVectorizer()
        countData = countVectorizerOBJ.fit_transform(self.df[clmName])
        commonWords = self.textPreProcessOBJ.getCommonWords(countData, countVectorizerOBJ, n)
        return commonWords

    def removeCommonWords(self, clmName, n=50):
        """
        Preprocess function inorder to remove common on data frame
        :param clmName: string param specifies target colmn
        """
        commonwords = self.getCommonWord(clmName, n)
        pat = r'\b(?:{})\b'.format('|'.join(commonwords))
        self.df[clmName] = self.df[clmName].str.replace(pat, '', regex=True)

    def lowercase(self, clmName):
        """
        Preprocess function inorder to lowercase data frame
        :param clmName: string param specifies target colmn
        """
        self.df[clmName] = self.df[clmName].str.lower()

    def getTransformerModelName(self):
        """
        Function inorder to return deafult transformer model
        :param clmName: string param specifies transformer
        """
        return (self.transformer_model_name)

    def getTrainedModelName(self):
        """
        Function inorder to return deafult transformer model
        :param clmName: string param specifies trained model name
        """
        return (self.trained_model_name)

    def ValidateTextInput(self, txtInput):
        """
        Input validation for user text input including service name and service Description
        :param txtInput: The value the user inputs for the "text" parameter
        """
        if not isinstance(text, str):
            raise ValueError("The text input must be a string")
        if not text:
            raise ValueError("The text input must have at least one character")



    def preprocessServicesData(self, clmName, actions=[]):
        """
        Preprocess gathered service data for SVM model
        :return: preprocessed df
        """
        if ("lower" in actions):
            self.lowercase(clmName)
        self.removeNullrows(clmName)
        self.removeStopWords(clmName)
        self.tokenize(clmName)
        self.wordLemmatizer(clmName)
        self.cleanPunc(clmName)
        self.removeHTMLTags(clmName)
        self.getCommonWord(clmName, 60)
        self.removeCommonWords(clmName)

        return self.df

    def TrainLinerSVCModel(self):
        """
        Train preprocessed data using SVM
        """
        self.preprocessServicesData(self.targetCLMName)
        self.X = self.trainDF[self.targetCLMName]
        self.y = self.trainDF[self.classCLMName]
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer
        self.model = Pipeline([('vect', CountVectorizer()),
                               ('tfidf', TfidfTransformer()),
                               ('clf', LinearSVC(class_weight='balanced', random_state=777)),
                               ])
        self.model.fit(self.X, self.y)
        # save models
        packagePath = os.path.abspath(os.path.dirname(__file__))
        result = pickle.dump(self.model,
                             open(os.path.join(packagePath + "/trained_models/", self.svm_model_name), 'wb'))

    def predictBOWML(self, x):
        """
        Predict service class based on user input text and ML trained model
        :return: preprocessed df
        """
        x = x.lower()
        x = self.textPreProcessOBJ.removeStopWords(x)
        x = self.textPreProcessOBJ.cleanPunc(x)
        if (len(x) < 2):
            return False

        if (self.IsTrainedModelExist(self.svm_model_name)):
            try:
                self.loadSavedModel("BOWML")
            except ValueError:
                print("Could not load model.")
        else:
            self.loadData()
            self.TrainLinerSVCModel()

        pred = self.model.predict([x])
        f_class_ = pred[0]
        s_class_ = ''
        classes = [f_class_, s_class_]
        return classes

    def get_prediction(self, text, k=2):
        """
        Predict service class based on user input text and DL trained model
        :return: string param specifies service class
        """
        # prepare our text into tokenized sequence
        inputs = self.tokenizer_class(text, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        # perform inference to our model
        outputs = self.classifier_model(**inputs)
        probs = outputs[0].softmax(1)
        top_tensors = torch.topk(probs.flatten(), 2).indices
        # get id and lable from model config
        id2label =  self.classifier_config.id2label
        top_cat_id = []
        for i in range(0, k ):
            top_cat_id.append(top_tensors[i].item())
        top_cat_lable = []
        for i in range(0, k ):
            class_ = id2label[(top_cat_id[i])]
            top_cat_lable.append(class_)

        return top_cat_lable

    import sys

    def install(self, package):
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    def loadSavedModel(self, modelName):
        """
        Load Saved ML models
        :return: bool
        """
        import os, sys
        sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
        path = self.getTrainedModelsDirectory()
        if modelName == 'BOWML':
            isfile = os.path.exists(os.path.join(path, self.svm_model_name))
            if isfile:
                self.model = pickle.load(open(path + self.svm_model_name, 'rb'))
                return True
            return False
        else:
            return False
        return False

    # def loadTrainedClassifier(self):
    #     """
    #     Load conceptual embedding trained web service classifier
    #     :return: string param specifies service class
    #     """
    #     import pickle
    #     from transformers import BertForSequenceClassification, AdamW, BertConfig
    #     from transformers import BertTokenizer, BertForMaskedLM
    #
    #     trained_models_folder = self.getTrainedModelsDirectory()
    #     trained_models_files = self.getTrainedModelName()
    #     model_path = os.path.join(trained_models_folder, trained_models_files)
    #     if not (os.path.exists(model_path)):
    #         print("+" * 40)
    #         print("No training DL service classifier file is exist")
    #         print("+" * 40)
    #         return False
    #
    #     try:
    #
    #         self.classifier_model = BertForSequenceClassification.from_pretrained(model_path)
    #         self.tokenizer_class = BertTokenizer.from_pretrained(model_path)
    #
    #         print("+" * 40)
    #         print("Trained DL service classifier  is loaded on " + self.device)
    #         print("+" * 40)
    #
    #     except FileNotFoundError:
    #         print("+" * 40)
    #         print("No training DL service classifier file is exist")
    #         print("+" * 40)
    #
    #     return (self.classifier_model)
    def loadTrainedClassifier(self):
        """
        Load trained web service classifier
        :return: trained model obj
        """
        import pickle
        from transformers import BertForSequenceClassification, AdamW, BertConfig
        from transformers import BertTokenizer, BertForMaskedLM,BertConfig
        try:
            # trained_models_folder = self.getTrainedModelsDirectory()
            # trained_models_files = self.getTrainedModelName()
            # model_path = os.path.join(trained_models_folder, trained_models_files)

            # model_path = "zakieh/servclassification"  # Download from model hub
            # self.classifier_model = BertForSequenceClassification.from_pretrained(model_path)
            # self.tokenizer_class = BertTokenizer.from_pretrained(model_path)
            model_hub = "zakieh/serv_classification"
            self.classifier_model = BertForSequenceClassification.from_pretrained(model_hub,force_download=True)
            self.tokenizer_class = BertTokenizer.from_pretrained(model_hub,force_download=True)
            self.classifier_config= BertConfig.from_pretrained(model_hub,force_download=True)

            print("+" * 40)
            print("Trained model is loaded on " + self.device)
            print("+" * 40)

        except FileNotFoundError:
            print("+" * 40)
            print("No training file is exist GPT2-2-2")
            print("+" * 40)

        return (self.classifier_model)