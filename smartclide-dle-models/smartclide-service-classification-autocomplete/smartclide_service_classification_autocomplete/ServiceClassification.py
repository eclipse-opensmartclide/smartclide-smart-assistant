#!/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.
import os

path = os.getcwd()
_PATH_ROOT_ = path + '/smartclide_service_classification_autocomplete/'
from .AIPipelineConfiguration import *
from .PreProcessTextData import *

_PATH_dataset_ = './data'

import sys

sys.path.insert(1, _PATH_dataset_)

import smartclide_service_classification_autocomplete
import io
import os
import torch
# import ktrain
import pickle
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.svm import LinearSVC
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# # toknize
import nltk


class ServiceClassificationModel(AIPipelineConfiguration):
    X = []
    y = []
    model = None
    learner = None
    predictor = None
    lbMapped = None
    testDF = None
    trainDF = None
    tokenizer = None
    testLabels = None
    trainLabels = None
    testFeatures = None
    trainFeatures = None
    maxLenEmbedding = 80
    rawCodeList = [];
    method = ['Default', 'BSVM']
    df = pd.DataFrame();
    tfidf = TfidfVectorizer()

    def __init__(self, useSavedModel=True,
                 targetCLMName='Description',
                 classCLMName='Category',
                 defaultServiceDataset='',
                 defaultServiceTrainDataset='',
                 defaultServiceTestDataset=''
                 ):
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

    def loadData(self, path=''):
        if (path == ''):
            #             currentPath=self.getCurrentDirectory()
            currentPath = os.path.abspath(os.path.dirname(__file__))
            self.df = pd.read_csv(os.path.join(currentPath, self.defaultServiceDataset))
            self.trainDF = pd.read_csv(os.path.join(currentPath, self.defaultServiceTrainDataset))
            self.testDF = pd.read_csv(os.path.join(currentPath, self.defaultServiceTestDataset))
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        return self.df

    def getCurrentDirectory(self):
        path = os.getcwd()
        return (path)

    def getParentDirectory(self):
        path = os.getcwd()
        return (os.path.abspath(os.path.join(path, os.pardir)))

    def getTrainedModelsDirectory(self):
        from smartclide_service_classification_autocomplete import getPackagePath
        packageRootPath = getPackagePath()
        # packageRootPath = os.getcwd()
        return (packageRootPath + "/trained_models/")

    def getTrainedModel(self, modelName):
        TrainrdModelPath = self.getTrainedModelsDirectory()
        path = TrainrdModelPath + '/' + modelName
        return path

    def IsTrainedModelExist(self, modelName):
        TrainrdModelPath = self.getTrainedModelsDirectory()
        isfile = os.path.exists(TrainrdModelPath + '/' + modelName)
        return isfile

    def removeHTMLTags(self, clmName):
        textPreProcessOBJ = TextDataPreProcess()
        self.df[clmName] = self.df[clmName].astype(str).apply(textPreProcessOBJ.removeHTMLTags)
        return (self.df)

    def removeNullrows(self, clmName):
        self.df = self.df[pd.notnull(self.df[clmName])]
        return (self.df)

    def cleanPunc(self, clmName):
        textPreProcessOBJ = TextDataPreProcess()
        self.df[clmName] = self.df[clmName].astype(str).apply(textPreProcessOBJ.cleanPunc)
        return (self.df)

    def removeStopWords(self, clmName):
        textPreProcessOBJ = TextDataPreProcess()
        self.df[clmName] = self.df[clmName].apply(textPreProcessOBJ.removeStopWords)
        return (self.df)

    def removeDuplicates(self, clmName):
        textPreProcessOBJ = TextDataPreProcess()
        self.df[clmName] = self.df[clmName].astype(str).apply(textPreProcessOBJ.uniqueList)
        return (self.df)

    def tokenize(self, clmName):
        tokenizer = RegexpTokenizer(r'\w+')
        self.df[clmName] = self.df[clmName].apply(lambda x: tokenizer.tokenize(x))
        # self.df[clmName] = self.df[clmName].apply(lambda x:tokenizer.tokenize(x.lower()))
        return (self.df)

    def wordLemmatizer(self, clmName=''):
        if clmName == '':
            clmName = 'description'
        textPreProcessOBJ = TextDataPreProcess()
        self.df[clmName] = self.df[clmName].apply(textPreProcessOBJ.wordLemmatizer)
        return (self.df)

    def getCommonWord(self, clmName, n=50):
        from sklearn.feature_extraction.text import CountVectorizer
        textPreProcessOBJ = TextDataPreProcess()
        countVectorizerOBJ = CountVectorizer()
        countData = countVectorizerOBJ.fit_transform(self.df[clmName])
        commonWords = textPreProcessOBJ.getCommonWords(countData, countVectorizerOBJ, n)
        return commonWords

    def removeCommonWords(self, clmName, n=50):
        commonwords = self.getCommonWord(clmName, n)
        pat = r'\b(?:{})\b'.format('|'.join(commonwords))
        self.df[clmName] = self.df[clmName].str.replace(pat, '', regex=True)

    def preprocessServicesData(self, clmName, actions=[]):
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

    def lowercase(self, clmName):
        self.df[clmName] = self.df[clmName].str.lower()

    def TrainLinerSVCModel(self):
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
        result=pickle.dump(self.model,
                    open(os.path.join(packagePath + "/trained_models/", "service_classification_bow_svc.pkl"), 'wb'))

    def predictBOWML(self, x):
        textPreProcessOBJ = TextDataPreProcess()
        x = x.lower()
        x = textPreProcessOBJ.removeStopWords(x)
        x = textPreProcessOBJ.cleanPunc(x)
        if (len(x) < 2):
            return False

        if (self.IsTrainedModelExist('service_classification_bow_svc.pkl')):
            try:
                self.loadSavedModel("BOWML")
            except ValueError:
                print("Could not load model.")
        else:
            self.loadData()
            self.TrainLinerSVCModel()

        pred = self.model.predict([x])
        return pred

    def VectorizeDataset(self, maxLen=60):
        import transformers as ppb
        # self.df=self.df[:15]
        self.preprocessServicesData(self.targetCLMName)
        model_class, tokenizer_class, pretrained_weights = (
            ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        tokenized = self.df[self.targetCLMName].apply(
            (lambda x: tokenizer.encode(x, max_length=60, add_special_tokens=True)))
        padded = np.array([i + [0] * (maxLen - len(i)) for i in tokenized.values])
        input_ids = torch.tensor(padded)
        return input_ids

    def getCurrentDirectory(self):
        path = os.getcwd()
        return (path)

    def getParentDirectory(self):
        path = os.getcwd()
        return (os.path.abspath(os.path.join(path, os.pardir)))

    def getTrainedModelsDirectory(self):
        from smartclide_service_classification_autocomplete import getPackagePath
        packageRootPath = getPackagePath()
        return (packageRootPath + "/trained_models/")

    def getTrainedModel(self, modelName):
        TrainrdModelPath = self.getTrainedModelsDirectory()
        path = TrainrdModelPath + '/' + modelName
        return path

    def IsTrainedModelExist(self, modelName):
        TrainrdModelPath = self.getTrainedModelsDirectory()
        isfile = os.path.exists(TrainrdModelPath + '/' + modelName)
        return isfile

    def ContextualEmbedingTrain(self, maxLen=20):
        self.loadData()
        X_train = self.trainDF[self.targetCLMName].values
        y_train = self.trainDF["Category_lable"].values
        trainFeatures = self.getX_featuers(self.trainDF)
        trainLabels = y_train
        from sklearn.svm import LinearSVC
        self.model = LinearSVC(class_weight='balanced', random_state=777)
        print("modeling")
        self.model.fit(trainFeatures, trainLabels)
        try:
            import pickle
            packagePath = os.path.abspath(os.path.dirname(__file__))
            pickle.dump(self.model,
                        open(os.path.join(packagePath + "/trained_models/", "service_classification_bert_svc.pkl"),
                             'wb'))
        except OSError as err:
            print("OS error: {0}".format(err))
        except ValueError:
            print("Could not dump model.")
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")

    def getX_featuers(self, df):
        import torch
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig, \
            DistilBertModel
        import transformers as ppb
        dftemp = pd.DataFrame()
        dftemp['Description'] = df['Description']
        # For DistilBERT:
        model_class, tokenizer_class, pretrained_weights = (
            ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        # Load pretrained model/tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        tokenized = dftemp['Description'].apply((lambda x: self.tokenizer.encode(x, max_length=self.maxLenEmbedding,
                                                                                 add_special_tokens=True,
                                                                                 truncation=True)))
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)
        padded = np.array([i + [0] * (self.maxLenEmbedding - len(i)) for i in tokenized.values])
        attention_mask = np.where(padded != 0, 1, 0)
        attention_mask.shape
        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)
        with torch.no_grad():
            last_hidden_states = model(input_ids)
        features = last_hidden_states[0][:, 0, :].numpy()
        return features

    def predictBSVMModel(self, x):
        import torch
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig, \
            DistilBertModel
        import transformers as ppb
        if (len(x) < 2):
            return False
        df_input = pd.DataFrame();
        df_input['Description'] = ""
        df_input = pd.DataFrame(columns=['Description'])
        df_input.loc[0] = [x]
        #         if self.tokenizer==None:
        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel,
                                                            ppb.DistilBertTokenizer,
                                                            'distilbert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)

        tokenized = df_input['Description'].apply(
            (lambda x: self.tokenizer.encode(x, max_length=self.maxLenEmbedding, add_special_tokens=True)))
        padded = np.array([i + [0] * (self.maxLenEmbedding - len(i)) for i in tokenized.values])
        input_x_ids = torch.tensor(padded)
        with torch.no_grad():
            last_hidden_states = model(input_x_ids)
        features = last_hidden_states[0][:, 0, :].numpy()

        if (self.useSavedModel and self.IsTrainedModelExist('service_classification_bert_svc.pkl')):
            try:
                self.loadSavedModel("BSVM")
            except ValueError:
                print("Could not load model.")
        else:
            self.ContextualEmbedingTrain()
        pred = self.model.predict(features)
        return pred

    def loadSavedModel(self, modelName):
        path = self.getTrainedModelsDirectory()
        if modelName == 'BOWML':
            isfile = os.path.exists(os.path.join(path, 'service_classification_bow_svc.pkl'))
            if isfile:
                self.model = pickle.load(open(path + 'service_classification_bow_svc.pkl', 'rb'))
                return True
            return False
        elif modelName == 'BSVM':
            isfile = os.path.exists(os.path.join(path, 'service_classification_bert_svc.pkl'))
            if isfile:
                self.model = pickle.load(open(path + 'service_classification_bert_svc.pkl', 'rb'))
                return True
            return False
        else:
            return False
        return False
