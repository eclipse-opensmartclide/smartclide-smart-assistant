#!/usr/bin/python3
# Eclipse Public License 2.0

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
import smartclide_service_classification_autocomplete
from sklearn.feature_extraction.text import TfidfVectorizer

# # toknize
import nltk

_PATH_dataset_ = './data'


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
    method = ['Default', 'BSVM','Advanced']


    
    def __init__(self, useSavedModel=True,
                 targetCLMName='Description',
                 classCLMName='Category',
                 defaultServiceDataset='',
                 defaultServiceTrainDataset='',
                 defaultServiceTestDataset=''
                 ):
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

    def loadData(self, path=''):
        if (path == ''):
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
        print("1111111preprocesss")
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
        print("--------")
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


    def loadSavedModel(self, modelName):
        path = self.getTrainedModelsDirectory()
        if modelName == 'BOWML':
            isfile = os.path.exists(os.path.join(path, 'service_classification_bow_svc.pkl'))
            if isfile:
                self.model = pickle.load(open(path + 'service_classification_bow_svc.pkl', 'rb'))
                return True
            return False
        else:
            return False
        return False
    
    
    
    
