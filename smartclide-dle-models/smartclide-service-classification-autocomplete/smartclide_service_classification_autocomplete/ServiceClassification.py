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
    rawCodeList = [];
    method =[ 'Default' ,'BSVM']
    df = pd.DataFrame();
    tfidf = TfidfVectorizer()

    def __init__(self, useSavedModel=True, targetCLMName='Description', classCLMName='Category', defaultDatasetName=''):
        self.targetCLMName = targetCLMName
        self.classCLMName = classCLMName
        self.useSavedModel = useSavedModel
        self.defaultDatasetName = defaultDatasetName
        if (defaultDatasetName == ''):
            self.defaultDatasetName = self.defaultServiceTrainDataset
        if not (self.useSavedModel):
            self.loadData()

    def loadData(self, path=''):
        if (path == ''):
            here = os.path.abspath(os.path.dirname(__file__))
        self.df = pd.read_csv(os.path.join(here, "services_processed_mapped.csv"))
        return (self.df)

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

    def loadSavedModel(self, modelName):
        path = os.path.abspath(os.path.dirname(__file__))
        print(path)
        path2=path +'/'+ self.defaultTrainedModelPath + 'Model_BOWML/Model_CLFLinearSVC.pk'
        print(path2)
        if modelName == 'BOWML':
            isfile = os.path.exists(
                os.path.join(path +'/'+ self.defaultTrainedModelPath + 'Model_BOWML/',
                             'Model_CLFLinearSVC.pk'))
            if isfile:
                self.model = pickle.load(open(path +'/'+ self.defaultTrainedModelPath + 'Model_BOWML/Model_CLFLinearSVC.pk', 'rb'))
                self.tfidf = pickle.load(open(path +'/'+ self.defaultTrainedModelPath + 'Model_BOWML/Vector_tfidf.pkl','rb'))
                return True
            return False
        elif modelName == 'BSVM':
            isfile = os.path.exists(
                os.path.join(os.getcwd(), _PATH_ROOT_ + self.defaultTrainedModelPath + 'Model_BSVM/', 'Model_SVC.pk'))
            if isfile:
                self.model = pickle.load(
                    open(_PATH_ROOT_ + self.defaultTrainedModelPath + 'Model_BSVM/Model_SVC.pk', 'rb'))
                return True
            return False
        else:
            return False
        return False


    def TrainLinerSVCModel(self):
        self.preprocessServicesData(self.targetCLMName)
        self.tfidf = TfidfVectorizer()
        d = self.df[self.targetCLMName]
        self.X = self.tfidf.fit_transform(d)
        self.y = self.df[self.classCLMName]
        self.model = LinearSVC(class_weight='balanced', random_state=777)
        self.model.fit(self.X, self.y)
        print(self.classCLMName)
        # save models
        here = os.path.abspath(os.path.dirname(__file__))
        pickle.dump(self.model,
                    open(os.path.join(here + "/trained_models/Model_BOWML/", "Model_CLFLinearSVC.pk"), 'wb'))
        pickle.dump(self.tfidf, open(os.path.join(here + "/trained_models/Model_BOWML/", 'Vector_tfidf.pkl'), 'wb'))

    def predictBOWML(self, x):
        textPreProcessOBJ = TextDataPreProcess()
        x = x.lower()
        x = textPreProcessOBJ.removeStopWords(x)
        x = textPreProcessOBJ.cleanPunc(x)
        if (len(x) < 2):
            return False
        here = os.path.abspath(os.path.dirname(__file__))
        isfile = os.path.exists(os.path.join(here + self.defaultTrainedModelPath + 'Model_BOWML/','Model_CLFLinearSVC.pk'))
        print("00000000000000000000000000000000000000")
        print(isfile)
        print(os.path.join(here + self.defaultTrainedModelPath + 'Model_BOWML/','Model_CLFLinearSVC.pk'))
        # load saved model
        if self.useSavedModel == True and isfile:
            self.loadSavedModel("BOWML")
        else:
            self.loadData()
            self.TrainLinerSVCModel()
        x = self.tfidf.transform([x])
        x = x.toarray()
        pred = self.model.predict(x)
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

    def ContextualEmbedingTrain(self, maxLen=60):
        import transformers as ppb
        from sklearn.model_selection import train_test_split
        model_class, tokenizer_class, pretrained_weights = (
            ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)

        input_ids = self.VectorizeDataset(maxLen)
        with torch.no_grad():
            last_hidden_states = model(input_ids)
        features = last_hidden_states[0][:, 0, :].numpy()
        labels = self.df[self.classCLMName]
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
        from sklearn.svm import LinearSVC
        self.model = LinearSVC(class_weight='balanced', random_state=777)
        self.model.fit(train_features, train_labels)
        pickle.dump(self.model, open(_PATH_ROOT_ + self.defaultTrainedModelPath + 'Model_BSVM/Model_SVC.pk', 'wb'))

    def predictBSVMModel(self, x, maxLen=60):
        import torch
        import transformers as ppb
        model_class, tokenizer_class, pretrained_weights = (
            ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        textPreProcessOBJ = TextDataPreProcess()
        x = x.lower()
        x = textPreProcessOBJ.removeStopWords(x)
        x = textPreProcessOBJ.cleanPunc(x)
        if (len(x) < 2):
            return False
        df_input = pd.DataFrame(columns=[self.targetCLMName])
        df_input.loc[0] = [x]
        tokenized = df_input['Description'].apply(
            (lambda x: tokenizer.encode(x, maxLength=60, add_special_tokens=True)))
        padded = np.array([i + [0] * (maxLen - len(i)) for i in tokenized.values])
        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        input_x_ids = torch.tensor(padded)
        with torch.no_grad():
            last_hidden_states = model(input_x_ids)
        features = last_hidden_states[0][:, 0, :].numpy()
        isfile = os.path.exists(
            os.path.join(os.getcwd(), _PATH_ROOT_ + self.defaultTrainedModelPath + 'Model_BSVM/', 'Model_SVC.pk'))
        # load saved model
        if self.useSavedModel == True and isfile:
            self.loadSavedModel("BSVM")
        else:
            self.ContextualEmbedingTrain()
        pred = self.model.predict(features)
        return pred
