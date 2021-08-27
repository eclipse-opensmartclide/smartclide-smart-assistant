#!/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.

from configparser import ConfigParser

# Read config.ini file
config_object = ConfigParser()
config_object.read("config.ini")
# Get the _PATH_ROOT_
_PATH_ROOT_ = config_object["Path"]["root"]

import sys
sys.path.insert(1, _PATH_ROOT_ + '/DLE/utils')
from ImportingModules import *
from PreProcessTextData import *
from AIPipelineConfiguration import *


class ServiceClassification(AIPipelineConfiguration):
    X = []
    y = []
    model = None
    learner = None
    predictor = None
    tfidf = TfidfVectorizer()
    df = pd.DataFrame();
    rawCodeList = [];

    def __init__(self,defaultDatasetName,useSavedModel=True,targetCLMName='Description',classCLMName='Category'):
        self. targetCLMName = targetCLMName
        self.classCLMName = classCLMName
        self.useSavedModel = useSavedModel
        self.defaultDatasetName = defaultDatasetName

    def loadData(self, path=''):
        if (path == ''):
            path =_PATH_ROOT_ +  self.defaultDatasetsFolder+self.defaultDatasetName
        self.df = pd.read_csv(path)
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
        self.df[clmName] = self.df[clmName].str.replace(pat, '')

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

    def provideMLModelInput(self):
        self.tfidf = TfidfVectorizer()
        self.X = self.tfidf.fit_transform(self.df[self.targetCLMName])
        self.y = self.df['category']

    def TrainLinerSVCModel(self):
        self.preprocessServicesData(self.targetCLMName)
        self.tfidf = TfidfVectorizer()
        d = self.df[self.targetCLMName]
        self.X = self.tfidf.fit_transform(d)
        self.y = self.df[self.classCLMName]
        self.model = LinearSVC(class_weight='balanced', random_state=777)
        self.model.fit(self.X, self.y)
        # save models
        pickle.dump(self.model, open(_PATH_ROOT_ + self.defaultTrainedModelPath+'Model_BOWML/Model_CLFLinearSVC.pk', 'wb'))
        pickle.dump(self.tfidf, open(_PATH_ROOT_ + self.defaultTrainedModelPath+'Model_BOWML/Vector_tfidf.pkl', 'wb'))

    def predictBOWML(self, x):
        textPreProcessOBJ = TextDataPreProcess()
        x = x.lower()
        x = textPreProcessOBJ.removeStopWords(x)
        x = textPreProcessOBJ.cleanPunc(x)
        if (len(x) < 2):
            return False
        isfile = os.path.exists(
            os.path.join(os.getcwd(), _PATH_ROOT_ + self.defaultTrainedModelPath+'Model_BOWML/', 'Model_CLFLinearSVC.pk'))
        # load saved model
        if self.useSavedModel == True and isfile:
            self.loadSavedModel("BOWML")
        else:
            self.TrainLinerSVCModel()
        x = self.tfidf.transform([x])
        x = x.toarray()
        pred = self.model.predict(x)
        return pred

    def TrainFasttextModel(self):
        print("PreProcessing started....")
        start = time.time()
        df = self.preprocessServicesData(self.targetCLMName)
        df.to_csv('clustered_datatrain_preprocessed_bert.csv')
        PATH = "clustered_datatrain_preprocessed_bert.csv"
        NUM_WORDS = 1000000
        MAXLEN = 700
        end = time.time()
        print("PreProcessing completed....")
        print(end - start)
        print("Embeding started....")
        start = time.time()
        df = self.preprocessServicesData(self.targetCLMName)
        train, val, preproc = text.texts_from_csv(PATH, 'Description', label_columns='Category',
                                                  ngram_range=1, max_features=NUM_WORDS, maxlen=MAXLEN)
        end = time.time()
        print("Embeding completed....")
        print(end - start)
        print("Classifying started....")
        start = time.time()
        model = text.text_classifier('fasttext', train, preproc)
        self.learner = ktrain.get_learner(model, train, val)
        self.learner.autofit(0.01, self.epoch)
        end = time.time()
        print("Classifying completed....")
        print(end - start)
        print("Saving started....")
        start = time.time()
        self.predictor = ktrain.get_predictor(self.learner.model, preproc)
        self.predictor.save( _PATH_ROOT_ + self.defaultTrainedModelPath +'Model_FastText/')
        end = time.time()
        print("Saving completed....")
        print(end - start)
        self.learner.validate(class_names=predictor.get_classes())  # class_names must be string values
        return

    def predictFastTextModel(self, x):
        textPreProcessOBJ = TextDataPreProcess()
        x = x.lower()
        x = textPreProcessOBJ.removeStopWords(x)
        x = textPreProcessOBJ.cleanPunc(x)
        if (len(x) < 2):
            return False
        isfile = os.path.exists(
            os.path.join(os.getcwd(), _PATH_ROOT_ + self.defaultTrainedModelPath +'Model_FastText/', 'tf_model.h5'))
        # load saved model
        if self.useSavedModel == True and isfile:
            self.loadSavedModel("FastText")
        else:
            self.TrainFasttextModel()
        pred = self.predictor.predict([x])
        return pred

    def loadSavedModel(self, modelName):
        if modelName == 'BOWML':
            isfile = os.path.exists(
                os.path.join(os.getcwd(), _PATH_ROOT_ + self.defaultTrainedModelPath +'Model_BOWML/', 'Model_CLFLinearSVC.pk'))
            if isfile:
                self.model = pickle.load(
                    open(_PATH_ROOT_ + self.defaultTrainedModelPath +'Model_BOWML/Model_CLFLinearSVC.pk', 'rb'))
                self.tfidf = pickle.load(open(_PATH_ROOT_ + self.defaultTrainedModelPath +'Model_BOWML/Vector_tfidf.pkl', 'rb'))
                return True
            return False
        elif modelName == 'FastText':
            from ktrain import text
            self.predictor = ktrain.load_predictor(_PATH_ROOT_  + self.defaultTrainedModelPath +'Model_FastText/')
            print("FastText")
            print(self.predictor)
            return False
        else:
            return False
        return False
