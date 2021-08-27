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

import tensorflow
from tensorflow.keras  import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


class CodeGeneration(AIPipelineConfiguration):
    X=[]
    y=[]
    EncodedCodes=[];
    nGramcodeList=[]
    rawCodeList = [] ;
    maxLengthPadding=40
    predictCodeLength=2
    df = pd.DataFrame() ;
    modelLSTM=Sequential()
    coustomDatasetLine=2000
    paddedCodeNgramSequences=[]
    # defaultDatasetName = "top_poject_source_codes.csv"



    def __init__( self,defaultDatasetName="top_poject_source_codes.csv",useSavedModel=True,codeLineColumn='codes',epochs=60,batchSize=64,seqXLength=0,codeVocabSize=0):
        self.epochs=epochs
        self.seqXLength=seqXLength
        self.codeVocabSize=codeVocabSize
        self.batchSize=batchSize
        self.codeLineColumn = "codes";
        self.useSavedModel = useSavedModel
        self.defaultDatasetName = defaultDatasetName
        self.token=Tokenizer(lower=False, filters='!"#$%&*+,-./:;<=>?@[\\]^_`{|}~\t\n');


    def loadCodeData(self, path=''):
        if (path == ''):
            path =_PATH_ROOT_ +  self.defaultDatasetsFolder + self.defaultDatasetName
        self.df = pd.read_csv(path)
        self.df=self.df[:self.coustomDatasetLine]
        return (self.df)


    def codeToRawCodeList(self):
        self.rawCodeList=self.df[self.codeLineColumn].tolist()
        return (self.rawCodeList)


    def getTotalWords(self):
        return len(" ".join(self.rawCodeList))

    def getWordFrequencyCounts(self):
        self.tokenizeCodes()
        self.encodeWords()
        return len(" ".join(self.rawCodeList))

    def totalWords(self):
        return len(" ".join(self.rowcodeList))

    def tokenizeCodes(self):
        self.token.fit_on_texts(self.rawCodeList)
        self.EncodedCodes=self.token.texts_to_sequences(self.rawCodeList)
        return self.EncodedCodes

    def encodedWordsCount(self):
        self.tokenizeCodes();
        word_counts=self.token.word_counts
        return word_counts

    def encodedWordsIndex(self):
        self.tokenizeCodes();
        word_index=self.token.word_index
        return word_index

    def getAllCodesWordSize(self):
        vocabSize=len(self.token.word_counts)+1
        return vocabSize

        # break each line as ngram
    def provideNgramSequences(self):
        nGramcodeList=[]

        for d in self.EncodedCodes:
            if len(d)>1:
                for i in range(2,len(d)):
                    nGramcodeList.append(d[:i])
        #                     print (d[:i])
        self.nGramcodeList=nGramcodeList
        return nGramcodeList;

    # provide padding input for ML or DL algorithms
    def nGramcodeSeqPadding(self):
        self.paddedCodeNgramSequences=pad_sequences(self.nGramcodeList, maxlen=self.maxLengthPadding,padding="pre")
        return self.paddedCodeNgramSequences


    def provideModelInputOutPut(self):
        self.X=self.paddedCodeNgramSequences[:,:-1]
        self.y=self.paddedCodeNgramSequences[:,-1]
        # categorize y
        self.codeVocabSize=self.getAllCodesWordSize()
        self.y=to_categorical(self.y,num_classes=self.codeVocabSize)
        # X shape
        self.seqXLength=self.X.shape[1]


    def TrainLSTMModel(self):

        self.modelLSTM=Sequential()
        self.modelLSTM.add(Embedding(self.codeVocabSize ,50,input_length=self.seqXLength))
        self.modelLSTM.add(LSTM( 100,return_sequences=True))
        self.modelLSTM.add(LSTM(100))
        self.modelLSTM.add(Dense( 100,activation="relu"))
        self.modelLSTM.add(Dense(self.codeVocabSize,activation="softmax"))
        #self.modelLSTM.summary()
        self.modelLSTM.compile(loss='categorical_crossentropy',optimizer='adam',metrics=["accuracy"])
        self.modelLSTM.fit(self.X,self.y, batch_size=self.batchSize,epochs=self.epochs)
        self.modelLSTM.save( _PATH_ROOT_ + self.defaultTrainedModelPath +'CodeGeneration/LSTM_code_generation_model')


    def loadSavedModel(self):
        import os
        isfile =os.path.exists(os.path.join(os.getcwd(), _PATH_ROOT_ + self.defaultTrainedModelPath +'CodeGeneration/LSTM_code_generation_model', 'saved_model.pb'))
        if  isfile:
            self.modelLSTM = tensorflow.keras.models.load_model(_PATH_ROOT_ + self.defaultTrainedModelPath +'CodeGeneration/LSTM_code_generation_model')
            return True
        return False


    def generate_code(self,seed_text,n_lines):
        self.loadCodeData()
        self.codeToRawCodeList()
        self.tokenizeCodes()
        self.encodedWordsCount()
        self.provideNgramSequences()
        self.nGramcodeSeqPadding()
        self.provideModelInputOutPut()
        isDir =os.path.isdir(_PATH_ROOT_ + self.defaultTrainedModelPath +'CodeGeneration/LSTM_code_generation_model')
        isfile =os.path.exists(os.path.join(os.getcwd(), _PATH_ROOT_ + self.defaultTrainedModelPath +'CodeGeneration/LSTM_code_generation_model', 'saved_model.pb'))
        # load saved model
        if self.useSavedModel==True and isDir and isfile:
            self.loadSavedModel()
        else:
            self.TrainLSTMModel()

        predictionList= [ ]
        seq_length=self.seqXLength
        for i in range(n_lines):
            text=[]
            for _ in range(self.predictCodeLength):
                encoded=self.token.texts_to_sequences([seed_text])
                encoded=pad_sequences(encoded,maxlen=seq_length,padding='pre')
                y_pred=np.argmax( self.modelLSTM.predict(encoded),axis=1)
                # find to word dictinary which word mapped number
                predicted_code=""
                for word,index in self.token.word_index.items():
                    if index== y_pred:
                        predicted_code=word
                        break
                seed_text=seed_text +' '+ predicted_code
                text.append(predicted_code)

            seed_text=text[-1]
            text=' ' . join(text)
            print(text)
            predictionList.append(text)

        return  predictionList