#!/usr/bin/env python
# coding: utf-8

# In[165]:


#!usr/bin/python

# Copyright 2021 AIR Institute
# See LICENSE for details.

import re
import os
import time
import json
import csv
import sys
import warnings
import string 
import requests
import pandas as pd
import numpy as np

# Corpus Processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.tokenize import WhitespaceTokenizer
from nltk import sent_tokenize, word_tokenize

#lemmatizer
from nltk.stem.wordnet import WordNetLemmatizer

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.preprocessing import normalize 

#classifiers
from sklearn.svm import LinearSVC
# from . import TextDataPreProcess

import pickle

#toknize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# In[174]:


class TextDataPreProcess:

    def cleanPunc(self,sentence): 
        
        cleaned = re.sub(r'[?|!|\[|\]|\'|"|#]',r'',sentence)
        cleaned = re.sub(r'[.|)|(|\|/]',r' ',cleaned)
        cleaned = re.sub(r'[\d]',r' ',cleaned)#remove digite
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\n","")
        return cleaned


    def removeStopWords(self,sentence): 
        global re_stop_words
        stop_words = set(stopwords.words('english'))
        stop_words.update(['monthly',"google","api"])
        re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

        return re_stop_words.sub(" ", sentence)
    
    
    def word_lemmatizer(self,sentence):
        lemmatizer=WordNetLemmatizer()
        lem_sentence=" ".join([lemmatizer.lemmatize(i) for i in sentence])
        return lem_sentence

    def getTopNwords(self,corpus,n=None):
        vec=CountVectorizer.fit(corpus)
        bagofWords= vec.transform(corpus)
        sumWords=bagofWords.sum(axis=o)
        wordFreq=[(word,sumWords[0,indx]) for word, idx in vec.vocabulary_items()]
        wordFreq=sorted(wordFreq,key = lambda x: x[1] , reverse=True)
        return wordsFreq[:n]

    def getCommonWords(self,count_data, count_vectorizer,n):
        words = count_vectorizer.get_feature_names()
        total_counts = np.zeros(len(words))
        for t in count_data:
            total_counts+=t.toarray()[0]

        count_dict = (zip(words, total_counts))
        count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:n]
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words)) 
        return words    
    


class ServiceClassification:
    X=[]
    y=[]
    model=None
    tfidf=TfidfVectorizer()
    df = pd.DataFrame() ;
    rawCodeList = [] ;
    EncodedCodes=[];
    token=Tokenizer(lower=False, filters='!"#$%&*+,-./:;<=>?@[\\]^_`{|}~\t\n');
    defaultDatasetPath="data/Clustered_dataset.csv";
    coustomDataRow=4000
    _Use_saved_model=True; 
    
    def __init__( self):
        self.level = 1

   
    def loadData(self):
        self.df=pd.read_csv(self.defaultDatasetPath)
        #self.df=self.df[:self.coustomDataRow]
        return (self.df)
    
    def removeNullrows(self,columnName):
        self.df=self.df[pd.notnull(self.df[columnName])]
        self.df=self.df[:self.coustomDataRow]
        return (self.df)
    
    def cleanPunc(self): 
        textPreProcessOBJ=TextDataPreProcess()
        self.df['description'] = self.df['description'].astype(str).apply(textPreProcessOBJ.cleanPunc)
        return (self.df)

 
    def removeStopWords(self): 
        textPreProcessOBJ=TextDataPreProcess()
        self.df['description'] = self.df['description'].apply(textPreProcessOBJ.removeStopWords)
        return (self.df)

    def tokenize(self): 
        tokenizer=RegexpTokenizer(r'\w+')
        self.df['description'] = self.df['description'].apply(lambda x:tokenizer.tokenize(x.lower()))
        return (self.df)

    def word_lemmatizer(self): 
        textPreProcessOBJ=TextDataPreProcess()
        self.df['description'] = self.df['description'].apply(textPreProcessOBJ.word_lemmatizer)
        return (self.df)
    
    def getCommonWord(self,sentence,n=50):
        textPreProcessOBJ=TextDataPreProcess()
        count_vectorizer = CountVectorizer()
        count_data = count_vectorizer.fit_transform(self.df['description'])
        commonWords= textPreProcessOBJ.getCommonWords(count_data, count_vectorizer,n)
        return commonWords   
    
    
    def removeCommonWords(self,n=50):
        commonwords=self.getCommonWord(n)
        pat = r'\b(?:{})\b'.format('|'.join(commonwords))
        self.df['description'] = self.df['description'].str.replace(pat, '') 
        
        
    def preprocessServicesData(self):
        self.loadData()
        self.removeNullrows('description')
        self.removeStopWords()
        self.tokenize()
        self.word_lemmatizer()
        self.getCommonWord(60)
        self.removeCommonWords()
      
    
    def provideMLModelInput(self):
        self.tfidf=TfidfVectorizer()
        self.X=self.tfidf.fit_transform(self.df["description"])
        self.y=self.df['category']
        
        
    
    def TrainLinerSVCModel(self):
        
        self.preprocessServicesData()

        self.tfidf=TfidfVectorizer()


        d=self.df["description"]
        self.X=self.tfidf.fit_transform(d)
        self.y=self.df['category']
        
        self.model=LinearSVC(class_weight='balanced',random_state=777)
        self.model.fit(self.X,self.y)
        #save models
        pickle.dump(self.model,open('clf.pkl','wb'))
        pickle.dump(self.tfidf,open('tfidf.pkl','wb'))

        
        
        
    def loadSavedModel(self):

        isfile =os.path.exists(os.path.join(os.getcwd(), './', 'clf.pkl'))
        if  isfile:     
            self.model =pickle.load(open('clf.pkl','rb'))
            self.tfidf =pickle.load(open('tfidf.pkl','rb'))
            return True
        
        return False
  

        
    def predictBOWML(self,x):
        
        textPreProcessOBJ=TextDataPreProcess()       
        x=x.lower()
        x=textPreProcessOBJ.removeStopWords(x)
        x=textPreProcessOBJ.cleanPunc(x)
        
        if(len(x)<2):
            return False
        
        
        isfile =os.path.exists(os.path.join(os.getcwd(), './', 'clf.pkl'))

        #load saved model
        if self._Use_saved_model==True and isfile:
                self.loadSavedModel()
        else:
                self.TrainLinerSVCModel()
                

        
        
        x=self.tfidf.transform([x])
        x=x.toarray()
        pred=self.model.predict(x)
        return pred


# In[175]:


#sample Code
serviceObj=ServiceClassification()

X2="If you are still coming to grips with the very basics of programming, you really want to work your way through a few tutorials first. The Python Wiki lists several options for you. Stack Overflow may not be the best place to ask for help with your issues."
X3="This mashup combines job listings from Indeed with company reviews from Glassdoor for job searches within the data science"
X4="This Python example demonstrates how to perform datatype binding using the Numpy library. Also, this test module performs table-related operations."

pred=serviceObj.predictBOWML(X3)


# In[176]:


print(pred)


# In[ ]:




