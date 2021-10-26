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
import pandas as pd
import numpy as np
# from sklearn import *
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import normalize
from nltk.stem.wordnet import WordNetLemmatizer

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# # toknize
import nltk

class ServiceClusterModel(AIPipelineConfiguration):
    X = []
    y = []
    tree = None
    model = None
    spacyNLP = None
    embededCorpus= [];
    df = pd.DataFrame();
    dfRowData = pd.DataFrame();

    def __init__(self,defaultDatasetName='programmableweb_apis.csv',targetCLMName='Category',k_cluster=50):
        self. targetCLMName = targetCLMName
        self. k_cluster = k_cluster
        self.defaultDatasetName = defaultDatasetName

    def loadData(self, path=''):
        if (path == ''):
            path =self.defaultDatasetName
        self.df = pd.read_csv(path)
        return (self.df)

    def filterColumn(self, clmnNames=["Description", "Category"]):
        self.df = self.df[clmnNames]
        self.df = self.df.rename(columns={clmnNames[0]: 'description', clmnNames[1]: 'category'}, inplace=False)
        return (self.df)

    def filterLowFrequencyCategory(self, countLimit=30, clmnNames="Category"):
        self.removeNullrows(clmnNames)
        counts_category = self.df.groupby(clmnNames)[clmnNames].transform(len)
        mask = (counts_category > countLimit)
        self.df = self.df[mask]
        return (self.df)

    def removeHTMLTags(self, clmName):
        textPreProcessOBJ = TextDataPreProcess()
        self.df[clmName] = self.df[clmName].astype(str).apply(textPreProcessOBJ.removeHTMLTags)
        return (self.df)

    def removeNullrows(self, columnName):
        self.df = self.df[pd.notnull(self.df[columnName])]
        return (self.df)

    def splitUpperCase(self, sentence):
        splitted = re.findall('[A-Z]+[a-z]*', sentence)
        return splitted

    def getUniqueCategoryCount(self, colmnName):
        return (self.df[colmnName].nunique())

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

    # remove duplicate words from category
    def unique_list(self, l):
        ulist = []
        [ulist.append(x) for x in l if x not in ulist]
        return ulist

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
        self.df[clmName] = self.df[clmName].str.replace(pat, '',regex=True)

    def lowercase(self, clmName):
        self.df[clmName] = self.df[clmName].str.lower()

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

    def preProcessLabels(self, colmnName):
        import re
        # tokenize labels by Uper case
        df_api = self.df[pd.notnull(self.df[colmnName])]
        temp = self.df[colmnName]
        self.df[colmnName] = self.df[colmnName].astype(str).apply(self.splitUpperCase)
        self.df[colmnName] = self.df[colmnName]
        self.df['Tags'] = self.df[colmnName].apply(' '.join)
        df_categories = self.df['Tags'].str.split(',', expand=True)
        self.df[colmnName] = df_categories[0]
        self.df = self.df[pd.notnull(self.df[colmnName])]
        self.df[colmnName] = self.df[colmnName].str.split(' ').str[0]

        return (self.df)

    def getword2vecVec(self, text):
        doc = self.spacyNLP(text)
        # Word2vec feature eng
        vec = doc.vector
        return vec

    def getCoulmnWord2vecVec(self, clmName):
        import spacy
        self.spacyNLP = spacy.load('en_core_web_lg')
        # get word2vec from data frame
        self.df["vec"] = self.df[clmName].apply(lambda x: self.getword2vecVec(x))

    def provideWord2vecXInput(self, dimention=300):

        # providing X for AI algorithm
        X = self.df['vec'].to_numpy()
        # For converting one row and   300 coulmn
        X = X.reshape(-1, 1)
        # For have row*300 dimention
        X = np.concatenate(np.concatenate(X, axis=0), axis=0).reshape(-1, dimention)
        self.X = X

    def dimensionReductionPCA(self, n):
        import numpy as np
        from sklearn.decomposition import PCA
        from matplotlib import pyplot as plt
        newX = self.X
        # Descending to 2D
        pca = PCA(n_components=n)
        pca.fit(newX)
        newX = pca.fit_transform(newX)
        return newX

    def clusterKMeans(self, k=30):
        from sklearn.cluster import KMeans
        from sklearn import metrics

        self.model = KMeans(n_clusters=k, random_state=0).fit(self.X)
        labels = self.model.labels_
        # Glue back to originaal data
        self.df['clusters'] = labels

    def getClusteredClmName(self, clmName):
        # Lets analyze the clusters
        df = self.df
        df = df.groupby('clusters').agg({clmName: lambda x: ' '.join(x), 'description': 'first', 'clusters': 'first'})
        df[clmName] = df[clmName].astype(str).apply(serviceClustertObj.splitUpperCase)
        df[clmName] = df[clmName].apply(' '.join)
        df[clmName] = df[clmName].apply(lambda x: ' '.join(self.unique_list(x.split())))
        return df

    def clusterKMeansUsingBert(self, clmName):
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        # Convert Category to arryy inorder to send to bert encoder
        corpus = self.df[clmName].to_numpy()
        embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        start = time.time()
        corpus_embeddings = embedder.encode(corpus)
        end = time.time()
        print(end - start)
        # Cluster BERT vectors by Kmeans
        num_clusters = self.k_cluster
        clustering_model = KMeans(n_clusters=num_clusters)
        # Fit the embedding with kmeans clustering.
        clustering_model.fit(corpus_embeddings)
        # Get the cluster id assigned to each data.
        cluster_assignment = clustering_model.labels_
        labels = cluster_assignment
        # Add back to originaal data
        self.df['Clusters'] = labels
        return (self.df)


    def embeding(self, clmName):
        from sentence_transformers import SentenceTransformer
        # Convert Category to arryy inorder to send to bert encoder
        corpus= self.df[clmName].to_numpy()
        embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        start = time.time()
        self.embededCorpus  = embedder.encode(corpus )
        end = time.time()
        print(end - start)
   

    def clusterHACBert(self, clmName,n):
        import scipy.cluster.hierarchy as hac
        from scipy.cluster.hierarchy import fcluster
        self.tree = hac.linkage(self.embededCorpus  , method="complete",metric="euclidean")
        clustering = fcluster(self.tree,n,'maxclust')
        # Get the cluster id assigned to each data.
        # Add back to originaal data
        labels =clustering
        self.df['Clusters'] = labels
        return (self.df)
    
   
    def  plotdendoGram(self, h,w):  
        import matplotlib.pyplot as plt 
        import scipy.cluster.hierarchy as hac
        fg, axs = plt.subplots(1, 1, figsize=(h,w))
        plt.clf()
        hac.dendrogram(self.tree)
        plt.show()
        
        



