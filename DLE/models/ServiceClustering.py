#!/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.

from configparser import ConfigParser
#Read config.ini file
config_object = ConfigParser()
config_object.read("config.ini")
#Get the _PATH_ROOT_
_PATH_ROOT_ = config_object["Path"]["root"]

import sys
sys.path.insert(1, _PATH_ROOT_ +'/DLE/utils')
from ImportingModules import *
from PreProcessTextData import *
from AIPipelineConfiguration import *


class ServiceCluster(AIPipelineConfiguration):
    X = []
    y = []
    model = None
    spacyNLP = None
    rawCodeList = [];
    EncodedCodes = [];
    df = pd.DataFrame();
    dfRowData = pd.DataFrame();

    def __init__(self,defaultDatasetName='programmableweb_apis.csv',targetCLMName='Category',k_cluster=50):
        self. targetCLMName = targetCLMName
        self. k_cluster = k_cluster
        self.defaultDatasetName = defaultDatasetName

    def loadData(self, path=''):
        if (path == ''):
            path =_PATH_ROOT_ +  self.defaultDatasetsFolder+self.defaultDatasetName
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
        self.df[clmName] = self.df[clmName].str.replace(pat, '')

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
        print("Embeding started....")
        embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        start = time.time()
        corpus_embeddings = embedder.encode(corpus)
        end = time.time()
        print("Embeding completed....")
        print(end - start)
        # Cluster BERT vectors by Kmeans
        num_clusters = self.k_cluster
        clustering_model = KMeans(n_clusters=num_clusters)
        print("Clustering  started....")
        # Fit the embedding with kmeans clustering.
        clustering_model.fit(corpus_embeddings)
        print("Clustering completed....")
        # Get the cluster id assigned to each data.
        cluster_assignment = clustering_model.labels_
        labels = cluster_assignment
        # Add back to originaal data
        self.df['Clusters'] = labels
        #         self.df['Clusters']=self.df['clusters']
        #         self.df =self.df.rename(columns = {'clusters': 'Clusters'}, inplace = False)
        return (self.df)





