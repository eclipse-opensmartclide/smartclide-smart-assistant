#!/usr/bin/python3
# Eclipse Public License 2.0

import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('omw-1.4') 
   

class TextDataPreProcess:

    def __init__(self):
        self.level=1


    def cleanPunc(self, sentence):
        import re
        cleaned = re.sub(r'[?|!|\[|\]|\'|"|#]', r'', sentence)
        cleaned = re.sub(r'[.|)|(|\|/]', r' ', cleaned)
        cleaned = re.sub(r'[\d]', r' ', cleaned)  # remove digite
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\n", "")
        return cleaned

    def removeStopWords(self, sentence):
        try:
            from nltk.corpus import stopwords
            global re_stop_words
            stop_words = set(stopwords.words('english'))
            stop_words.update(['monthly', "google", "api", "apis", 'json','Json',"service", "provide","including", "data", "REST", "RESTFUL", "website", "site"])
        except:
            nltk.download('stopwords')
            from nltk.corpus import stopwords
            global re_stop_words
            stop_words = set(stopwords.words('english'))
            stop_words.update(['monthly', "google", "api", "apis", 'json','Json',"service", "provide","including", "data", "REST", "RESTFUL", "website", "site"])
              
        re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
        return re_stop_words.sub(" ", sentence)

    def wordLemmatizer(self, sentence):
        #nltk.download('punkt')
        #nltk.download('wordnet')
        try:
            from nltk.stem.wordnet import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            lem_sentence = " ".join([lemmatizer.lemmatize(i) for i in sentence])
        except:
            nltk.download('wordnet')
            nltk.download('punkt')
            nltk.download('omw-1.4') 
            from nltk.stem.wordnet import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            lem_sentence = " ".join([lemmatizer.lemmatize(i) for i in sentence])
        return lem_sentence

    def getTopNwords(self, corpus, n=None):
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer.fit(corpus)
        bagofWords = vec.transform(corpus)
        sumWords = bagofWords.sum(axis=o)
        wordFreq = [(word, sumWords[0, indx]) for word, idx in vec.vocabulary_items()]
        wordFreq = sorted(wordFreq, key=lambda x: x[1], reverse=True)
        return wordsFreq[:n]

    def getCommonWords(self, count_data, countVectorizerOBJ, n):
#         words = countVectorizerOBJ.get_feature_names_out()
        try:
              words = countVectorizerOBJ.get_feature_names_out()
        except:
              words = countVectorizerOBJ.get_feature_names() 



        total_counts = np.zeros(len(words))
        for t in count_data:
            total_counts += t.toarray()[0]
        count_dict = (zip(words, total_counts))
        count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:n]
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words))
        return words

    def removeHTMLTags(self, string):
        result = re.sub('<.*?>', '', string)
        return result

    def uniqueList(self, l):
        ulist = []
        [ulist.append(x) for x in l if x not in ulist]
        return ulist

