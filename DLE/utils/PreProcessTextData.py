_PATH_ROOT_ = '/home/zakieh/PycharmProjects/'

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, _PATH_ROOT_ +'/DLE/utils')
sys.path.insert(1, _PATH_ROOT_ +'/DLE/utils')

from ImportingModules import *


class TextDataPreProcess:

    def __init__(self):
        self.level = 1

    def cleanPunc(self, sentence):
        import re
        cleaned = re.sub(r'[?|!|\[|\]|\'|"|#]', r'', sentence)
        cleaned = re.sub(r'[.|)|(|\|/]', r' ', cleaned)
        cleaned = re.sub(r'[\d]', r' ', cleaned)  # remove digite
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\n", "")
        return cleaned

    def removeStopWords(self, sentence):
        from nltk.corpus import stopwords
        global re_stop_words

        stop_words = set(stopwords.words('english'))
        stop_words.update(['monthly', "google", "api", "apis", "service", "data", "REST", "RESTFUL", "website", "site"])
        re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
        return re_stop_words.sub(" ", sentence)

    def wordLemmatizer(self, sentence):
        from nltk.stem.wordnet import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        lem_sentence = " ".join([lemmatizer.lemmatize(i) for i in sentence])
        return lem_sentence

    def getTopNwords(self, corpus, n=None):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = CountVectorizer.fit(corpus)
        bagofWords = vec.transform(corpus)
        sumWords = bagofWords.sum(axis=o)
        wordFreq = [(word, sumWords[0, indx]) for word, idx in vec.vocabulary_items()]
        wordFreq = sorted(wordFreq, key=lambda x: x[1], reverse=True)
        return wordsFreq[:n]

    def getCommonWords(self, count_data, countVectorizerOBJ, n):
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
