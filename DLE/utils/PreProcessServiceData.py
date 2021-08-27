_PATH_ROOT_ = '/home/zakieh/PycharmProjects/'

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, _PATH_ROOT_ +'/DLE/utils')

from ImportingModules import *
from PreProcessTextData import *


class ServiceTextDataPreProcess(TextDataPreProcess):

    def __init__(self):
        self.level = 1

    def removeServiceDataStopWords(self, sentence):

        from nltk.corpus import stopwords
        global re_stop_words

        stopWordsList = self("data/service_stopwords.txt")
        stop_words = set(stopwords.words('english'))
        stop_words.update(['monthly', "google", "api", "apis", "service", "data", "REST", "RESTFUL", "website", "site"])
        re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

        return re_stop_words.sub(" ", sentence)

    def loadStopwords(self, path):
        """
        This function loads a stopword list from the *path* file and returns a
        set of words. Lines begining by '#' are ignored.
        """

        # Set of stopwords
        stopwords = set([])

        # For each line in the file
        for line in codecs.open(path, 'r', 'utf-8'):
            if not re.search('^#', line) and len(line.strip()) > 0:
                stopwords.add(line.strip())

        # Return the set of stopwords
        return stopwords
