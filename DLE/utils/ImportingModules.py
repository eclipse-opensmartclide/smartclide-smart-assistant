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

# lemmatizer
from nltk.stem.wordnet import WordNetLemmatizer

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# classifiers
from sklearn.svm import LinearSVC
# from . import TextDataPreProcess
import ktrain
from ktrain import text

import pickle

# toknize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')