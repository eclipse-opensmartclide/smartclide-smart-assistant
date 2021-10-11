import os
path = os.getcwd()
_PATH_ROOT_ = path+'/smartclide_service_classification_autocomplete/'
from .AIPipelineConfiguration import  *
from .PreProcessTextData import *


import os
import torch
import tensorflow
import numpy as np
import tensorflow
import pandas as pd
from tensorflow.keras  import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers.experimental import preprocessing



class CodeGenerationModel(AIPipelineConfiguration):
    X = []
    y = []
    seedText = ''
    generatorModel=None
    TrainedModel='gpt2'
    tokenizer = None
    EncodedCodes = [];
    nGramcodeList = []
    rawCodeList = [];
    maxLengthPadding = 40
    predictCodeLength =2
    maxLineReturn = 3
    df = pd.DataFrame();
    modelLSTM = Sequential()
    coustomDatasetLine = 2000
    paddedCodeNgramSequences = []

    def __init__(self, useSavedModel=True,
                 defaultDatasetName='',
                 device='cpu',
                 codeLineColumn='codes',
                 transformerModelName="congcongwang/gpt2_medium_fine_tuned_coder",
                 epochs=60,
                 batchSize=64,
                 seqXLength=0,
                 codeVocabSize=0,
                 temperature_GPT=1.0,
                 top_k_GPT=40,
                 top_p_GPT=0.9,
                 repetition_penalty_GPT=1.0

                 ):
        self.device = device
        self.epochs = epochs
        self.batchSize = batchSize
        self.top_k_GPT = top_k_GPT
        self.top_p_GPT = top_p_GPT
        self.seqXLength = seqXLength
        self.codeLineColumn = "codes";
        self.useSavedModel = useSavedModel
        self.codeVocabSize = codeVocabSize
        self.temperature_GPT = temperature_GPT
        self.defaultDatasetName = defaultDatasetName
        self.repetition_penalty_GPT = repetition_penalty_GPT
        self.transformerModelName = "congcongwang/distilgpt2_fine_tuned_coder",
        self.token = Tokenizer(lower=False, filters='!"#$%&*+,-./:;<=>?@[\\]^_`{|}~\t\n');
        if (defaultDatasetName == ''):
            self.defaultDatasetName = self.defaultCodeTrainDataset



    def loadCodeData(self, path=''):
        # if (path == ''):
        here = os.path.abspath(os.path.dirname(__file__))
        self.df = pd.read_csv(os.path.join(here, "project_source_codes.csv"))
        self.df = self.df[:self.coustomDatasetLine]
        return (self.df)

    def codeToRawCodeList(self):
        self.rawCodeList = self.df[self.codeLineColumn].tolist()
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
        self.EncodedCodes = self.token.texts_to_sequences(self.rawCodeList)
        return self.EncodedCodes

    def encodedWordsCount(self):
        self.tokenizeCodes();
        word_counts = self.token.word_counts
        return word_counts

    def encodedWordsIndex(self):
        self.tokenizeCodes();
        word_index = self.token.word_index
        return word_index

    def getAllCodesWordSize(self):
        vocabSize = len(self.token.word_counts) + 1
        return vocabSize

        # break each line as ngram

    def provideNgramSequences(self):
        nGramcodeList = []

        for d in self.EncodedCodes:
            if len(d) > 1:
                for i in range(2, len(d)):
                    nGramcodeList.append(d[:i])
        #                     print (d[:i])
        self.nGramcodeList = nGramcodeList
        return nGramcodeList;

    # provide padding input for ML or DL algorithms
    def nGramcodeSeqPadding(self):
        self.paddedCodeNgramSequences = pad_sequences(self.nGramcodeList, maxlen=self.maxLengthPadding, padding="pre")
        return self.paddedCodeNgramSequences

    def provideModelInputOutPut(self):
        self.X = self.paddedCodeNgramSequences[:, :-1]
        self.y = self.paddedCodeNgramSequences[:, -1]
        # categorize y
        self.codeVocabSize = self.getAllCodesWordSize()
        self.y = to_categorical(self.y, num_classes=self.codeVocabSize)
        # X shape
        self.seqXLength = self.X.shape[1]

    def TrainLSTMModel(self):

        self.modelLSTM = Sequential()
        self.modelLSTM.add(Embedding(self.codeVocabSize, 50, input_length=self.seqXLength))
        self.modelLSTM.add(LSTM(100, return_sequences=True))
        self.modelLSTM.add(LSTM(100))
        self.modelLSTM.add(Dense(100, activation="relu"))
        self.modelLSTM.add(Dense(self.codeVocabSize, activation="softmax"))
        # self.modelLSTM.summary()
        self.modelLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        self.modelLSTM.fit(self.X, self.y, batch_size=self.batchSize, epochs=self.epochs)
        here = os.path.abspath(os.path.dirname(__file__))
        self.modelLSTM.save(os.path.join(here + "/trained_models/CodeGeneration/", 'LSTM_code_generation_model') )

    def loadSavedModel(self):
        import os
        here = os.path.abspath(os.path.dirname(__file__))
        isfile = os.path.exists(os.path.join(here + "/trained_models/CodeGeneration/LSTM_code_generation_model/", 'saved_model.pb'))
        if isfile:
            self.modelLSTM = tensorflow.keras.models.load_model(os.path.join(here + "/trained_models/CodeGeneration/", 'LSTM_code_generation_model'))
            return True
        return False

    def generate_code(self, seed_text, n_lines):
        self.loadCodeData()
        self.codeToRawCodeList()
        self.tokenizeCodes()
        self.encodedWordsCount()
        self.provideNgramSequences()
        self.nGramcodeSeqPadding()
        self.provideModelInputOutPut()
        # load saved model
        if self.useSavedModel == True :
            self.loadSavedModel()
        else:
            self.TrainLSTMModel()

        predictionList = []
        seq_length = self.seqXLength
        for i in range(n_lines):
            text = []
            for _ in range(self.predictCodeLength):
                encoded = self.token.texts_to_sequences([seed_text])
                encoded = pad_sequences(encoded, maxlen=seq_length, padding='pre')
                y_pred = np.argmax(self.modelLSTM.predict(encoded), axis=1)
                # find to word dictinary which word mapped number
                predicted_code = ""
                for word, index in self.token.word_index.items():
                    if index == y_pred:
                        predicted_code = word
                        break
                seed_text = seed_text + ' ' + predicted_code
                text.append(predicted_code)

            seed_text = text[-1]
            text = ' '.join(text)
            print(text)
            predictionList.append(text)

        return predictionList

    def LoadCodeCorpos(self, ClmName):
        self.X = list(set(self.df[ClmName]))
        return self.X

    def PrepareModelGPT2(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from transformers import AutoTokenizer, AutoModelWithLMHead
        self.loadCodeData()
        self.LoadCodeCorpos('codes')
        self.tokenizer = AutoTokenizer.from_pretrained("congcongwang/gpt2_medium_fine_tuned_coder")
        self.model = AutoModelWithLMHead.from_pretrained("congcongwang/gpt2_medium_fine_tuned_coder")

    def generateCodeByGPT2(self, seed_text, maxLineReturn, lengthCodeLine):
        self.seedText = seed_text
        self.predictCodeLength = lengthCodeLine
        self.maxLineReturn = lengthCodeLine
        self.PrepareModelGPT2()
        MAX_LENGTH = int(1000)
        encoded_prompt = self.tokenizer.encode(self.seedText, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        generatedcodeOutput = self.model.generate(
            input_ids=encoded_prompt,
            max_length=self.predictCodeLength + len(encoded_prompt[0]),
            temperature=self.temperature_GPT,
            top_k=self.top_k_GPT,
            top_p=self.top_p_GPT,
            repetition_penalty=self.repetition_penalty_GPT,
            do_sample=True,
            num_return_sequences=self.maxLineReturn,
        )
        if len(generatedcodeOutput.shape) > 2:
            generatedcodeOutput.squeeze_()

        generatedCodesequences = []

        for generated_sequence_idx, generated_sequence in enumerate(generatedcodeOutput):
            generated_sequence = generated_sequence.tolist()
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            total_sequence = (seed_text + text[len(self.tokenizer.decode(encoded_prompt[0],
                                                                         clean_up_tokenization_spaces=True)):])
            generatedCodesequences.append(total_sequence)

        return generatedCodesequences
    
    def getCurrentDirectory(self):
        import os
        path = os.getcwd()
        return(path)
    
    def getParentDirectory(self):
        import os
        path = os.getcwd()
        return(os.path.abspath(os.path.join(path, os.pardir)))
    
    def getTrainedModelsDirectory(self):
        import os
        from smartclide_service_classification_autocomplete import getPackagePath
        packageRootPath = getPackagePath()
        return (packageRootPath+"/trained_models/CodeGeneration/")
    

    def getTrainedModel(self,modelName):
        import os
        TrainrdModelPath = self.getTrainedModelsDirectory()
        path=TrainrdModelPath+'/'+modelName
        return path

    def IsTrainedModelExist(self,modelName):
        import os
        TrainrdModelPath = self.getTrainedModelsDirectory()
        isfile = os.path.exists(TrainrdModelPath+'/'+modelName)
        return isfile
    
    def loadGenerator(self):
        import os
        import pickle
        from transformers import pipeline
        if (self.IsTrainedModelExist('GPTgenerator.pkl')):
            file = open(self.getTrainedModel('GPTgenerator.pkl'), 'rb')
            self.generatorModel = pickle.load(file)
            file.close()
        else:
            self.generatorModel = pipeline('text-generation', model=self.TrainedModel)
#             print("=========================")
#             print(os.path.join(self.getTrainedModelsDirectory(), "GPTgenerator.pkl"))
#             from smartclide_service_classification_autocomplete import getPackagePath
#             print(getPackagePath())
#             print("=========================")
            pickle.dump(self.generatorModel,open(os.path.join(self.getTrainedModelsDirectory(), "GPTgenerator.pkl"), 'wb'))  
            
        return (self.generatorModel)
    
    
    
    def GenerateCode(self,codeInput,maxLength):
        if (self.generatorModel != None) :
                generated_code_arr = self.generatorModel(codeInput, max_length=maxLength, do_sample=True, temperature=0.9) 
                generatedCode=generated_code_arr[0]['generated_text']
                return(generatedCode)




