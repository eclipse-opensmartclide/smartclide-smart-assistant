#!/usr/bin/python3
# Eclipse Public License 2.0

import os
import csv
import torch
import logging
import numpy as np
import pandas as pd

from .PreProcessTextData import *
from .AIPipelineConfiguration import *

from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

MAX_LENGTH = int(10000)
logger = logging.getLogger(__name__)


class CodesDataset(Dataset):

    def __init__(self, codes_dataset_path='./'):
        """Creates a torch Dataset with opensource codes ."""
        super().__init__()
        code_dataset_path = AIPipelineConfiguration.defaultCodeTrainDataset
        self.code_list = []
        self.end_of_text_token = "<|endcode|>"
        with open(code_dataset_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            x = 0
            for row in csv_reader:
                code_str = f"Code:{row[1]}{self.end_of_text_token}"
                self.code_list.append(code_str)

    def __len__(self):
        """return target column as list"""
        return len(self.code_list)

    def __getitem__(self, item):
        return self.code_list[item]


class CodeGenerationModel(AIPipelineConfiguration):
    seed_text = ''
    generator_model = None
    tokenizer = None
    encoded_codes = [];
    ngram_codeList = []
    max_length_padding = 40
    predict_code_length = 2
    max_line_return = 3
    df = pd.DataFrame();
    custom_dataset_line = 2000
    padded_code_ngram_sequences = []
    transformer_model_name = 'gpt2'
    stop_token = '<|endcode|>'
    trained_model = 'gpt2_codegenerator_trained.pt'

    codeLoder = []
    max_seqlen = 200
    learning_rate = 3e-5
    tokenizer_class = None

    model_classes = {
        "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
        # "gpt2-medium": (GPT2LMHeadModel, GPT2Tokenizer),
        # "distilgpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    }

    def __init__(self, use_saved_model=True,
                 default_dataset_name='',
                 device='cpu',
                 code_line_column='codes',
                 epochs=3,
                 batch_size=16,
                 seqx_length=0,
                 code_vocab_size=0,
                 temperature_GPT=1.0,
                 top_k_GPT=40,
                 top_p_GPT=0.9,
                 repetition_penalty_GPT=1.0
                 ):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if default_dataset_name == '':
            self.default_dataset_name = self.defaultCodeTrainDataset

        if temperature_GPT > 0.7 or temperature_GPT > 40:
            logger.info(" lower temperatures or top_k is recommended.")
        self.top_k_GPT = top_k_GPT
        self.top_p_GPT = top_p_GPT
        self.temperature_GPT = temperature_GPT

        if batch_size > 32 or epochs > 5:
            logger.info(" 3-5  epoch and 16/32 batch_size recommended.")
        self.epochs = epochs
        self.batch_size = batch_size

        self.seqx_length = seqx_length
        self.code_line_column = "codes";
        self.use_saved_model = use_saved_model
        self.code_vocab_size = code_vocab_size
        self.default_dataset_name = default_dataset_name
        self.repetition_penalty_GPT = repetition_penalty_GPT

    def loadCodeData(self, path=''):
        here = os.path.abspath(os.path.dirname(__file__))
        self.df = pd.read_csv(os.path.join(here, "codes.csv"))
        self.df = self.df[:self.custom_dataset_line]
        return (self.df)

    def getTransformerModel(self):
        return (self.transformer_model_name)

    def getTransformerModelName(self):
        return (self.transformer_model_name)

    def getCurrentDirectory(self):
        path = os.getcwd()
        return (path)

    def getParentDirectory(self):
        path = os.getcwd()
        return (os.path.abspath(os.path.join(path, os.pardir)))

    def getTrainedModelsDirectory(self):
        from smartclide_service_classification_autocomplete import getPackagePath
        packageRootPath = getPackagePath()
        return (packageRootPath + "/trained_models/")

    def getTrainedModel(self, modelName):
        TrainrdModelPath = self.getTrainedModelsDirectory()
        path = TrainrdModelPath + '/' + modelName
        return path

    def IsTrainedModelExist(self, modelName):
        TrainrdModelPath = self.getTrainedModelsDirectory()
        isfile = os.path.exists(TrainrdModelPath + '/' + modelName)
        return isfile

    def loadSavedModel(self):
        import os
        here = os.path.abspath(os.path.dirname(__file__))
        isfile = os.path.exists(os.path.join(here + "/trained_models/LSTM_code_generation_model/", 'saved_model.pb'))
        if isfile:
            self.modelLSTM = tensorflow.keras.models.load_model(
                os.path.join(here + "/trained_models/", 'LSTM_code_generation_model'))
            return True
        return False

    def LoadCodeCorpos(self, ClmName):
        self.X = list(set(self.df[ClmName]))
        return self.X

    

    def GenerateCode(self, codeInput, maxLength):
        if (self.generatorModel != None):
            generated_code_arr = self.generatorModel(codeInput, max_length=maxLength, do_sample=True, temperature=0.9)
            generatedCode = generated_code_arr[0]['generated_text']
            return (generatedCode)

    def loadDataset(self):
        dataset = CodesDataset()
        self.codeLoder = DataLoader(CodesDataset(), batch_size=1, shuffle=True)
        return self.codeLoder

    def trainGenerator(self, model_type=""):

        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        from transformers import AdamW

        if model_type == "":
            model_type = self.transformer_model_name
            print(model_type)

        try:
            model_type = model_type.lower()
            model_class, tokenizer_class = self.model_classes[model_type]
        except KeyError:
            raise KeyError("the model {} you specified is not supported.)")

        tokenizer_class = GPT2Tokenizer.from_pretrained(self.getTransformerModel())
        model_class = GPT2LMHeadModel.from_pretrained(self.getTransformerModel())
        model_class = model_class.to(self.device)
        model_class.train()
        optimizer = AdamW(model_class.parameters(), lr=self.learning_rate)
        proc_seq_count = 0
        sum_loss = 0.0
        batch_count = 0
        tmp_codes_tens = None
        models_folder = "trained_models"
        if not os.path.exists(models_folder):
            os.mkdir(models_folder)

        epoch = self.epoch
        print(f"EPOCH {epoch} started" + '=' * 30)
        for idx, code in enumerate(self.codeLoder):
            code_tens = torch.tensor(tokenizer_class.encode(code[0])).unsqueeze(0).to(self.device)
            # Skip sample from dataset if it is longer than MAX_SEQ_LEN
            if code_tens.size()[1] > self.max_seqlen:
                continue

            # The first code sequence in the sequence
            if not torch.is_tensor(tmp_codes_tens):
                tmp_codes_tens = code_tens
                continue
            else:
                # The next code does not fit in so we process the sequence and leave the last code
                # as the start for next sequence
                if tmp_codes_tens.size()[1] + code_tens.size()[1] > self.max_seqlen:
                    work_codes_tens = tmp_codes_tens
                    tmp_codes_tens = code_tens
                else:
                    # Add the code to sequence, continue and try to add more
                    tmp_codes_tens = torch.cat([tmp_codes_tens, code_tens[:, 1:]], dim=1)
                    continue

            outputs = model_class(work_codes_tens, labels=work_codes_tens)
            loss, logits = outputs[:2]
            loss.backward()
            sum_loss = sum_loss + loss.detach().data

            proc_seq_count = proc_seq_count + 1
            if proc_seq_count == self.batch_size:
                proc_seq_count = 0
                batch_count += 1
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                model_class.zero_grad()

            if batch_count == 100:
                print(f"sum loss {sum_loss}")
                batch_count = 0
                sum_loss = 0.0

        # Store the model after each epoch to compare the performance of them
        torch.save(model_class.state_dict(), os.path.join(models_folder, f"gpt2_codegenerator_trained.pt"))

    def loadGenerator(self):
        import os
        import pickle
        from transformers import pipeline
        if (self.IsTrainedModelExist('GPTgenerator.pkl')):
            file = open(self.getTrainedModel('GPTgenerator.pkl'), 'rb')
            self.generatorModel = pickle.load(file)
            file.close()
        else:
            self.generatorModel = pipeline('text-generation', model='gpt2')
            pickle.dump(self.generatorModel,
                        open(os.path.join(self.getTrainedModelsDirectory(), "GPTgenerator.pkl"), 'wb'))

        return (self.generatorModel)

    def loadGPT2Generator(self):
        import os
        import torch

        try:
            models_folder = "trained_models"
            model_path = os.path.join(models_folder, f"gpt2_codegenerator_trained.pt")
            self.tokenizer_class = GPT2Tokenizer.from_pretrained(self.getTransformerModel())
            self.generatorModel = GPT2LMHeadModel.from_pretrained(self.getTransformerModel(),
                                                                  pad_token_id=self.tokenizer_class.eos_token_id)
            self.generatorModel.load_state_dict(torch.load(model_path))
        except FileNotFoundError:
            print("No training file is exist")
        return (self.generatorModel)

    def generateCodeByTrainedGPT2(self, inputCode):
        try:
            self.generatorModel.to(self.device)
        except FileNotFoundError:
            self.loadGPT2Generator()
            self.generatorModel.to(self.device)
        length = 6
        tokenizer = GPT2Tokenizer.from_pretrained(self.getTransformerModel())
        encoded_prompt = tokenizer.encode(inputCode, add_special_tokens=False, return_tensors="pt")
        # encoded_prompt.to(device)
        output_sequences = self.generatorModel.generate(input_ids=encoded_prompt, max_length=6)
        generated_sequence = output_sequences[0].tolist()
        code = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        code = code[: code.find(self.stop_token) if stop_token == 1 else None]
        print(code)
        return code

    def loadTrainedGenerator(self):
        """ Load trained web service code generator"""
        import pickle
        import torch
        from transformers import pipeline

        try:
            models_folder = "trained_models"
            model_path = os.path.join(models_folder, f"gpt2_codegenerator_trained.pt")
            #             self.tokenizer_class = GPT2Tokenizer.from_pretrained(self.getTransformerModelName())
            #             self.generatorModel = GPT2LMHeadModel.from_pretrained(self.getTransformerModelName())
            self.tokenizer_class = GPT2Tokenizer.from_pretrained('gpt2')
            self.generatorModel = GPT2LMHeadModel.from_pretrained('gpt2')
            self.generatorModel.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))


        except FileNotFoundError:
            print("No training file is exist")
        return (self.generatorModel)

    def generate_code_trainedGPT2(self, seed_text, lengthCodeLine, max_line_return):
        """ Generate code using trained  web service code generator
            with optional lengthCodeLine and max_line_return
        """

        self.seed_text = seed_text
        self.predict_code_length = lengthCodeLine
        self.max_line_return = max_line_return

        encoded_prompt = self.tokenizer_class.encode(self.seed_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        output_sequences = self.generatorModel.generate(
            input_ids=encoded_prompt,
            max_length=self.predict_code_length, num_beams=5, no_repeat_ngram_size=2, num_return_sequences=3,
            early_stopping=True)

        for output in output_sequences:
            print("--------")
            generated_sequence = output.tolist()
            generatedCodes = self.tokenizer_class.decode(generated_sequence, clean_up_tokenization_spaces=True)
            generatedCodes = generatedCodes[: generatedCodes.find(self.stop_token) if stop_token == 1 else None]
            print(generatedCodes)

        return generatedCodes

    def loadTrainedGenerator(self):
        """
        Load trained web service code generator
        """
        import pickle
        import torch
        from transformers import pipeline

        try:
            trained_models_folder = self.getTrainedModelsDirectory()
            model_path = os.path.join(trained_models_folder, f"gpt2_codegenerator_trained.pt")
            self.tokenizer_class = GPT2Tokenizer.from_pretrained(self.getTransformerModelName())
            self.generatorModel = GPT2LMHeadModel.from_pretrained(self.getTransformerModelName())
            # self.tokenizer_class = GPT2Tokenizer.from_pretrained('gpt2')
            # self.generatorModel = GPT2LMHeadModel.from_pretrained('gpt2')
            self.generatorModel.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print("+" * 40)
            print("Trained model is loaded on " + self.device)
            print("+" * 40)

        except FileNotFoundError:
            print("+" * 40)
            print("No training file is exist GPT2-2-2")
            print("+" * 40)

        return (self.generatorModel)

    def post_process_code(self, code):
        if (code.find("<|endcode|>") != -1):
            self.stop_token = "<|endcode|>"
        else:
            self.stop_token = ";"

        if len(code) > 2:
            result = code.partition(self.stop_token)
            code1 = result[0]
            stop_token = result[1]
            code2 = result[0]

            if self.stop_token == ";":
                return code1 + ";"
            else:
                return code1

        return code

    def generate_code_trainedGPT2(self, seed_text, lengthCodeLine, max_line_return):

        """
            Generate code using trained  web service code generator
            with optional lengthCodeLine and max_line_return
        """

        self.seed_text = seed_text
        self.predict_code_length = lengthCodeLine
        self.max_line_return = max_line_return

        encoded_prompt = self.tokenizer_class.encode(self.seed_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        output_sequences = self.generatorModel.generate(
            input_ids=encoded_prompt,
            max_length=self.predict_code_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            num_return_sequences=self.max_line_return)

        text = None
        generatedCodesList = []
        for output in output_sequences:
            generated_sequence = output.tolist()
            code_line = self.tokenizer_class.decode(generated_sequence, clean_up_tokenization_spaces=True)
            code_line = code_line[: code_line.find(self.stop_token) if self.stop_token == 1 else None]
            generatedCodesList.append(self.post_process_code(code_line))
        print(generatedCodesList)
        return generatedCodesList

    # LSTM Model

    def codeToRawCodeList(self):
        self.rawCodeList = self.df[self.code_line_column].tolist()
        return (self.rawCodeList)

    def generate_code(self, seed_text, n_lines):
        self.loadCodeData()
        self.codeToRawCodeList()
        self.tokenizeCodes()
        self.encodedWordsCount()
        self.provideNgramSequences()
        self.nGramcodeSeqPadding()
        self.provideModelInputOutPut()
        # load saved model
        if self.use_saved_model == True:
            self.loadSavedModel()
        else:
            self.TrainLSTMModel()

        predictionList = []
        seq_length = self.seqx_length
        for i in range(n_lines):
            text = []
            for _ in range(self.predict_code_length):
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
        self.encoded_codes = self.token.texts_to_sequences(self.rawCodeList)
        return self.encoded_codes

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
        ngram_codeList = []

        for d in self.encoded_codes:
            if len(d) > 1:
                for i in range(2, len(d)):
                    ngram_codeList.append(d[:i])
        #                     print (d[:i])
        self.ngram_codeList = ngram_codeList
        return ngram_codeList;

    # provide padding input for ML or DL algorithms
    def nGramcodeSeqPadding(self):
        self.padded_code_ngram_sequences = pad_sequences(self.ngram_codeList, maxlen=self.max_length_padding, padding="pre")
        return self.padded_code_ngram_sequences

    def provideModelInputOutPut(self):
        self.X = self.padded_code_ngram_sequences[:, :-1]
        self.y = self.padded_code_ngram_sequences[:, -1]
        # categorize y
        self.code_vocab_size = self.getAllCodesWordSize()
        self.y = to_categorical(self.y, num_classes=self.code_vocab_size)
        # X shape
        self.seqx_length = self.X.shape[1]

    def TrainLSTMModel(self):

        self.modelLSTM = Sequential()
        self.modelLSTM.add(Embedding(self.code_vocab_size, 50, input_length=self.seqx_length))
        self.modelLSTM.add(LSTM(100, return_sequences=True))
        self.modelLSTM.add(LSTM(100))
        self.modelLSTM.add(Dense(100, activation="relu"))
        self.modelLSTM.add(Dense(self.code_vocab_size, activation="softmax"))
        # self.modelLSTM.summary()
        self.modelLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        self.modelLSTM.fit(self.X, self.y, batch_size=self.batch_size, epochs=self.epochs)
        here = os.path.abspath(os.path.dirname(__file__))
        self.modelLSTM.save(os.path.join(here + "/trained_models/", 'LSTM_code_generation_model'))
