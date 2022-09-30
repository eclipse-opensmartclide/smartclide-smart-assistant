#!/usr/bin/python3
# Eclipse Public License 2.0

import os
import csv
import torch
import logging
import pandas as pd
from .AIPipelineConfiguration import *
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

MAX_LENGTH = int(10000)
logger = logging.getLogger(__name__)


# import warnings
# warnings.filterwarnings("error")

class CodesDataset(Dataset):

    def __init__(self, code_dataset_path):
        """Creates a torch Dataset with opensource codes ."""
        super().__init__()
        self.code_list = []
        self.end_of_text_token = "<|endoftext|>"
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
    tokenizer = None
    max_codeSuggLen = 40
    predict_code_length = 3
    max_line_return = 4
    df = pd.DataFrame()
    generatedCodesList = []
    custom_dataset_line = 2000
    transformer_model_name = 'gpt2'
    stop_token = '<|endoftext|>'
    trained_model = 'gpt2_codegenerator_trained.pt'
    spechialtokens = ['<|endoftext|>']
    trained_model_distilGPT2 = 'api_distilGPT2_denoise'
    trained_model_distilGPT2_folder = 'api_distilGPT2_denoise/'

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
                 epochs=3,
                 batch_size=16,
                 seqx_length=0,
                 code_vocab_size=0,
                 temperature_GPT=0.6,
                 top_k_GPT=50,
                 top_p_GPT=0.85,
                 repetition_penalty_GPT=1.0
                 ):
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.device = 'cpu'
        else:
            self.device = 'cpu'

        if default_dataset_name == '':
            self.default_dataset_name = self.defaultCodeTrainDataset

        if temperature_GPT > 0.8 or top_k_GPT > 40:
            logger.info("Code Generation Model: lower temperatures or top_k is recommended.")
        self.top_k_GPT = top_k_GPT
        self.top_p_GPT = top_p_GPT
        self.temperature_GPT = temperature_GPT

        if batch_size > 32 or epochs > 5:
            logger.info("Code Generation Model: 3-5  epoch and 16/32 batch_size recommended.")
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
        from servcodegen import getPackagePath
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

    def loadDataset(self):
        code_dataset_path = self.getDataDirectory() + self.defaultCodeTrainDataset
        self.codeLoder = DataLoader(CodesDataset(code_dataset_path), batch_size=1, shuffle=False)
        return self.codeLoder

    def trainGenerator(self, model_type=""):

        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        from transformers import AdamW

        if model_type == "":
            model_type = self.transformer_model_name
            print(model_type)

        try:
            model_type = model_type.lower()
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
        models_folder = self.getTrainedModelsDirectory()
        if not os.path.exists(models_folder):
            os.mkdir(models_folder)

        epoch = self.epoch
        print(f"EPOCH {epoch}" + '+' * 30)
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

    def getTrainedModel_distilGPT2(self):
        """
        Function inorder to return deafult transformer model
        :param clmName: string param specifies trained model name
        """
        return (self.trained_model_distilGPT2)

    def loadTrainedGenerator(self):
        """
        Load trained web service code generator
        """

        trained_models_folder = self.getTrainedModelsDirectory()
        trained_models_files = self.getTrainedModel_distilGPT2()
        model_path = os.path.join(trained_models_folder, trained_models_files)

        # print(trained_models_folder)
        # print(model_path)
        if not (os.path.exists(model_path)):
            print("+" * 40)
            logger.error("No trained model file is exist.")
            print("No trained model file is exist.")
            print("+" * 40)
            self.generatorModel = None
            return (self.generatorModel)

        try:
            self.tokenizer_class = GPT2Tokenizer.from_pretrained('gpt2')
            self.generatorModel = GPT2LMHeadModel.from_pretrained(model_path,
                                                                  pad_token_id=self.tokenizer_class.eos_token_id)

            # self.tokenizer_class = GPT2Tokenizer.from_pretrained(self.getTransformerModelName())
            # self.generatorModel = GPT2LMHeadModel.from_pretrained(self.getTransformerModelName())
            # self.generatorModel.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print("+" * 40)
            print("Trained model is loaded on " + self.device)
            print("+" * 40)

        except FileNotFoundError:
            self.generatorModel = None  # as deafult GPT will run which we dont need its result
            print("+" * 40)
            print("There is a problem with correctly loading a trained model.")
            print("+" * 40)

        return (self.generatorModel)

    def input_validation(self, seed_text, lengthCodeLine, max_line_return):
        import re
        result = {
            "code_input_error": False,
            "codeSuggLen_error": False,
            "codeSuggLines_error": False,
        }

        if len(seed_text) < 1 and len(seed_text) > 200:
            result["code_input_error"] = True
            logging.info("The models need input between 1-" + str(self.max_codeSuggLen) + " character ")

        if int(max_line_return) > self.max_line_return:
            result["codeSuggLines_error"] = True
            logging.info("The max_line parameter must be between 1-" + str(self.max_line_return))

        if int(lengthCodeLine) > self.max_codeSuggLen:
            result["codeSuggLen_error"] = True
            logging.info("The codeSuggLen parameter must be between 2-" + str(self.max_codeSuggLen))

        return result

    def generate_code_trainedGPT2(self, seed_text, lengthCodeLine, max_line_return):

        """
            Generate code using trained  web service code generator
            with optional lengthCodeLine and max_line_return
        """
        self.generatedCodesList = []
        self.seed_text = seed_text
        self.predict_code_length = lengthCodeLine
        self.max_line_return = max_line_return

        encoded_prompt = self.tokenizer_class.encode(self.seed_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        output_sequences = self.generatorModel.generate(
            input_ids=encoded_prompt,
            max_length=self.predict_code_length,
            # num_beams=self.max_line_return,
            do_sample=True,
            top_k=self.top_k_GPT,
            top_p=self.top_p_GPT,
            temperature=self.temperature_GPT,
            num_return_sequences=self.max_line_return,
        )

        text = None
        for output in output_sequences:
            generated_sequence = output.tolist()
            code_line = self.tokenizer_class.decode(generated_sequence, clean_up_tokenization_spaces=True,
                                                    skip_special_tokens=True)
            code_line = code_line[: code_line.find(self.stop_token) if self.stop_token == 1 else None]
            self.generatedCodesList.append(self.post_process_code(code_line))
            self.generatedCodesList = list(dict.fromkeys(self.generatedCodesList))
        return self.generatedCodesList

    def post_process_code(self, code):
        import re

        # #ToDO remove duplicate
        # if (code.find("<|endcode|>") != -1):
        #     self.stop_token = "<|endcode|>"
        # else:
        #     self.stop_token = ";"

        if len(code) > 2:
            code_post_process = code.replace(". ", ".")
            return code_post_process

        return code

        # if len(code_line)  > 2:
        #     code_post_process = code_line.replace("<STRING>", '""')
        #     return code_post_process
        # else:
        #     return code_line
        # return code
