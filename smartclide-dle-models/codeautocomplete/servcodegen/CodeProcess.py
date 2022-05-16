#!/usr/bin/python3
# Eclipse Public License 2.0

import re
import nltk
import glob
import numpy as np
import os, fnmatch
import pandas as pd
from typing import List
from pandas.core.frame import DataFrame


class FileNotFoundError(Exception):
    def __init__(self, path, message="no such file or directory but file exists"):
        self.path = path
        self.message = message
        super().__init__(self.message)


class CodeProcess:
    def __init__(self,
                 min_code_length: int = 4,
                 max_code_length: int = 500,
                 codeline_splitter: str = ";",
                 lang: str = None,
                 string_regex: List = None,
                 comment_regex: List = None,
                 lexical_placeholder: List = None):
        self.min_code_length = min_code_length
        self.max_code_length = max_code_length
        self.codeline_splitter = '\n'
        self.lexical_placeholder = {
            "general": "",
            "comment": "<COMMENT>",
            'spaces': '',  # TODO
            "string": "<STRING>",
            "string_regx": "<STRING_REGX>",
            "int": "<INT>",
            "char": "<CHAR>"  # inside single  qoute
        }

    def get_package_path(self) -> str:
        from servcodegen import getPackagePath
        package_root_path = getPackagePath()
        return (package_root_path)

    def is_project_path_exist(self, path: str) -> bool:
        isfile = os.path.exists(path)
        return isfile


class CodePreProcess(CodeProcess):
    files_count = 0
    list_of_code_lists = []

    def __init__(self,
                 target_clmn='codes',
                 target_clean_clmn='clean_codes',
                 target_clean_clmn_len='clean_codes_len',
                 min_code_length: int = 3,
                 max_code_length: int = 5000,
                 codeline_splitter=';',
                 lang='java',
                 string_regex: List = None,
                 comment_regex: List = None,
                 lexical_placeholder: List = None):

        super().__init__(min_code_length, max_code_length, codeline_splitter, lang, string_regex, comment_regex,
                         lexical_placeholder)
        self.target_clmn = target_clmn
        self.target_clean_clmn = target_clean_clmn
        self.target_clean_clmn_len = target_clean_clmn_len
        self.max_code_length = max_code_length

    def file_preprocess(self, file_path: str) -> str:

        if self.is_project_path_exist(file_path) != True:
            raise FileNotFoundError(file_path)

        try:
            # countries_str =open(os.path.join(args.base_dir, file_name)).readlines()
            countries_str = open(file_path).readlines()
            print(countries_str)

        except Exception:
            pass
            # raise FileNotFoundError(project_path)

    def find_directory_files(self, directory: str, file_ext_pattern: str) -> list:
        for root, dirs, files in os.walk(directory):
            for basename in files:
                if fnmatch.fnmatch(basename, file_ext_pattern):
                    filename = os.path.join(root, basename)
                    print(filename)
                    yield filename

    def find_project_files(self, project_path, file_ext_pattern='*.java') -> list:
        i = 0
        try:
            for filename in self.find_directory_files(project_path, file_ext_pattern):
                with open(filename) as f:
                    i = i + 1
                    for line in f:
                        inner_list = [elt.strip() for elt in line.split(',')]
                        if (len(inner_list[0]) > self.min_code_length):
                            self.list_of_code_lists.append(inner_list)
        except Exception:
            pass
        self.files_count = i
        return self.list_of_code_lists

    def post_process_pred_lines(self, pred_lines: list) -> list:
        pred_clean = []
        for li in pred_lines:
            code = pred = self.code_post_process(li)
            pred_clean.append(code)
        return pred_clean

    def get_project_code_lines(self, project_path, file_ext_pattern='*.java') -> DataFrame:

        if self.is_project_path_exist(project_path) != True:
            raise FileNotFoundError

        self.find_project_files(project_path, file_ext_pattern)
        # Convert List of lists to list of Strings
        list_all_code_lines = [''.join(ele) for ele in self.list_of_code_lists]
        self.df = pd.DataFrame(list_all_code_lines, columns=[self.target_clmn])
        return self.df

    def replace_placeholder(self, inputLine,pattern_com=r'<COMMENT>|.*?<COMMENT>') -> DataFrame:
        placeholder = self.lexical_placeholder['general']
        filteredInput = re.sub(pattern_com, placeholder, inputLine)    
        return filteredInput

    def replace_placeholder_dataset(self, target_clmn, des_clmn) -> DataFrame:
        self.df[des_clmn] = self.df[target_clmn].astype(str).apply(self.replace_placeholder)
        return self.df

    def remove_low_length_lines(self, N_low=5) -> DataFrame:
        self.df[self.target_clean_clmn_len] = self.df[self.target_clean_clmn].str.len()
        self.df[self.target_clean_clmn_len] = self.df[self.target_clean_clmn_len].astype('Int64')
        self.df = self.df[~(self.df[self.target_clean_clmn_len] < N_low)]
        return self.df

    def remove_high_length_lines(self, N_high=130) -> DataFrame:
        self.df[self.target_clean_clmn_len] = self.df[self.target_clean_clmn].str.len()
        self.df[self.target_clean_clmn_len] = self.df[self.target_clean_clmn_len].astype('Int64')
        self.df = self.df[~(self.df[self.target_clean_clmn_len] > N_high)]
        return self.df
