#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.


from typing import List, Tuple

from smartclide_service_classification_autocomplete import AutocompleteCodeModel


class CodeMarkovSuggest:
    
    def __init__(self):
        self.model = AutocompleteCodeModel()

    def predict(self, method:str, language:str, code_input:str, code_sugg_len:int, code_sugg_lines:int) -> Tuple[List[str], int, int, str, str]:

        # predict
        result = self.model.generateCode(code_input, code_sugg_len, code_sugg_lines)

        # format results
        code_suggestions = [result['result']['code_sugg1'], result['result']['code_sugg2']]

        code = result['result']['code_sugg']
        code_len = result['result']['codeSuggLen']
        code_lines = result['result']['codeSuggLines']
        method = result['result']['Method']
        lang = result['result']['language']

        return code, code_len, code_lines, method, lang
