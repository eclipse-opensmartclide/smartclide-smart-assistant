#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.


from typing import List

from smartclide_service_classification_autocomplete import AutocompleteCodeModel


class CodeMarkovSuggest:

    def predict(self, method:str, language:str, code_input:str, code_sugg_len:int, code_sugg_lines:int) -> List[str]:

        # predict
        code_autocomplete = AutocompleteCodeModel()
        result = code_autocomplete.generateCode(code_input, code_sugg_len, code_sugg_lines)

        # format results
        code_suggestions = [result['result']['code_sugg1'], result['result']['code_sugg2']]

        return code_suggestions