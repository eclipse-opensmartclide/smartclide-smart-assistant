#!/usr/bin/python3
#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************



from typing import List, Tuple

from servcodegen import AutocompleteCodeModel


class CodeMarkovSuggest:
    
    def __init__(self):
        self.model = AutocompleteCodeModel()

    def predict(self, method:str, language:str, code_input:str, code_sugg_len:int, code_sugg_lines:int) -> Tuple[List[str], int, int, str, str]:

        # predict
        result = self.model.generateCode(code_input, code_sugg_len, code_sugg_lines)

        # format results
        code = result['result']['code_sugg']
        code_len = result['result']['codeSuggLen']
        code_lines = result['result']['codeSuggLines']
        method = result['result']['Method']
        lang = result['result']['language']

        return code, code_len, code_lines, method, lang
