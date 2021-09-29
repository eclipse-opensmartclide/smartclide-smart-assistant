# !/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.

from .CodeGeneration import *
from flask import jsonify

class AutocompleteCodeModel():
     def generateCode(self, codeInput, codeSuggLen, codeSuggLines, method="Default",language="java"):
         error=''
         generated_code_arr = []
         if len(codeInput) > 2:
             if method == 'GPT2':
                 codeGenObj = CodeGenerationModel(True)
                 generated_code_arr = codeGenObj.generateCodeByGPT2(codeInput, int(codeSuggLines), int(codeSuggLen))
                 if not generated_code_arr:
                     error = 'Training need more resource'
             if method == 'Default':
                 codeGenObj = CodeGenerationModel(True)
                 generated_code_arr = codeGenObj.generate_code(codeInput, int(codeSuggLines))
             # results = []
             if not generated_code_arr:
                 result = {
                     "Error": error,
                 }
             else:
                 result = {
                     "code_sugg1": generated_code_arr[0],
                     "code_sugg2": generated_code_arr[1],
                     "Method": method,
                     "language": language
                 }

             # results.append(result)
             return ({'result': result})

