# !/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.

from .CodeGeneration import *
from flask import jsonify


class AutocompleteCodeModel():

    def __init__(self):
        codeGenObj = CodeGenerationModel(True)
        self.generator = codeGenObj.loadGenerator()
        
    def getCodeGenerator(self):
        return self.generator

    def generateCode(self, codeInput, codeSuggLen, codeSuggLines=1, method="Default", language="java"):
        error = ''
        generated_code_arr = ''
        if len(codeInput) > 2:
            if method == 'Default':
                generated_code_arr = self.generator(codeInput, max_length=codeSuggLen, do_sample=True, temperature=0.9)
                generatedCode = generated_code_arr[0]['generated_text']
    
        result = {
            "code_sugg1": generatedCode,
            "Method": method,
            "language": language
        }
    
        return ({'result': result})
