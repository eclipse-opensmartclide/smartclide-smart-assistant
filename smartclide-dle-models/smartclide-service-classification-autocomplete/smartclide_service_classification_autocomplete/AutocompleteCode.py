#!/usr/bin/python3
# Eclipse Public License 2.0

from .CodeGeneration import *
from flask import jsonify

class AutocompleteCodeModel():
    """
    These trained models need a gateway between the trained models and user interfaces.
    This Class provide interface for models and DLE apis
    """
    
    def __init__(self):
        codeGenObj = CodeGenerationModel(True)
        self.generator = codeGenObj.loadGenerator()
        
    def getCodeGenerator(self):
        return self.generator

    def generateCode(self, codeInput, codeSuggLen, codeSuggLines=1, method="Default", language="java"):
        error = ''
        generatedCode = ''
        if len(codeInput) > 2:
            if method == 'GPT' or method == 'Default':
                generated_code_arr = self.generator(codeInput, max_length=codeSuggLen, do_sample=True, temperature=0.9)
                generatedCode = generated_code_arr[0]['generated_text']
            if method == 'GPT-2' :
                autocomplete = CodeGenerationModel()
                autocomplete.loadGPT2Generator() 
                autocomplete.generateCodeByTrainedGPT2("Code:"+codeInput)
    
        result = {
            "code_sugg1": generatedCode,
            "code_sugg2":'',
            "Method": method,
            "language": language
        }
    
        return ({'result': result})
