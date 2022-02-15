#!/usr/bin/python3
# Eclipse Public License 2.0

from .CodeGeneration import *
from flask import jsonify

class AutocompleteCodeModel():

    def __init__(self):
        codeGenObj = CodeGenerationModel()
        codeGenObj.loadTrainedGenerator()
        self.generator = codeGenObj
        
#         codeGenObj = CodeGenerationModel()
        #ToDO remove this line after adding lgfs train file
#         codeGenObj.loadTrainedGenerator()
#         self.generator = codeGenObj

        
    def getCodeGenerator(self):
        return self.generator

    def generateCode(self, codeInput, codeSuggLen, codeSuggLines=1, method="Default", language="java"):
        error = ''
        generatedCodeList = []
        generatedCode = ''
        if len(codeInput) > 2:
            #ToDO remove this line after adding lgfs train file
            if method == 'GPT' or method == 'Default':
                generatedCodeList.append("Under develope,waiting for upload git lgfs file ...")

            if method == 'GPT-2' :
                generatedCodeList.append("Under develope,waiting for upload git lgfs file ...")
                # generatedCodeList=self.generator.generate_code_trainedGPT2(codeInput,codeSuggLen,codeSuggLines)

    
        result = {
            "code_sugg1": generatedCodeList,
            "code_sugg2": generatedCodeList,
            "Method": method,
            "codeSuggLen": codeSuggLen,
            "codeSuggLines": codeSuggLines,
            "language": language
        }
    
        return ({'result': result})
