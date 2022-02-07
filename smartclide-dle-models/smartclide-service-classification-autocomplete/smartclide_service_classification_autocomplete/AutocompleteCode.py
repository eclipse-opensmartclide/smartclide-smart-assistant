#!/usr/bin/python3
# Eclipse Public License 2.0

from .CodeGeneration import *
from flask import jsonify

class AutocompleteCodeModel():

    def __init__(self):
        self.level=1
#         codeGenObj = CodeGenerationModel()
        #ToDO remove this line after adding lgfs train file
#         codeGenObj.loadTrainedGenerator()
#         self.generator = codeGenObj

        
    def getCodeGenerator(self):
        return self.generator

    def generateCode(self, codeInput, codeSuggLen, codeSuggLines=1, method="Default", language="java"):
        error = ''
        generatedCode = ''
        if len(codeInput) > 2:
            #ToDO remove this line after adding lgfs train file
            if method == 'GPT' or method == 'Default':
                generatedCode[0]="Under develope,waiting for upload git lgfs file ..."

#             if method == 'GPT' or method == 'Default':
#                 generated_code_arr = self.generator(codeInput, max_length=codeSuggLen, do_sample=True, temperature=0.9)
#                 generatedCode = generated_code_arr[0]['generated_text']
#             if method == 'GPT-2' :
#                 generatedCode=self.generator.generate_code_trainedGPT2(codeInput,codeSuggLen,codeSuggLines)
    
        result = {
            "code_sugg1": generatedCode[0],
            "code_sugg2":'',
            "Method": method,
            "language": language
        }
    
        return ({'result': result})
