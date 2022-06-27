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

from .CodeGeneration import *
from flask import jsonify

class AutocompleteCodeModel():
    generator=None
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

            if method == 'GPT-2' or method == 'GPT' or method == 'Default':
                if self.generator.generatorModel is not None:
                    generatedCodeList=self.generator.generate_code_trainedGPT2(codeInput,codeSuggLen,codeSuggLines)
                else:
                    generatedCodeList.append("Make sure that trained model that is loaded and accessible")



    
        result = {
            "code_sugg": generatedCodeList,
            "Method": method,
            "codeSuggLen": codeSuggLen,
            "codeSuggLines": codeSuggLines,
            "language": language
        }
    
        return ({'result': result})
