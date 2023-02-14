#!/usr/bin/python3
# Eclipse Public License 2.0

from .CodeGeneration import *

class AutocompleteCodeModel():
    generator=None
    def __init__(self):
        
        self.cnf_models= AIPipelineConfiguration()
        codeGenObj = CodeGenerationModel()
        self.generator = codeGenObj    
        if self.cnf_models.code_generation_load_model=="Enabled":
            codeGenObj.loadTrainedGenerator()
        

        
    def getCodeGenerator(self):
        return self.generator

    def generateCode(self, codeInput, codeSuggLen, codeSuggLines=1, method="Default", language="java"):
        error = ''
        generatedCodeList = ['']
        #remove input tail space to ignore wrong results
        codeInput = codeInput.rstrip()

        if (self.cnf_models.code_generation_load_model=="Disabled"):
            result = {"Error": "The \"AIPipelineConfiguration\"  class is configured for not loading the code generator model; Please use \"Enabled\" for loading the model in the AIPipelineConfiguration file to use the code generator."}
            return result
        
        result=self.generator.input_validation(codeInput,codeSuggLen,codeSuggLines)
        
        if result['code_input_error'] ==False:
           if result['codeSuggLen_error'] ==True:
              codeSuggLen= self.generator.max_codeSuggLen
           if result['codeSuggLines_error'] ==True:
              codeSuggLines=self.generator.max_line_return
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
