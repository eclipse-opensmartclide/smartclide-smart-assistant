import requests
import pandas as pd
from typing import Tuple 
from typing import List
from smartclide_service_classification_autocomplete import AutocompleteCodeModel

class CodeCompletion:
    
    def __init__(self):
        self.model = AutocompleteCodeModel()

    
    def predict2(self, method:str, language:str, code_input:str, code_sugg_len:int, code_sugg_lines:int) -> List[str]:
        # predict
        result = self.model.generateCode(code_input, code_sugg_len, code_sugg_lines,method)
        return result



'''
Loading model recommended to execute on background
Make sure the gpt2_codegenerator_trained.pt is in trained models directory
'''
codecomplete_obj = CodeCompletion()


'''
max_lenth specify max lenth line suggestion, recommended value is between 15-20
max_sugges_line specify max line suggestion, recommended value is between 3-5
'''
Method="GPT-2"
lang="java"
max_lenth=15
max_sugges_line=3
code_input="import android."
result=codecomplete_obj.predict2(Method,lang,code_input,max_lenth,max_sugges_line)
print(result)

'''
OUTPUT:


{'result':
 {'code_sugg': [' code line 1','code line 2','code line 3'], #array of string which includes suggested code lines
  'Method': 'GPT-2', 
  'codeSuggLen': 15, 
  'codeSuggLines': 3,
   'language': 'java'
   }
}

Warning : if the model file is not avalible you will recive :'Under develope,waiting for upload git lgfs file ...' as output

'''