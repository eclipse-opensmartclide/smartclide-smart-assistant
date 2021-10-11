from smartclide_service_classification_autocomplete.AutocompleteCode import AutocompleteCodeModel

codeInput="import java."
codeSuggLen=6
codeSuggLines=1
# method="GPT2"
method="GPT"
genCodeObj = AutocompleteCodeModel()
pred = genCodeObj.generateCode(codeInput,codeSuggLen,codeSuggLines,method)
print(pred)

