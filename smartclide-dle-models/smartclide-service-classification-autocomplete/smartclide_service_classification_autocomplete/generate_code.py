from smartclide_service_classification_autocomplete.AutocompleteCode import AutocompleteCodeModel

codeInput="import java.util.Arrays"
codeSuggLen=2
codeSuggLines=2
# method="GPT2"
method="Default"
genCodeObj = AutocompleteCodeModel()
pred = genCodeObj.generateCode(codeInput,codeSuggLen,codeSuggLines,method)
print(pred)

