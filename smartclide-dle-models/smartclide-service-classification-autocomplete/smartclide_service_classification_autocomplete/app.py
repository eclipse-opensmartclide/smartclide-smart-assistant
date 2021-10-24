# !/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.
from configparser import ConfigParser


# !/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.

from smartclide_service_classification_autocomplete.CodeGeneration import CodeGenerationModel
from smartclide_service_classification_autocomplete.PredictServiceClass import PredictServiceClassModel


from flask import Flask, render_template, request
from flask import jsonify

app = Flask(__name__)
import pickle
global generator
codeGenObj = CodeGenerationModel(True)
generator=codeGenObj.loadGenerator()
            



@app.route("/")
def mail():
    return "SmartCLIDE"


@app.route('/dle/classify_service', methods=['POST', 'GET'])
def data():
    serviceDesc = ''
    if request.method == 'POST':
        json_data = request.json
        method = json_data["method"]
        serviceID = json_data["service_id"]
        serviceName = json_data["service_name"]
        serviceDesc = json_data["service_desc"]
    if len(serviceDesc) > 2:
        serviceObj = PredictServiceClassModel()
        results = serviceObj.predict(serviceName,serviceDesc,serviceID,method)
        return jsonify({'result': results})


@app.route('/dle/code_autocomplete', methods=['POST', 'GET'])
def codegen():
    codeInput = ''
    generated_code_arr = []
    error = ''
    if request.method == 'POST':
        json_data = request.json
        method = json_data["method"]
        language = json_data["language"]
        codeInput = json_data["code_input"]
        codeSuggLen = json_data["code_sugg_len"]
#         codeSuggLines = json_data["code_sugg_lines"]
        codeSuggLine=1

    if len(codeInput) > 2:
        if method == 'GPT':
                from app import generator
                generated_code_arr = generator(codeInput, max_length=4, do_sample=True, temperature=0.9) 
                generatedCode=generated_code_arr[0]['generated_text']
                return jsonify({'result': generatedCode})
            
        if method == 'GPT2':
                codeGenObj = CodeGeneration(True)
                generated_code_arr=codeGenObj.generateCodeByGPT2(codeInput,int(codeSuggLines),int(codeSuggLen))
                if not generated_code_arr:
                    error='Training need more resource'
        if method == 'Default':
            codeGenObj = CodeGeneration(True)
            generated_code_arr = codeGenObj.generate_code(codeInput, int(codeSuggLines))

        results = []
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

        results.append(result)
        return jsonify({'result': results})


if __name__ == "__main__":
    app.run('0.0.0.0', debug=True)  # default port is 5000

