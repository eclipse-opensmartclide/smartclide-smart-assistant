# !/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.
from configparser import ConfigParser


# !/usr/bin/env python
# coding: utf-8
# Copyright 2021 AIR Institute
# See LICENSE for details.

from configparser import ConfigParser
# Read config.ini file
config_object = ConfigParser()
config_object.read("config.ini")
# Get the _PATH_ROOT_
_PATH_ROOT_ = config_object["Path"]["root"]
print(_PATH_ROOT_)
import sys
sys.path.insert(1, _PATH_ROOT_ + 'utils')
sys.path.insert(1, _PATH_ROOT_ + 'models')
from ImportingModules import *
from ServiceClassification import *
from CodeGeneration import *
from flask import Flask, render_template, request
from flask import jsonify

app = Flask(__name__)


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
        if method == 'Fasttext':
            serviceObjFastText = ServiceClassification(True)
            serviceObjFastText.loadData()
            pred = serviceObjFastText.predictFastTextModel(serviceDesc)
            if not pred:
                    error='Training need more resource'
            else:
                    serviceClass = pred[0]
        if method == 'BSVM':
            serviceObjBSVM = ServiceClassification(True)
            serviceObjBSVM.loadData()
            pred = serviceObjBSVM.predictBSVMModel(serviceDesc)
            if not pred:
                    error='Training need more resource'
            else:
                    serviceClass = pred[0]
        if method == 'Default':
            serviceObjML = ServiceClassification(True)
            serviceObjML.loadData()
            pred = serviceObjML.predictBOWML(serviceDesc)
            if not pred:
                    error='Training need more resource'
            else:
                    serviceClass = pred[0]
        results = []

        if not  len(serviceClass) < 1 :
            result = {
                "Error": error,
            }
        else:
            result = {
                "service_id": serviceID,
                "service_name": serviceName,
                "Method": method,
                "Service_class": serviceClass
            }

        results.append(result)
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
        codeSuggLines = json_data["code_sugg_lines"]

    if len(codeInput) > 2:
        if method == 'CPT2':
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

