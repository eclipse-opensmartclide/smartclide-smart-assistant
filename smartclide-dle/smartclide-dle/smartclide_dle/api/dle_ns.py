#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.


import xml
import flask
import datetime
import pandas as pd
from flask_restx import Resource
from urllib.parse import unquote

from smartclide_dle.api.v1 import api
from smartclide_dle.core import cache, limiter
from smartclide_dle.utils import handle400error, handle404error, handle500error

from smartclide_dle.api.dle_models import enviroment_model, enviroment_prediction_model, \
acceptance_model, acceptance_prediction_model, developer_model, developer_prediction_model, \
codegen_model, codegen_prediction_model, service_classification_model, service_classification_prediction_model, \
code_generation_templates_model, code_generation_templates_prediction_model, bpmnitemrecommendation_model, \
bpmnitemrecommendation_prediction_model


from smartclide_dle.dle import ResourceSuggest, DeveloperSuggest, CodeMarkovSuggest, GherkinSuggest, \
ServiceClassification, CodeTemplateGeneration, BPMNItemRecommender


dle_ns = api.namespace('dle', description='Provides the DLE funcionality (Deep Learning Engine) predictions')


model_codegen = CodeMarkovSuggest()
model_service_classification = ServiceClassification()

@dle_ns.route('/serviceclassification')
class ServiceClassificationEndpoint(Resource):

    @api.expect(service_classification_model)
    @api.response(404, 'Data not found')
    @api.response(500, 'Unhandled errors')
    @api.response(400, 'Invalid parameters')
    @api.marshal_with(service_classification_prediction_model, code=200, description='OK', as_list=False)
    @limiter.limit('1000000/hour') 
    @cache.cached(timeout=1, query_string=True)
    def post(self):
        """
        Provides recommendations on service classification.
        """

        global model_service_classification

        model = model_service_classification

        # retrieve arguments
        try:
            obj = flask.request.get_json()
        except:
            return handle400error(dle_ns, 'The providen arguments are not correct. Please, check the swagger documentation at /v1')

        # check arguments
        if 'service_id' not in obj or 'service_name' not in obj or 'service_desc' not in obj:
            return handle400error(dle_ns, 'Missing parameters')

        service_id = obj['service_id'] 
        service_name = obj['service_name'] 
        service_desc = obj['service_desc']

        # perform prediction
        try:
            categories, method, _ = model.predict(service_id, service_name, service_desc)
        except:
            return handle500error(dle_ns, 'The DLE suffered an unexpected error, please retry in a few seconds.')

        # format and return results
        result = {
            'method': method,
            'service_id': service_id,
            'service_class': categories,
            'service_name': service_name
        }

        return result


@dle_ns.route('/templatecodegen')
class TemplateCodegenEndpoint(Resource):

    @api.expect(code_generation_templates_model)
    @api.response(404, 'Data not found')
    @api.response(500, 'Unhandled errors')
    @api.response(400, 'Invalid parameters')
    @api.marshal_with(code_generation_templates_prediction_model, code=200, description='OK', as_list=False)
    @limiter.limit('1000000/hour') 
    @cache.cached(timeout=1, query_string=True)
    def post(self):
        """
        Provides code generation via template.
        """

        # retrieve arguments
        try:
            obj = flask.request.get_json()
        except:
            return handle400error(dle_ns, 'The providen arguments are not correct. Please, check the swagger documentation at /v1')

        # check arguments
        if 'bpmn_file' not in obj:
            return handle400error(dle_ns, 'Missing parameters')

        bpmn_file = obj['bpmn_file'] 

        # prepare file and perform prediction
        try:
            bpmn_file = unquote(bpmn_file)
            bpmn_file = bpmn_file.replace('\n','')
            model = CodeTemplateGeneration()
            code_generated = list(model.generate(bpmn_file))
        except xml.etree.ElementTree.ParseError:
            return handle400error(dle_ns, 'Malformed BPMN file, please check the XML syntax.')
        except:
            return handle500error(dle_ns, 'The DLE suffered an unexpected error, please retry in a few seconds.')

        # format and return results
        result = {
            'code': code_generated
        }

        return result


@dle_ns.route('/predictivemodeltoolassistant')
class PredictiveModelToolAssistant(Resource):

    #@api.expect(predictivemodeltoolassistant_model)
    @api.response(404, 'Data not found')
    @api.response(500, 'Unhandled errors')
    @api.response(400, 'Invalid parameters')
    #@api.marshal_with(predictivemodeltoolassistant_prediction_model, code=200, description='OK', as_list=False)
    @limiter.limit('1000000/hour') 
    @cache.cached(timeout=1, query_string=True)
    def post(self):
        """
        Provides predictive model tool assistant predictions.
        """

        # retrieve arguments
        try:
            obj = flask.request.get_json()
        except:
            return handle400error(dle_ns, 'The providen arguments are not correct. Please, check the swagger documentation at /v1')

        # check arguments

        # perform prediction
        try:
            result = None
        except:
            return handle500error(dle_ns, 'The DLE suffered an unexpected error, please retry in a few seconds.')

        # format and return results

        return result


@dle_ns.route('/codegen')
class CodeSuggestEndpoint(Resource):

    @api.expect(codegen_model)
    @api.response(404, 'Data not found')
    @api.response(500, 'Unhandled errors')
    @api.response(400, 'Invalid parameters')
    @api.marshal_with(codegen_prediction_model, code=200, description='OK', as_list=False)
    @limiter.limit('1000000/hour') 
    @cache.cached(timeout=1, query_string=True)
    def post(self):
        """
        Provides recommendations on what JAVA code to write based on machine learning techniques.
        """

        global model_codegen

        model = model_codegen

        # retrieve arguments
        try:
            obj = flask.request.get_json()
        except:
            return handle400error(dle_ns, 'The providen arguments are not correct. Please, check the swagger documentation at /v1')

        # check arguments
        if 'code_input' not in obj or 'method' not in obj or 'language' not in obj or 'code_sugg_len' not in obj or 'code_sugg_lines' not in obj:
            return handle400error(dle_ns, 'Missing parameters')

        method = obj['method'] 
        language = obj['language'] 
        code_input = obj['code_input']
        code_sugg_len = obj['code_sugg_len'] 
        code_sugg_lines = obj['code_sugg_lines']

        # perform prediction
        try:
            code, code_len, code_lines, method, lang = model.predict(method, language, code_input, code_sugg_len, code_sugg_lines)
        except:
            return handle500error(dle_ns, 'The DLE suffered an unexpected error, please retry in a few seconds.')

        # format and return results
        result = {
            'method': method,
            'language': lang,
            'code_suggestions': code,
            'code_len': code_len,
            'code_lines': code_lines
        }

        return result


@dle_ns.route('/coderepo')
class DevSuggestEndpoint(Resource):

    @api.expect(developer_model)
    @api.response(404, 'Data not found')
    @api.response(500, 'Unhandled errors')
    @api.response(400, 'Invalid parameters')
    @api.marshal_with(developer_prediction_model, code=200, description='OK', as_list=False)
    @limiter.limit('1000000/hour') 
    @cache.cached(timeout=1, query_string=True)
    def post(self):
        """
        Performs reccomendations for the developer in order to commit in the repository's regular rithm.
        """

        # retrieve arguments
        try:
            obj = flask.request.get_json()
        except:
            return handle400error(dle_ns, 'The providen arguments are not correct. Please, check the swagger documentation at /v1')

        # check arguments

        required_arguments = ['header', 'repo_id','state','user','branch','time_since_last_commit','number_of_files_modified']

        for arg in required_arguments:
            if arg not in obj:
                return handle400error(dle_ns, f'The information is incomplete, missing {arg} parameter')

        header = obj['header']
        state = obj['state']
        repo_id = obj['repo_id']
        user = obj['user']
        branch = obj['branch']
        time_since_last_commit = obj['time_since_last_commit']
        number_of_files_modified = obj['number_of_files_modified']

        # perform prediction
        try:
            model = DeveloperSuggest()
            conviction_to_commit_from_files, conviction_to_commit_from_time = model.predict(number_of_files_modified, time_since_last_commit)
        except:
            return handle500error(dle_ns, 'The DLE suffered an unexpected error, please retry in a few seconds.')

        # format and return results
        result = {
            'header': header,
            'state': state,
            'repo_id': repo_id,
            'user': user,
            'branch': branch,
            'conviction_to_commit_from_files': conviction_to_commit_from_files,
            'conviction_to_commit_from_time': conviction_to_commit_from_time
        }

        return result


@dle_ns.route('/environment')
class ResourceSuggestEndpoint(Resource):

    @api.expect(enviroment_model)
    @api.response(404, 'Data not found')
    @api.response(500, 'Unhandled errors')
    @api.response(400, 'Invalid parameters')
    @api.marshal_with(enviroment_prediction_model, code=200, description='OK', as_list=False)
    @limiter.limit('1000000/hour') 
    @cache.cached(timeout=1, query_string=True)
    def post(self):
        """
        It makes recommendations of the amount of minimum requirements for a machine depending on the service to be executed in matter of CPU, RAM memory, disk, etc.
        """

        # retrieve arguments
        try:
            obj = flask.request.get_json()
        except:
            return handle400error(dle_ns, 'The providen arguments are not correct. Please, check the swagger documentation at /v1')

        # check arguments
        if 'service' not in obj or 'current_memory' not in obj or 'current_space' not in obj or 'current_user_volume' not in obj or 'current_number_of_threads' not in obj: 
            return handle400error(dle_ns, 'The providen service information is incomplete')
        
        if 'id' not in obj['service'] or 'name' not in obj['service']:
            return handle400error(dle_ns, 'The providen service information is incomplete')

        service_id = obj['service']['id']
        service_name = obj['service']['name']
        current_memory = obj['current_memory']
        current_space = obj['current_space']
        current_user_volume = obj['current_user_volume']
        current_number_of_threads = obj['current_number_of_threads']

        # perform prediction
        try:
            model = ResourceSuggest()
            predictions = list(model.predict(ram=current_memory, disk=current_space, \
                num_thread=current_number_of_threads, initial_user_volume=current_user_volume))
        except:
            return handle500error(dle_ns, 'The DLE suffered an unexpected error, please retry in a few seconds.')

        # format and return results
        result = {
            "header": "resource_used",
            "service": {
                'id': service_id,
                'name': service_name
            },
            "environments": predictions
        }

        return result


@dle_ns.route('/acceptance')
class GherkinAcceptanceSuggestEndpoint(Resource):

    @api.expect(acceptance_model)
    @api.response(404, 'Data not found')
    @api.response(500, 'Unhandled errors')
    @api.response(400, 'Invalid parameters')
    @api.marshal_with(acceptance_prediction_model, code=200, description='OK', as_list=False)
    @limiter.limit('1000000/hour') 
    @cache.cached(timeout=1, query_string=True)
    def post(self):
        """
        Provides the reccomended Gherkin acceptance test in relation with a concrete BMPN file. For that purpose looks for similar BPMN files in the internal BPMN project repository and when the reccomendation is performed, the given BPMN file is stored to increment the number of BPMN examples.
        """

        # retrieve arguments
        try:
            obj = flask.request.get_json()
        except:
            return handle400error(dle_ns, 'The providen arguments are not correct. Please, check the swagger documentation at /v1')

        # check arguments
        if 'bpmn' not in obj:
            return handle400error(dle_ns, 'No BPMN provided')

        bpmn = obj['bpmn']

        # perform prediction
        try:
            bpmn = unquote(bpmn)
            bpmn = bpmn.replace('\n','')
            model = GherkinSuggest()
            suggestions = model.predict(bpmn)
        except:
            return handle500error(dle_ns, 'The DLE suffered an unexpected error, please retry in a few seconds.')

        # format and return results
        result = {
            'recommended_gherkins': suggestions
        }

        return result

        
@dle_ns.route('/bpmnitemrecommendation')
class BPMNItemRecommendation(Resource):

    @api.expect(bpmnitemrecommendation_model)
    @api.response(404, 'Data not found')
    @api.response(500, 'Unhandled errors')
    @api.response(400, 'Invalid parameters')
    @api.marshal_with(bpmnitemrecommendation_prediction_model, code=200, description='OK', as_list=False)
    @limiter.limit('1000000/hour') 
    @cache.cached(timeout=1, query_string=True)
    def post(self):
        """
        Provides the reccomended item in a BPMN diagram using NLP techniques.
        """

        # retrieve arguments
        try:
            obj = flask.request.get_json()
        except:
            return handle400error(dle_ns, 'The providen arguments are not correct. Please, check the swagger documentation at /v1')

        # check arguments
        if 'bpmn' not in obj:
            return handle400error(dle_ns, 'Missing parameters')

        bpmn = obj['bpmn'] 

        # prepare file and perform prediction
        try:
            bpmn = unquote(bpmn)
            bpmn = bpmn.replace('\n','')
            model = BPMNItemRecommender()
            recommended_services = model.predict(bpmn)
        except xml.etree.ElementTree.ParseError:
            return handle400error(dle_ns, 'Malformed BPMN file, please check the XML syntax.')
        except:
            return handle500error(dle_ns, 'The DLE suffered an unexpected error, please retry in a few seconds.')

        # format and return results
        result = {
            'recommended_services': recommended_services
        }

        return result



