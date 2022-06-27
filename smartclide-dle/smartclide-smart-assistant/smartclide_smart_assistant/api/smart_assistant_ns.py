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



import uuid
import flask
import datetime
import requests
import pandas as pd
from flask import redirect
from flask_restx import Resource
from urllib.parse import unquote

import pycbr

from smartclide_smart_assistant import config
from smartclide_smart_assistant.api.v1 import api
from smartclide_smart_assistant.core import cache, limiter
from smartclide_smart_assistant.utils import handle400error, handle404error, handle500error

from smartclide_smart_assistant.db_model import DBModel
from smartclide_smart_assistant.api.smart_assistant_models import enviroment_model, enviroment_prediction_model, \
acceptance_model, acceptance_prediction_model, developer_model, developer_prediction_model, \
codegen_model, codegen_prediction_model, service_classification_model, service_classification_prediction_model, \
code_generation_templates_model, bpmnitemrecommendation_model, bpmnitemrecommendation_prediction_model




smart_assistant_ns = api.namespace('smartassistant', description='Provides the Smart Assistant funcionality, based on the DLE (Deep Learning Engine) predictions')


db_model = DBModel()


@smart_assistant_ns.route('/codegen')
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
        Provides recommendations on what JAVA code to write based on machine learning techniques and markov chains.
        """

        # retrieve arguments
        try:
            obj = flask.request.get_json()
        except:
            return handle400error(smart_assistant_ns, 'The providen arguments are not correct. Please, check the swagger documentation at /v1')

        # check arguments
        if 'code_input' not in obj or 'method' not in obj or 'language' not in obj or 'code_sugg_len' not in obj or 'code_sugg_lines' not in obj:
            return handle400error(smart_assistant_ns, 'No code, line or cursor providen')

        # perform prediction
        try:
            result = requests.post(url=f'{config.DLE_BASE_URL}/codegen',json=obj)
            result = result.json()
        except:
            return handle500error(smart_assistant_ns, 'The DLE suffered an unexpected error, please retry in a few seconds.')

        # format and return results

        return result


@smart_assistant_ns.route('/coderepo')
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

        global db_model

        # retrieve arguments
        try:
            obj = flask.request.get_json()
        except:
            return handle400error(smart_assistant_ns, 'The providen arguments are not correct. Please, check the swagger documentation at /v1')

        # check arguments

        required_arguments = ['header', 'repo_id','state','user','branch','time_since_last_commit','number_of_files_modified']

        for arg in required_arguments:
            if arg not in obj:
                return handle400error(smart_assistant_ns, f'The information is incomplete, missing {arg} parameter')

        allowed_headers = ['new commit', 'new file changed']
        if obj['header'] not in allowed_headers:
            return handle400error(smart_assistant_ns, f'The providen header not in allowed headers {allowed_headers}')

        header = obj['header']
        state = obj['state']
        repo_id = obj['repo_id']
        user = obj['user']
        branch = obj['branch']
        time_since_last_commit = obj['time_since_last_commit']
        number_of_files_modified = obj['number_of_files_modified']

        # store if headers specifies that else request info to DLE
        if header == 'new commit':
            try:
                db_model.store('developer', f'{repo_id}.{branch}.{user}', {
                    'time_since_last_commit': time_since_last_commit,
                    'number_of_files_modified': number_of_files_modified
                })
            except:
                return handle500error(smart_assistant_ns, 'The Smart Assistant suffered an unexpected error, please retry in a few seconds.')

        else:
            try:
                results = db_model.load('developer', f'{repo_id}.{branch}.{user}')
            except:
                return handle500error(smart_assistant_ns, 'The Smart Assistant suffered an unexpected error, please retry in a few seconds.')

            if not results:
                return handle404error(smart_assistant_ns, 'The Smart Assistant not found any entries with the given id, for the requested calculation')

            try:
                obj = {
                    "header": header
                    ,"state": state
                    ,"repo_id": repo_id
                    ,"user": user
                    ,"branch": branch
                    ,"time_since_last_commit": max(r["time_since_last_commit"] if r is not None and r["time_since_last_commit"] is not None else 0 for r in results)
                    ,"number_of_files_modified": max(r["number_of_files_modified"] if r is not None and r["number_of_files_modified"] is not None else 0 for r in results)
                }
                result = requests.post(url=f'{config.DLE_BASE_URL}/coderepo',json=obj)
                result = result.json()
                result['messages'] = [
                    f'You have {obj["number_of_files_modified"]} modified files without committing. This exceeds {round(result["conviction_to_commit_from_files"]*100,2)}% of commits registered by the system. We recommend that you make a commit to persist the information in the code repository',
                    f'You have not commited in the last {obj["time_since_last_commit"]} seconds. This exceeds {round(result["conviction_to_commit_from_time"]*100,2)}% of commits registered by the system. We recommend that you make a commit to persist the information in the code repository'
                ]

                return result
            except:
                return handle500error(smart_assistant_ns, 'The Smart Assistant suffered an unexpected error, please retry in a few seconds.')


@smart_assistant_ns.route('/enviroment')
class ResourceSuggestEndpointX(Resource):

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

        global db_model

        # retrieve arguments
        try:
            obj = flask.request.get_json()
        except:
            return handle400error(smart_assistant_ns, 'The providen arguments are not correct. Please, check the swagger documentation at /v1')

        # check arguments
        if 'header' not in obj or 'service' not in obj or 'current_memory' not in obj or 'current_space' not in obj or 'current_user_volume' not in obj or 'current_number_of_threads' not in obj: 
            return handle400error(smart_assistant_ns, 'The providen service information is incomplete')
        
        if 'id' not in obj['service'] or 'name' not in obj['service']:
            return handle400error(smart_assistant_ns, 'The providen service information is incomplete')

        allowed_headers = ['resource used', 'start deploy']
        if obj['header'] not in allowed_headers:
            return handle400error(smart_assistant_ns, f'The providen header not in allowed headers {allowed_headers}')

        header = obj['header']
        service_id = obj['service']['id']
        service_name = obj['service']['name']
        current_memory = obj['current_memory']
        current_space = obj['current_space']
        current_user_volume = obj['current_user_volume']
        current_number_of_threads = obj['current_number_of_threads']

        # store if headers specifies that else request info to DLE
        if header == 'start deploy':
            try:
                db_model.store('enviroment', service_id, {
                    'current_memory': current_memory,
                    'current_space': current_space,
                    'current_user_volume': current_user_volume,
                    'current_number_of_threads': current_number_of_threads
                })
            except:
                return handle500error(smart_assistant_ns, 'The Smart Assistant suffered an unexpected error, please retry in a few seconds.')
        else:
            try:
                results = db_model.load('enviroment', service_id)
            except:
                return handle500error(smart_assistant_ns, 'The Smart Assistant suffered an unexpected error, please retry in a few seconds.')

            if not results:
                return handle404error(smart_assistant_ns, 'The Smart Assistant not found any entries with the given id, for the requested calculation')

            try:
                avg = lambda l: sum(l)/(len(l) if l else 1)
                obj = {
                    "header": header
                    ,"service": {
                        'id': service_id,
                        'name': service_name
                    }
                    ,"current_memory": avg([r["current_memory"] if r is not None and r['current_memory'] is not None else 0 for r in results])
                    ,"current_space": avg([r["current_space"] if r is not None and r['current_space'] is not None else 0 for r in results])
                    ,"current_user_volume": avg([r["current_user_volume"] if r is not None and r['current_user_volume'] is not None else 0 for r in results])
                    ,"current_number_of_threads": avg([r["current_number_of_threads"] if r is not None and r['current_number_of_threads'] is not None else 0 for r in results])
                }
                result = requests.post(url=f'{config.DLE_BASE_URL}/environment',json=obj)
                return result.json()
            except:
                return handle500error(smart_assistant_ns, 'The Smart Assistant suffered an unexpected error, please retry in a few seconds.')


@smart_assistant_ns.route('/acceptance')
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
            return handle400error(smart_assistant_ns, 'The providen arguments are not correct. Please, check the swagger documentation at /v1')

        # check arguments
        if 'bpmn' not in obj:
            return handle400error(smart_assistant_ns, 'No BPMN provided')

        bpmn = obj['bpmn']
        name = obj['name'] if 'name' in obj else str(uuid.uuid4())
        gherkins = obj['gherkins'] if 'gherkins' in obj else None

        # fix possible bad characters
        try:
            bpmn = unquote(bpmn)
            bpmn = bpmn.replace('\n','')
        except:
            pass

        try:
            gherkins_ = []
            for g in gherkins:
                try:
                    g = unquote(g)
                    g = g.replace('\n','')
                    gherkins_.append(g)
                except:
                    pass
            gherkins = gherkins_
        except:
            pass

        if name is not None and gherkins is not None:
            try:
                cbr = pycbr.CBR([],"ghrkn_recommendator", config.MONGO_IP)
                cbr.add_case(
                    {
                        'name': name,
                        'text': bpmn,
                        'gherkins': gherkins
                    }
                )
            except:
                return handle500error(smart_assistant_ns, 'The Smart Assistant suffered an unexpected error, please retry in a few seconds.')
        else:
            try:
                obj = {
                    "bpmn": bpmn
                }
                result = requests.post(url=f'{config.DLE_BASE_URL}/acceptance',json=obj)
                return result.json()
            except:
                return handle500error(smart_assistant_ns, 'The Smart Assistant suffered an unexpected error, please retry in a few seconds.')


@smart_assistant_ns.route('/bpmnitemrecommendation')
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
            return handle400error(smart_assistant_ns, 'The providen arguments are not correct. Please, check the swagger documentation at /v1')

        # check arguments
        if 'bpmn' not in obj:
            return handle400error(dle_ns, 'Missing parameters')

        # perform prediction
        try:
            result = requests.post(url=f'{config.DLE_BASE_URL}/bpmnitemrecommendation',json=obj)
            return result.json()
        except:
            return handle500error(smart_assistant_ns, 'The DLE suffered an unexpected error, please retry in a few seconds.')
