#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.

import os

# api config
PORT = os.getenv('SA_API_PORT', 5000) 
HOST = os.getenv('SA_API_BIND', '0.0.0.0')
URL_PREFIX = '/smartclide/v1'
DEBUG_MODE = True

# DLE configuration
DLE_BASE_URL = os.getenv('DLE_BASE_URL', 'http://smartclide.ddns.net:5001/smartclide/v1/dle') 
SMART_ASSISTANT_BASE_URL = os.getenv('SMART_ASSISTANT_BASE_URL', 'http://smartclide.ddns.net:5000/smartclide/v1/smartassistant') 

# mongodb configuration
MONGO_DB = os.getenv('SA_MONGODB_DB', 'smartclide-smart-assistant')
MONGO_IP = os.getenv('SA_MONGODB_HOST', 'localhost') 
MONGO_PORT = os.getenv('SA_MONGODB_PORT', 27017)
MONGO_URI = f'mongodb://{MONGO_IP}:{MONGO_PORT}/{MONGO_DB}'

# rabbitmq configuration
rabbitmq_host = os.getenv('RABBITMQ_HOST', 'localhost')
rabbitmq_port = int(os.getenv('RABBITMQ_PORT', 5672))
rabbitmq_user = os.getenv('RABBITMQ_USER', 'user')
rabbitmq_password = os.getenv('RABBITMQ_PASSWORD', 'password')
channel_endpoint_mappings = os.getenv('RABBITMQ_MAPPINGS',  
	{
	k: f'{SMART_ASSISTANT_BASE_URL}/{v}' 
	for k,v in {
	    'acceptance_tests_queue': 'acceptance',
	    'bpmn_item_recommendation_queue': 'bpmnitemrecommendation',
	    'code_generation_queue': 'codegen',
	    'code_repo_recommendation_queue': 'coderepo',
	    'enviroment_queue': 'enviroment'
	}.items()
})