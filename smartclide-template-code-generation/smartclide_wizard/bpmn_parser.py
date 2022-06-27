#!usr/bin/python
#*******************************************************************************
# Copyright (C) 2021-2022 AIR Institute
# 
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
# 
# SPDX-License-Identifier: EPL-2.0
#*******************************************************************************


import re
import uuid
import json
from os import path
from pprint import pprint
import xml.etree.ElementTree as ET


from .model import *
from .config import *
from .xml_tools import *

class BPMNParser:

	def _parse_file(self, file_content):
		# Warning The xml.etree.ElementTree module is not secure against maliciously constructed data.
		#file_content = file_content.replace('bpmn:','')
		root_xml = ET.XML(file_content)
		root_oredict = etree_to_dict(root_xml)

		# Replace urls, normalize dict
		str_root_oredict = re.sub('\{(http)(.*?)\}', '', json.dumps(root_oredict))
		# str to dict - interpreted
		parsed__root_oredict = json.loads(str_root_oredict)
		return parsed__root_oredict

	def _build_schema(self, file_content):	
	 
		workflow_schema = {} 

		# entry is a dict
		for entry in file_content:

			source = entry['id']			
			target = entry['outgoing']

			if source not in workflow_schema:
				workflow_schema[source] = []			

			# if target is none, the node is not connected to any other node
			if target is None:
				workflow_schema[target] = []
				continue

			# target can be a str if is only one, one connection
			if type(target) == str:
				if target not in workflow_schema:
						workflow_schema[target] = []
				workflow_schema[source].append(target)	
			
			# target can be a list also, multiple connections
			else:
				for tg in target:
					if tg not in workflow_schema:
						workflow_schema[tg] = []
					workflow_schema[source].append(tg)

		return workflow_schema

	def _build_node(self, node_content):
		
		# TODO except handle
		try:
			# the node_content is a list -> to dict
			node_content_dict = dict(node_content)   
		except ValueError:
			return None
		
		# get node id
		id_ = node_content_dict['id']
		name = node_content_dict['name']
		n_type = node_content_dict['type']
		params = node_content_dict['params']
		meta = node_content_dict['meta'] if 'meta' in node_content_dict else None

		input_params = [Parameter(name=p["name"], _type=p["dtype"]) for p in params if p['iotype'] == 'input']
		output_type = [Parameter(name=p["name"], _type=p["dtype"]) for p in params if p['iotype'] == 'output']
		output_type = output_type[0] if len(output_type) >=1 else None

		# ensemble node
		n = Node(id_=id_, type_=n_type, name=name, input_params=input_params, output_type=output_type, meta=meta)
	
		return n

	def _build_workflow(self, file_content):
		
		workflows = []

		# get name
		name = None

		# l_nodes is list of nodes but can also contain the information about subprocesses and adhoc-subprocesses that are also list of nodes	
		l_nodes = self._parse_xml(root=file_content['definitions']['process'])

		# Generate the workflow for the subprocesses and adhoc-subprocesses
		for node in l_nodes:
			if node['type'] in ['subprocess', 'adhocsubprocess']:
				s_nodes = node['data'] # list of nodes in the subprocess	
				# build subprocess nodes	
				sub_nodes = {n.id_: n for n in [self._build_node(n) for n in s_nodes]}
				# build subprocess workflow organization
				sub_workflow_schema = self._build_schema(file_content = s_nodes)
				# ensemble subprocess workflow
				workflows.append(Workflow(name=name, nodes=sub_nodes, schema=sub_workflow_schema))					

		# remove the subprocesses and adhoc-subprocesses from l_nodes to build the main process workflow
		l_nodes = [
		    n for n in l_nodes if n['type'] not in ['subprocess', 'adhocsubprocess']
		]
		# build process nodes		
		nodes = {n.id_: n for n in [self._build_node(n) for n in l_nodes]} # dict of nodes
		# build process workflow organization
		workflow_schema = self._build_schema(file_content=l_nodes)
		# ensemble process workflow
		workflows.append(Workflow(name=name, nodes=nodes, schema=workflow_schema))

		return workflows

	def parse(self, file_content) -> List[Workflow]:
		file_content = self._parse_file(file_content=file_content)
		workflow = self._build_workflow(file_content=file_content)
		return workflow

	# lowecase the keys
	def _normalize_dict_tolowercase(self, d):
		return {
			k.lower(): v
		for k,v in d.items()}

	# If a single-element dictionary is iterated, it will give an error, since it is not a list
	def _normalize_dict_entry(self, dict_entry):
		# if we receive a dict -> cast to list
		if type(dict_entry) == dict:
			return [dict_entry]
		return dict_entry
	 
	def _parse_xml(self, root):

		# lower case all the keys in process
		root_process = self._normalize_dict_tolowercase(root)

		bpmn_propertys = []
		bpmn_secuences = []
		bpmn_events = []	
		bpmn_gateways = []
		bpmn_tasks = []
		bpmn_nodes = []
		bpmn_subprocess = []

		for element in root_process:
			# TODO: parse sub/process properties			
			###
			if element == 'property':
				for element_property in self._normalize_dict_entry(root_process['property']):

					try:
						p_name = element_property['name']
					except KeyError:
						p_name = ''

					property_data = {
						'id':               element_property['id'],
						'name':             p_name,
						'itemSubjectRef':   element_property['itemSubjectRef'],
					}
					bpmn_propertys.append(property_data)

			if element == 'sequenceflow':
				for element_secuence in self._normalize_dict_entry(root_process['sequenceflow']):
					# dont get extensionElements
					if 'extensionElements' in element_secuence:
						continue

					secuence_data = {
						'id':               element_secuence['id'],
						'sourceRef':        element_secuence['sourceRef'],
						'targetRef':   		element_secuence['targetRef'],
						#'priority': 		element_secuence['priority'],
					}
					bpmn_secuences.append(secuence_data) 

			###
			# Event handling
			if element in ['boundaryevent', 'startevent', 'endevent', 'intermediatecatchevent']:

				if element == 'intermediatecatchevent':
					for element_intermediatecatchevent in self._normalize_dict_entry(root_process['intermediatecatchevent']):
						intermediatecatchevent_data = {
							'type':				'intermediatecatch_event',
							'id':               element_intermediatecatchevent['id'],
							'name':             element_intermediatecatchevent['name'],
							'subtype':			'timerEventDefinition' if 'timerEventDefinition' in element_intermediatecatchevent else '',
							'subtype_data':		element_intermediatecatchevent['timerEventDefinition']['timeCycle'] if 'timerEventDefinition' in element_intermediatecatchevent else '',
							'outgoing':   		element_intermediatecatchevent['outgoing'] if 'outgoing' in element_intermediatecatchevent else '',
							'incoming':   		element_intermediatecatchevent['incoming'] if 'incoming' in element_intermediatecatchevent else '',
							'params':			[],
						}						
						bpmn_events.append(intermediatecatchevent_data)
						bpmn_nodes.append(intermediatecatchevent_data)

				if element == 'boundaryevent':
					for element_boundaryevent in self._normalize_dict_entry(root_process['boundaryevent']):
						boundaryevent_data = {      
							'type':				'boundary_event',
							'subtype':			'timerEventDefinition' if 'timerEventDefinition' in element_boundaryevent else '',
							'subtype_data':		element_boundaryevent['timerEventDefinition']['timeCycle'] if 'timerEventDefinition' in element_boundaryevent else '',	
							'id':               element_boundaryevent['id'],
							'name':        		element_boundaryevent['name'],
							'outgoing':   		element_boundaryevent['outgoing'] if 'outgoing' in element_boundaryevent else '',
							'incoming':   		element_boundaryevent['incoming'] if 'incoming' in element_boundaryevent else '',
							'params':			[],
						}
						bpmn_events.append(boundaryevent_data)
						bpmn_nodes.append(boundaryevent_data)

				if element == 'startevent':
					# ensure start event is a list
					for element_startevent in self._normalize_dict_entry(root_process['startevent']):
						startevent_data = {      
							'type':				'start_event',    
							'id':               element_startevent['id'],
							'name':        		element_startevent['name'],
							'outgoing':   		element_startevent['outgoing'] if 'outgoing' in element_startevent else '',
							'incoming':   		'', # no incoming for start event
							'params':			[],
						}
						bpmn_events.append(startevent_data)
						bpmn_nodes.append(startevent_data)

				if element == 'endevent':
					# ensure endevent event is a list			
					for element_endevent in self._normalize_dict_entry(root_process['endevent']):

						in_params = []
						if 'dataInput' in element_endevent:
							# TODO signalEventDefinition
							# TODO resolve dataInput dtype, dataInputAssociation?
							in_params_data = {
								'id':           element_endevent['dataInput']['id'],
								'dtype':        '',
								'name':        	element_endevent['dataInput']['name'],
								'iotype':		'input',
							}
							in_params.append(in_params_data)
						
						endevent_data = {
							'type':						'end_event',
							'id':						element_endevent['id'],
							'name':						element_endevent['name'],
							'incoming':					element_endevent['incoming'] if 'incoming' in element_endevent else '',
							'outgoing':   				'', # end event has no outgoing
							'params':					in_params,
						}
						bpmn_events.append(endevent_data) 
						bpmn_nodes.append(endevent_data)
			
			# Gateway handling
			if element in ['parallelgateway', 'exclusivegateway']:
				if element == 'parallelgateway':
					for element_parallelgateway in self._normalize_dict_entry(root_process['parallelgateway']):
						parallelgateway_data = {
							'type':						'condition_parallelgateway',
							'id':						element_parallelgateway['id'],
							'name':						element_parallelgateway['name'],
							'incoming':					element_parallelgateway['incoming'] if 'incoming' in element_parallelgateway else '',
							'outgoing':   				element_parallelgateway['outgoing'] if 'outgoing' in element_parallelgateway else '',
							'condition':   				element_parallelgateway['condition'] if 'condition' in element_parallelgateway else '',
							'gatewayDirection':   		element_parallelgateway['gatewayDirection'] if 'gatewayDirection' in element_parallelgateway else '',
							'params':					[],
						}
						bpmn_gateways.append(parallelgateway_data)
						bpmn_nodes.append(parallelgateway_data)

				if element == 'exclusivegateway':
					for element_exclusive_gateway in self._normalize_dict_entry(root_process['exclusivegateway']):
						exclusivegateway_data = {
							'type':						'condition_exclusivegateway',
							'id':						element_exclusive_gateway['id'],						
							'name':						element_exclusive_gateway['name'],
							'incoming':					element_exclusive_gateway['incoming'] if 'incoming' in element_exclusive_gateway else '',
							'outgoing':   				element_exclusive_gateway['outgoing'] if 'outgoing' in element_exclusive_gateway else '',
							'condition':   				element_exclusive_gateway['condition'] if 'condition' in element_exclusive_gateway else '',
							'gatewayDirection':   		element_exclusive_gateway['gatewayDirection'] if 'gatewayDirection' in element_exclusive_gateway else '',
							'params':					[],
						}
						bpmn_gateways.append(exclusivegateway_data)
						bpmn_nodes.append(exclusivegateway_data)					

			# Subprocess handling
			if element in ['adhocsubprocess', 'subprocess']:
				# a subprocess is still a bmp inside another bmpn
				if element == 'subprocess':
					element_subprocessroot = root_process['subprocess']
					# call again _parse_xml
					subprocess_data = {
						'type':						'subprocess',
						'data':						self._parse_xml(element_subprocessroot),
					}
					bpmn_subprocess.append(subprocess_data)
					bpmn_nodes.append(subprocess_data)
				
				if element == 'adhocsubprocess':
					element_adhocsubprocessroot = root_process['adhocsubprocess']
					# call again _parse_xml
					adhocsubprocess_data = {
						'type':						'adhocsubprocess',
						'data':						self._parse_xml(element_adhocsubprocessroot),
					}
					bpmn_subprocess.append(adhocsubprocess_data)
					bpmn_nodes.append(adhocsubprocess_data)
					
			# It is possible to differentiate between task, usertask, servicetask and scripttask
			# Code is duplicated in case something changes...	
			# 
			# Task handling		
			if element in ['usertask', 'task', 'servicetask', 'scripttask', 'businessruletask']:
				if element == 'businessruletask':
					for element_businessruletask in self._normalize_dict_entry(root_process['task']):
						t_params = []
						businessruletask_data = {
							'type':				'usertask',
							'id':               element_businessruletask['id'],
							'name':        		element_businessruletask['name'],
							'incoming':   		element_businessruletask['incoming'] if 'incoming' in element_businessruletask else '',
							'outgoing':   		element_businessruletask['outgoing'] if 'outgoing' in element_businessruletask else '',
							'params':			t_params,
						}
						bpmn_tasks.append(businessruletask_data)
						bpmn_nodes.append(businessruletask_data)
				# task
				if element == 'task':
					for element_task in self._normalize_dict_entry(root_process['task']):
						t_params = []
						# Data Input
						if 'dataInput' in element_task['ioSpecification']:
							for io_input in self._normalize_dict_entry(element_task['ioSpecification']['dataInput']):							
								in_params_data = {								
									'id':               io_input['id'],
									'dtype':        	io_input['dtype'] if 'dtype' in io_input else '',
									'name':   			io_input['name'],
									#'itemSubjectRef':   io_input['itemSubjectRef'],
									'iotype':			'input',
								}
								t_params.append(in_params_data)
						# Data Output
						if 'dataOutput' in element_task['ioSpecification']:
							for io_output in self._normalize_dict_entry(element_task['ioSpecification']['dataOutput']):							
								out_params_data = {
									'id':               io_output['id'],
									'dtype':        	io_output['dtype'] if 'dtype' in io_output else '',
									'name':   			io_output['name'],
									#'itemSubjectRef':   element_usertask['itemSubjectRef'],
									'iotype':			'output',
								}
								t_params.append(out_params_data) 

						# Unify the data
						task_data = {
							'type':				'task',
							'id':               element_task['id'],
							'name':        		element_task['name'],
							'incoming':   		element_task['incoming'] if 'incoming' in element_task else '',
							'outgoing':   		element_task['outgoing'] if 'outgoing' in element_task else '',
							'params':			t_params,
						}
						bpmn_tasks.append(task_data)
						bpmn_nodes.append(task_data)
				
				if element == 'usertask':
					# usertask
					for element_usertask in self._normalize_dict_entry(root_process['usertask']):
						t_params = []
						if 'dataInput' in element_usertask['ioSpecification']:
							# Data Input
							for io_input in self._normalize_dict_entry(element_usertask['ioSpecification']['dataInput']):
								try:
									p_dtype = io_input['dtype']
								except KeyError:
									p_dtype = None

								in_params_data = {
									'id':               io_input['id'],
									'dtype':        	p_dtype,
									'name':   			io_input['name'],
									#'itemSubjectRef':   io_input['itemSubjectRef'],
									'iotype':			'input',
								}
								t_params.append(in_params_data)
					
						# Data Output
						if 'dataOutput' in element_usertask['ioSpecification']:
							for io_output in self._normalize_dict_entry(element_usertask['ioSpecification']['dataOutput']):
								out_params_data = {
									'id':               io_output['id'],
									'dtype':        	io_output['dtype'] if 'dtype' in io_output else '',
									'name':   			io_output['name'],
									#'itemSubjectRef':   element_usertask['itemSubjectRef'],
									'iotype':			'output',
								}
								t_params.append(out_params_data)
								0
						# Unify the data
						usertask_data = {
							'type':				'usertask',
							'id':               element_usertask['id'],
							'name':        		element_usertask['name'],
							'incoming':   		element_usertask['incoming'] if 'incoming' in element_usertask else '',
							'outgoing':   		element_usertask['outgoing'] if 'outgoing' in element_usertask else '',
							'params':			t_params,
						}
						bpmn_tasks.append(usertask_data)
						bpmn_nodes.append(usertask_data)

				if element == 'servicetask':
					# serviceTask
					for element_servicetask in self._normalize_dict_entry(root_process['servicetask']):
						
						# test.bpmn does not have params
						servicetask_data = {
							'type':				'servicetask',
							'id':               element_servicetask['id'],
							'name':        		element_servicetask['name'],
							'incoming':   		element_servicetask['incoming'] if 'incoming' in element_servicetask else '',
							'outgoing':   		element_servicetask['outgoing'] if 'outgoing' in element_servicetask else '',
							'params':			[],
						}
						bpmn_tasks.append(servicetask_data)
						bpmn_nodes.append(servicetask_data)

				if element == 'scripttask':					
					# scriptTask
					for element_scriptTask in self._normalize_dict_entry(root_process['scripttask']):
						
						scriptTask_data = {
							'type':				'scripttask',
							'id':               element_scriptTask['id'],
							'name':        		element_scriptTask['name'],
							'script':   		element_scriptTask['script'] if 'script' in element_scriptTask else '',
							'incoming':   		element_scriptTask['incoming'] if 'incoming' in element_scriptTask else '',
							'outgoing':   		element_scriptTask['outgoing'] if 'outgoing' in element_scriptTask else '',
							'params':			[],
						}						
						bpmn_tasks.append(scriptTask_data)
						bpmn_nodes.append(scriptTask_data)

		# for process end
		# Replace secuence_flow data references
		for entry in bpmn_nodes:
			for secuence in bpmn_secuences:				
				# Two types of replaces, list or str
				# outgoing
				if 'outgoing' in entry:
					if type(entry['outgoing']) == list:					
						# Replace each entry, cant use for k in v: beacuse is not a reference
						for i in range(len(entry['outgoing'])):   					 								
							if entry['outgoing'][i] == secuence['id']:
								entry['outgoing'][i] = secuence['targetRef']
					elif entry['outgoing'] == secuence['id']:
						entry['outgoing'] = secuence['targetRef']

				if 'incoming' in entry:
					# incoming
					if type(entry['incoming']) == list:
						# Replace each entry, cant use for k in v: beacuse is not a reference
						for i in range(len(entry['incoming'])):   					 								
							if entry['incoming'][i] == secuence['id']:
								entry['incoming'][i] = secuence['sourceRef']
					elif entry['incoming'] == secuence['id']:
						entry['incoming'] = secuence['sourceRef']

		return bpmn_nodes
