#!usr/bin/python

# Copyright 2021 AIR Institute
# See LICENSE for details.


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

			# target can be a str if is only one
			if type(target) == str:
				if target not in workflow_schema:
						workflow_schema[target] = []
				workflow_schema[source].append(target)	
			
			# target can be a list also
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
			None
		
		# get node id
		id_ = node_content_dict['id']
		name = node_content_dict['name']
		params = node_content_dict['params']
		n_type = node_content_dict['type']
		condition = name if n_type == 'condition_exclusivegateway' else None

		## TODO: differentiate the types of nodes> n_type
			# tasks: task, usertask and servicetask
			# startevents
			# endevents
			# conditions!

		input_params = [Parameter(name=p["name"], _type=p["dtype"]) for p in params if p['iotype'] == 'input']
		output_type = [Parameter(name=p["name"], _type=p["dtype"]) for p in params if p['iotype'] == 'output']
		output_type = output_type[0] if len(output_type) >=1 else None


		# ensemble node
		n = Node(id_=id_, name=name, input_params=input_params, output_type=output_type, condition=condition)
	
		return n

	def _build_workflow(self, file_content):

		# get name
		name = None

		# get the nodes
		l_nodes = self._parse_xml(root=file_content)

		# build nodes		
		nodes = {n.id_: n for n in [self._build_node(n) for n in l_nodes]}

		# build workflow organization
		workflow_schema = self._build_schema(file_content=l_nodes)
		
		# ensemble workflow
		w = Workflow(name=name, nodes=nodes, schema=workflow_schema)

		return w

	def parse(self, file_content):
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

		root_process = root['definitions']['process']

		# lower case all the keys in process
		root_process = self._normalize_dict_tolowercase(root_process)

		bpmn_propertys = []
		bpmn_secuences = []
		bpmn_startevents = []
		bpmn_endevents = []		
		bpmn_exclusivegateway = []
		bpmn_tasks = []
		bpmn_nodes = []

		for element in root_process:

			if element == 'property':
				for element_property in self._normalize_dict_entry(root_process['property']):
					property_data = {
						'id':               element_property['id'],
						'name':             element_property['name'],
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

			if element == 'startevent':
				# ensure start event is a list
				for element_startevent in self._normalize_dict_entry(root_process['startevent']):
					startevent_data = {      
						'type':				'start_event',    
						'id':               element_startevent['id'],
						'name':        		element_startevent['name'],
						'outgoing':   		element_startevent['outgoing'],
						'incoming':   		'',
						'params':			[],
					}
					bpmn_startevents.append(startevent_data)
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
						'incoming':					element_endevent['incoming'],
						'outgoing':   				'',
						'params':					in_params,
					}
					bpmn_endevents.append(endevent_data) 
					bpmn_nodes.append(endevent_data)

			if element == 'exclusivegateway':
				for element_exclusive_gateway in self._normalize_dict_entry(root_process['exclusivegateway']):
					# add all to the list
					bpmn_exclusivegateway.append(element_exclusive_gateway)

					exclusivegateway_data = {
						'type':						'condition_exclusivegateway',
						'id':						element_exclusive_gateway['id'],						
						'name':						element_exclusive_gateway['name'],
						'incoming':					element_exclusive_gateway['incoming'],
						'outgoing':   				element_exclusive_gateway['outgoing'],
						'condition':   				element_exclusive_gateway['condition'],
						'gatewayDirection':   		element_exclusive_gateway['gatewayDirection'],
						'params':					[],
					}
					bpmn_nodes.append(exclusivegateway_data)					

			# It is possible to differentiate between task, usertask and servicetask
			# Code is duplicated in case something changes...
			if element in ['usertask', 'task', 'servicetask']:
				# task
				if element == 'task':
					for element_task in self._normalize_dict_entry(root_process['task']):

						t_params = []
						# Data Input
						for io_input in self._normalize_dict_entry(element_task['ioSpecification']['dataInput']):							
							in_params_data = {								
								'id':               io_input['id'],
								'dtype':        	io_input['dtype'],
								'name':   			io_input['name'],
								#'itemSubjectRef':   io_input['itemSubjectRef'],
								'iotype':			'input',
							}
							t_params.append(in_params_data)
						# Data Output
						for io_output in self._normalize_dict_entry(element_task['ioSpecification']['dataOutput']):							
							out_params_data = {
								'id':               io_output['id'],
								'dtype':        	io_output['dtype'],
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
							'incoming':   		element_task['incoming'],
							'outgoing':   		element_task['outgoing'],
							'params':			t_params,
						}
						bpmn_tasks.append(task_data)
						bpmn_nodes.append(task_data)
				
				if element == 'usertask':
					# usertask
					for element_usertask in self._normalize_dict_entry(root_process['usertask']):

						t_params = []
						# Data Input
						for io_input in self._normalize_dict_entry(element_usertask['ioSpecification']['dataInput']):
							in_params_data = {
								'id':               io_input['id'],
								'dtype':        	io_input['dtype'],
								'name':   			io_input['name'],
								#'itemSubjectRef':   io_input['itemSubjectRef'],
								'iotype':			'input',
							}
							t_params.append(in_params_data)
						# Data Output
						for io_output in self._normalize_dict_entry(element_usertask['ioSpecification']['dataOutput']):
							out_params_data = {
								'id':               io_output['id'],
								'dtype':        	io_output['dtype'],
								'name':   			io_output['name'],
								#'itemSubjectRef':   element_usertask['itemSubjectRef'],
								'iotype':			'output',
							}
							t_params.append(out_params_data) 

						# Unify the data
						usertask_data = {
							'type':				'usertask',
							'id':               element_usertask['id'],
							'name':        		element_usertask['name'],
							'incoming':   		element_usertask['incoming'],
							'outgoing':   		element_usertask['outgoing'],
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
							'incoming':   		element_servicetask['incoming'],
							'outgoing':   		element_servicetask['outgoing'],
							'params':			[],
						}
						bpmn_tasks.append(servicetask_data)
						bpmn_nodes.append(servicetask_data)
		# for process end

		# Replace secuence_flow data references
		for entry in bpmn_nodes:
			for secuence in bpmn_secuences:
				# Two types of replaces, list or str
				# outgoing
				if type(entry['outgoing']) == list:					
					# Replace each entry, cant use for k in v: beacuse is not a reference
					for i in range(len(entry['outgoing'])):   					 								
						if entry['outgoing'][i] == secuence['id']:
							entry['outgoing'][i] = secuence['targetRef']
				else:
					if entry['outgoing'] == secuence['id']:
						entry['outgoing'] = secuence['targetRef']
				# incoming
				if type(entry['incoming']) == list:
					# Replace each entry, cant use for k in v: beacuse is not a reference
					for i in range(len(entry['incoming'])):   					 								
						if entry['incoming'][i] == secuence['id']:
							entry['incoming'][i] = secuence['sourceRef']
				else:
					if entry['incoming'] == secuence['id']:
						entry['incoming'] = secuence['sourceRef']

		return bpmn_nodes