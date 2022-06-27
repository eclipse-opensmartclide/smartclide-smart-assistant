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

import uuid
import pprint
import Levenshtein
from re import sub
from typing import List


def camel_case(s):
    """Transforms any text into CamelCase for making suitable the name of the components to the python enviroment
    """
    output = ''.join(x for x in s.title() if x.isalnum())
    return output[0].lower() + output[1:]


class Parameter:

	def __init__(self, name, _type): 
		self._type = _type if _type is not None else 'Object'
		self.name = self._normalize(name)

	def __str__(self):
		return pprint.pformat(self.__dict__())

	def __dict__(self):
		return {'type': self._type, 'name': self.name}

	def _normalize(self, s):
		if s is not None and s:
			s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
			s = ''.join([s[0].lower(), s[1:]])
			s = ''.join(ch for ch in s if ch.isalnum())
		return s


class Node:

	def __init__(self, id_, type_, name, input_params:List[Parameter], output_type:Parameter, meta: dict=None):
		self.id_ = id_
		self.type_ = type_
		self.name = self._normalize(name)
		self.input_params = input_params
		self.output_type = output_type
		self.meta = meta

	def __str__(self):
		return pprint.pformat(self.__dict__())

	def __dict__(self):
		return {'id_': self.id_, 'type_': self.type_, 'name': self.name, 'input_params': [p.__dict__() for p in self.input_params], 'output_type': self.output_type.__dict__(), 'meta': self.meta}

	def fix_input_params(self):		
		for p in self.input_params:
			if p._type is None:
				p._type = 'Object'

	def fix_output_params(self):
		if self.output_type is None:
			self.output_type = Parameter(None, None)
		if self.output_type._type is None:
			self.output_type._type = 'Object'
		if 'gateway' in self.type_:
			self.output_type._type = 'boolean'
		if self.output_type.name is None:
			self.output_type.name = f'{self.name}Output' if self.type_ != 'end_event' else 'result'

	def _normalize(self, s):
		if s is not None and s:
			s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
			s = ''.join([s[0].lower(), s[1:]])
			s = ''.join(ch for ch in s if ch.isalnum())
		return s


class Workflow:

	def __init__(self, name:str, nodes:List[Node], schema:dict):
		self.id_ = str(uuid.uuid4())
		self.name = name
		self.nodes = nodes
		self.schema = schema
		self._fix_node_names()
		self.param_mappings = self.generate_mappings()

	def __str__(self):
		return pprint.pformat(self.__dict__())

	def __dict__(self):
		return {'id_': self.id_, 'name': self.name, 'nodes': [n.__dict__() for n in self.node_list()], 'schema': self.schema, 'param_mappings': self.param_mappings}

	def _fix_node_names(self):
		method_id = 0
		for node in self.node_list():
			if node.name is None or not node.name:
				if node.id_ is None:
					node.name = f'method{method_id}'
					method_id += 1
				else:
					node.name = camel_case(node.id_)
			node.fix_output_params()
			node.fix_input_params()

	def node_list(self):
		return list(set(self.nodes.values()))

	def start_nodes(self):
		return [v for k,v in self.nodes.items() if v.type_ == 'start_event']

	def get_children(self, node_id):
		return [n for n in self.node_list() if n.id_ in self.schema[node_id]]

	def generate_mappings(self):
		# generate object with the available parameters
		node_queue = []
		avialable_parameters = {}

		# initialize start nodes
		node_queue = [{'node': n, 'inputs': []} for n in self.start_nodes()]

		for n in node_queue:
			# generate entry in obj
			avialable_parameters[n['node'].id_] = n['inputs']
			# expand children
			for c in self.get_children(n['node'].id_):
				node_queue.append({'node': c, 'inputs': n['inputs'] + [n['node'].output_type]})

		# for each node, generate the mappings
		mappings = {}
		for node in self.node_list():

			if node.input_params is None:
				continue

			mappings[node.id_] = {}

			available_node_parameters = avialable_parameters[node.id_] if node.id_ in avialable_parameters else []

			for parameter in node.input_params:

				# select inputs of same type or Object type
				filtered_inputs = [i for i in available_node_parameters if i._type in [parameter._type,'Object']]

				# if no parameters, put comment of completion
				if not filtered_inputs:
					mappings[node.id_][parameter.name] = f'/*TODO: {parameter._type} {parameter.name}*/'
					continue

				# select the ones with same parameters
				filtered_inputs_2 = [i for i in available_node_parameters if i._type == parameter._type]

				# if is the case select as candidates for name recognition
				name_recognition_candidates = filtered_inputs if not filtered_inputs_2 else filtered_inputs_2

				# select the ones with similar name
				name_recognition_candidates = [{"p": p, "score": Levenshtein.distance(parameter.name, p.name)} for p in name_recognition_candidates]
				name_recognition_candidates.sort(key=lambda x: x["score"])
				candidate = name_recognition_candidates[0]

				# generate param
				mappings[node.id_][parameter.name] = candidate["p"].name
				if candidate["p"]._type == 'Object':
					mappings[node.id_][parameter.name] = f'({parameter._type}){mappings[node.id_][parameter.name]}'

		return mappings
