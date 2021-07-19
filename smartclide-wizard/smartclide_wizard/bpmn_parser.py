#!usr/bin/python

# Copyright 2021 AIR Institute
# See LICENSE for details.


import uuid
import json
from pprint import pprint
import xml.etree.ElementTree as ET
from os import path


class BPMNParser:

	def _read_file_content(self, path):
		with open(path, 'r') as f:
			file_content =  f.read()
		return file_content

	def _map_to_json(self, data):
		xml_tree = ET.fromstring(data)
		elements = xml_tree.findall('.//mxCell')
		for e in elements:
			yield {
				'id': e.get('id'),
				'name': e.get('value'),
				'source': e.get('source'),
				'target': e.get('target'),
				'style': 'normal' if e.get('style') is None or 'dashed' not in e.get('style') else 'special'
			}

	def _prepare_arrow(self, node):

		# get all node children (arrows) and classify
		normal_arrows = [node for node in node['children'] if node['info']['data']['style'] == 'normal']
		special_arrows = [node for node in node['children'] if node['info']['data']['style'] == 'special']

		# output topic
		output_topic_name = f"{node['info']['data']['name']}_output"

		# set name for normal arrows
		for arrow_id, arrow in enumerate(normal_arrows):
			arrow['info']['data']['topic_name'] = output_topic_name
			arrow['info']['data']['associated_consumer_id'] = f"{node['info']['data']['name']}_{arrow_id}"

		# set name for shared arrows
		shared_arrows_group = f"shared_group"
		for arrow in special_arrows:
			arrow['info']['data']['topic_name'] = output_topic_name
			arrow['info']['data']['associated_consumer_id'] = shared_arrows_group

	def _build_connections(self, node):

		# calculate input topics
		for parent in node['parents']:
			node['info']['input_topics'].append(parent['info']['data']['topic_name'])
		node['info']['input_topics'] = list(set(node['info']['input_topics']))

		# calculate input topics
		for child in node['children']:
			node['info']['output_topics'].append(child['info']['data']['topic_name'])
		node['info']['output_topics'] = list(set(node['info']['output_topics']))

		# calculate consumer_id
		consumer_id = None
		for parent in node['parents']:
			if consumer_id != 'shared_group':
				consumer_id = parent['info']['data']['associated_consumer_id']
		node['info']['consumer_id'] = consumer_id

	def _build_component(self, node):
		return {
			'name': node['info']['data']['name'],
			'consumer_id': node['info']['consumer_id'],
			'input_topics': node['info']['input_topics'],
			'output_topics': node['info']['output_topics']
		}

	def _build_component_list(self, data):

		# build node list
		nodes = [{
			'parents': [],
			'children': [],
			'info': {
				'data': node,
				'input_topics': [],
				'output_topics': [],
				'consumer_id': None
			},
			'is_component': (node['name'] != None)
		} for node in data]

		# build graph
		for node in nodes:
			for n in nodes:
				# check if node is in fathers
				if n['info']['data']['id'] == node['info']['data']['source']:
					if node not in n['children']:
						n['children'].append(node)
					if n not in node['parents']:
						node['parents'].append(n)
				# check if node is in children
				if n['info']['data']['id'] == node['info']['data']['target']:
					if node not in n['parents']:
						n['parents'].append(node)
					if n not in node['children']:
						node['children'].append(n)

		# prepare each arrow to have the previous node name
		for node in nodes:
			if node['is_component']:
				self._prepare_arrow(node=node)

		# build connections between components
		for node in nodes:
			if node['is_component']:
				self._build_connections(node=node)

		# retrieve components
		components = [self._build_component(node) for node in nodes if node['is_component']]

		pprint(components)

		return components

	def parse(self, path, **kwargs):
		file_content = self._read_file_content(path=path)
		json_content = list(self._map_to_json(data=file_content))
		component_list = self._build_component_list(data=json_content)
		return component_list