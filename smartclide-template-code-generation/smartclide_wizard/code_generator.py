#!usr/bin/python

# Copyright 2021 AIR Institute
# See LICENSE for details.


from .model import *

from re import sub


class CodeGenerator:

	def generate(self, workflow: Workflow):
		pass


class JavaCodeGenerator(CodeGenerator):

	def _tabulate_line(self, line, num_indents=1):
		tabulation = ''.join(['\t' for i in range(num_indents)])
		return f"{tabulation}{line}"

	def _gen_start(self, workflow):
		name = workflow.name if workflow.name is not None else 'GeneratedClass'
		return f'public class {name} ' + '{'

	def _gen_end(self):
		return '}'

	def _gen_method(self, node):
		if node.type_ == 'start_event':
			return self._gen_unknown_method(node)
		elif node.type_ == 'end_event':
			return self._gen_unknown_method(node)
		elif node.type_ == 'task':
			return self._gen_task_metohd(node)
		elif node.type_ == 'scripttask':
			return self._gen_script_task(node)
		elif node.type_ == 'servicetask':
			return self._gen_service_task(node)
		elif node.type_ == 'usertask':
			return self._gen_user_task(node)
		elif node.type_ == 'adhocsubprocess':
			return self._gen_adhocsubprocess(node)
		elif 'gateway' in node.type_:
			return self._gen_conditional_node(node)
		elif 'event' in node.type_:
			return self._gen_event(node)
		else:
			raise Exception(f'Not found method: {node.type_}')
			return self._gen_unknown_method(node)


#########################################################################################################################

	def __gen_signature(self, node):

		output_type = node.output_type._type if node.output_type is not None and node.output_type._type is not None else 'void'
		input_params = node.input_params if node.input_params is not None else []

		for p in input_params:
			if p._type is None:
				p._type = 'Object'

		params_str = ','.join(f'{p._type } {p.name}' for p in input_params)

		signature = f'public {output_type} {node.name.strip()}({params_str})'
		
		return signature

	def _gen_task_metohd(self, node):
		
		signature = self.__gen_signature(node)
		
		method = [
			f'{signature}{{'
			,'\t//TODO: complete method'
			,'}'
		]

		return method

	def _gen_user_task(self, node):
		
		signature = self.__gen_signature(node)
		
		method = [
			f'{signature}{{'
			,f'\t//TODO: request user "{node.name}"'
			,'}'
		]

		return method

	def _gen_script_task(self, node):
		
		signature = self.__gen_signature(node)
		
		method = [
			f'{signature}{{'
			,f'\t//TODO: run script "{node.name}"'
			,'}'
		]

		return method

	def _gen_service_task(self, node):
		
		signature = self.__gen_signature(node)
		
		method = [
			f'{signature}{{'
			,f'\t//TODO: request data from service "{node.name}"'
			,'}'
		]

		return method

	def _gen_bussines_rule_task(self, node):

		signature = self.__gen_signature(node)
		
		method = [
			f'{signature}{{'
			,f'\t//TODO: introduce bussines rule to engine "{node.name}"'
			,'}'
		]

		return method

	def _gen_receive_task(self, node):

		signature = self.__gen_signature(node)
		
		method = [
			f'{signature}{{'
			,f'\t//TODO: wait for task "{node.name}"'
			,'}'
		]

		return method

	def _gen_manual_task(self, node):

		signature = self.__gen_signature(node)
		
		method = [
			f'{signature}{{'
			,f'\t//TODO: perform manual task "{node.name}"'
			,'}'
		]

		return method

	def _gen_email_task(self, node):

		signature = self.__gen_signature(node)
		
		method = [
			f'{signature}{{'
			,f'\t//TODO: send email to "{node.name}"'
			,'}'
		]

		return method

	def _gen_camel_task(self, node):

		signature = self.__gen_signature(node)
		
		method = [
			f'{signature}{{'
			,f'\t//TODO: receive from apache camel "{node.name}"'
			,'}'
		]

		return method

	def _gen_shell_task(self, node):

		signature = self.__gen_signature(node)
		
		method = [
			f'{signature}{{'
			,f'\t//TODO: perform shell task "{node.name}"'
			,'}'
		]

		return method

	def _gen_http_task(self, node):

		signature = self.__gen_signature(node)
		
		method = [
			f'{signature}{{'
			,f'\t//TODO: request via HTTP to "{node.name}"'
			,'}'
		]

		return method

	def _gen_mule_task(self, node):

		signature = self.__gen_signature(node)
		
		method = [
			f'{signature}{{'
			,f'\t//TODO: run Mule ESB process "{node.name}"'
			,'}'
		]

		return method

	def _gen_deccision_task(self, node):

		signature = self.__gen_signature(node)
		
		method = [
			f'{signature}{{'
			,f'\t//TODO: query DBM table "{node.name}"'
			,'}'
		]

		return method

	def _gen_event(self, node):

		signature = self.__gen_signature(node)		
		event_subtype = node.meta["subtype"] if node.meta is not None and "subtype" in node.meta else node.name

		method = [
			f'{signature}{{'
			,f'\t//TODO: perform event "{event_subtype}"'
			,'}'
		]

		return method

	def _gen_conditional_node(self, node):

		signature = self.__gen_signature(node)
		output_type = node.output_type._type if node.output_type is not None else 'void'
		signature = signature.replace(output_type, 'boolean')

		method = [
			f'{signature}{{'
			,f'\t//TODO: perform check "{node.name}"'
			,'}'
		]

		return method

	def _gen_unknown_method(self, node):

		signature = self.__gen_signature(node)
		
		method = [
			f'{signature}{{'
			,f'\t//TODO'
			,'}'
		]

		return method

#########################################################################################################################

	def _gen_exclusive_gateway(self, node, children_codes, children_nodes):

		code = []

		code.append(f'if({node.output_type.name}){{')
		code.extend(children_codes[0])

		if len(children_codes) > 2:
			for child_code in children_nodes[1:-1]:
				code.append('}else if(/*TODO: condition*/){')
				code.extend(child_code)

		code.append('}else{')
		code.extend(children_codes[-1])
		code.append('}')

		return code

	def _gen_inclusive_gateway(self, node, children_codes, children_nodes):

		code = []

		for child_code in children_nodes[1:-1]:
			code.append('if(/*TODO: condition*/){')
			code.extend(child_code)
			code.append('}')

		return code

	def _gen_paralell_or_event_gateway(self, node, children_codes, children_nodes):

		code = []
		conditions = ' && '.join([node.output_type.name for node in children_nodes]) 
		code.append('while(1){')
		for child_code in children_codes:
			code.extend(['\t' + c for c in child_code])
		code.append(f'\tif({conditions}){{')
		code.append('\t\tbreak;')
		code.append('\t}')
		code.append('}')

		return code

#########################################################################################################################

	def _gen_subprocess(self, workflow, start_nodes, end_nodes_=None):

		if end_nodes_ is None:
			end_nodes_ = [n for n in workflow.node_list() if n.type_ == 'end_node']

		def f(workflow, node, end_nodes=end_nodes_):

			# output code
			code = []

			# generate parameters string
			if node.input_params is not None:
				params_str = [f'{workflow.param_mappings[node.id_][p.name]}' for p in node.input_params]
				params_str = ', '.join(params_str)
			else:
				params_str = '/*TODO: Complete parameters*/'

			# generate node invocation
			method_invocation = f'{node.output_type._type} {node.output_type.name} = this.{node.name}({params_str});'
			code.append(method_invocation)

			# generate code for children
			children_code = [f(workflow, n) for n in workflow.get_children(node.id_)]

			# if node is in end nodes, finish
			if node not in end_nodes and children_code:
				# append children code if is a single one
				if 'gateway' not in node.type_:
					code.extend(children_code[0])
				else:
					children_code = [ [self._tabulate_line(l) for l in c] for c in children_code]
					if node.type_ in ['condition_parallelgateway', 'event_parallelgateway']:
						# its neccesary to take only first order children
						first_order_children = workflow.get_children(node.id_)
						children_code = [f(workflow, n, end_nodes=[n]) for n in workflow.get_children(node.id_)]
						code.extend(self._gen_paralell_or_event_gateway(node, children_code, workflow.get_children(node.id_)))
						# then apply it to the children of children
						for c in first_order_children:
							children_codes = [f(workflow, n) for n in workflow.get_children(c.id_)]
						# apply it
						for child_code in children_codes:
							code.extend(child_code)
					elif node.type_  == 'condition_exclusivegateway':
						code.extend(self._gen_exclusive_gateway(node, children_code, workflow.get_children(node.id_)))
					elif node.type_  == 'condition_inclusivegateway':
						code.extend(self._gen__inclusive_gateway(node, children_code, workflow.get_children(node.id_)))
					else:
						raise Exception(f'Not found node gateway {node.type_}')

			return code

		# generate code for each first node
		method_pieces = [f(workflow, node) for node in start_nodes]
		method_pieces = [[self._tabulate_line(l) for l in code] for code in method_pieces]

		# ensenmble
		method = []
		method.append(f'public void run()' + '{')
		for piece in method_pieces:
			method.extend(piece)
		method.append('}')

		return method

	def _ensemble_class(self, start, methods, main_method, end):
		serialized_methods = '\n\n'.join(['\n'.join([self._tabulate_line(l) for l in m]) for m in methods])
		serialized_main_method = '\n'.join([self._tabulate_line(l) for l in main_method])
		serialized_class = '\n\n'.join([start, serialized_methods, serialized_main_method, end])
		return serialized_class

	def generate(self, workflows: List[Workflow]) -> List[str]:

		for workflow in workflows:

			print('\n\n\n\n\n\n')
			print('='.join(['' for _ in range(80)]))
			print(workflow)
			print('\n\n\n\n\n\n')

			start = self._gen_start(workflow)
			end = self._gen_end()
			main_method = self._gen_subprocess(workflow=workflow, start_nodes=workflow.start_nodes())
			methods = list(filter(None, [self._gen_method(n) for n in workflow.node_list()]))

			ensembled_class = self._ensemble_class(start=start, methods=methods, main_method=main_method, end=end)

			print(ensembled_class)

			yield ensembled_class