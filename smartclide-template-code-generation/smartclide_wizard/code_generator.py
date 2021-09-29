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
		name = workflow.nam if workflow.name is not None else 'GeneratedClass'
		return f'public class {name} ' + '{'

	def _gen_end(self):
		return '}'

	def _gen_metohd(self, node):

		output_type = node.output_type._type if node.output_type is not None else 'void'
		input_params = node.input_params if node.input_params is not None else []

		params_str = ','.join(f'{p._type} {p.name}' for p in input_params)

		method = [
			f'public {output_type} {node.name}({params_str})' + '{'
			,'\t//TODO: complete method'
			,'}'
		]

		return method

	def _gen_conditional_method(self, node):
		input_params = node.input_params if node.input_params is not None else []
		params_str = ','.join(f'{p.type} {p.name}' for p in input_params)

		method = [
			f'public boolean {node.name}({params_str})' + '{'
			,f'\t//TODO: write condition "{node.condition}"'
			,f'\treturn false;'
			,'}'
		]

		return method

	def _gen_main_method(self, workflow):

		def f(workflow, node):

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

			# generate code for first nodes
			children_code = [f(workflow, n) for n in workflow.get_children(node.id_)]
			
			# append children code and generate conditionals
			if len(children_code) == 1:
				code.extend(children_code[0])
			if len(children_code) > 1:
				children_code = [ [self._tabulate_line(l) for l in c] for c in children_code]
				code.append(f'if({node.output_type.name})'+ '{' + (f'/*{node.condition}*/' if node.condition is not None else ''))
				code.extend(children_code[0])
				code.append('}else{')
				code.extend(children_code[1])
				code.append('}')

			return code

		# generate code for each first node
		method_pieces = [f(workflow, node) for node in workflow.start_nodes()]
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

	def generate(self, workflow: Workflow):

		start = self._gen_start(workflow)
		end = self._gen_end()
		main_method = self._gen_main_method(workflow)
		methods = [self._gen_conditional_method(n) if n.condition is not None else self._gen_metohd(n) for n in workflow.node_list()]

		ensembled_class = self._ensemble_class(start=start, methods=methods, main_method=main_method, end=end)

		return ensembled_class