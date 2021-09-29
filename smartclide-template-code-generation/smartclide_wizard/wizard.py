#!usr/bin/python

# Copyright 2021 AIR Institute
# See LICENSE for details.


import os
from . import BPMNParser, JavaCodeGenerator


class Wizard:

	@classmethod
	def generate(cls, input_file):

		# read input file
		with open(input_file,'r') as f:
			file_content = f.read()

		# parse diagram
		parser = BPMNParser()
		workflow = parser.parse(file_content)

		# build components
		code_generator = JavaCodeGenerator()
		class_generated = code_generator.generate(workflow)
		print(class_generated)
